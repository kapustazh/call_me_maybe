"""Textual pipeline UI: streaming log, branding sidebar, quit confirmation."""

from __future__ import annotations

import signal
import sys
import threading
from collections.abc import Callable
from itertools import cycle
from types import FrameType

from rich.align import Align
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key
from textual.widgets import Static

# Layout
_SIDEBAR_WIDTH_CELLS_DEFAULT = 20
_SIDEBAR_BRANDING_INNER_HEIGHT = 5

# Log stream → UI thread (batches)
_LOG_STREAM_CHARS_PER_UI_BATCH = 42
_LOG_SCROLL_COALESCE_MIDLINE_CHUNKS = True

# Midline scroll coalesce: poll interval when scroll deferred (seconds)
_SCROLL_COALESCE_TICK_SEC = 0.08
_CAT_ANIMATION_TICK_SEC = 0.3

# Rich styles (sidebar filler behind signature / cat)
_SIDEBAR_BRANDING_FILL_STYLE = "rgb(48,48,48)"

# Copy + assets
_QUIT_CONFIRMATION_BAR_TEXT = (
    " Press q or Esc again to quit." " Any other key closes this bar. "
)

_CAT_SPRITE_FRAME_LINES: tuple[tuple[str, ...], ...] = (
    (r" /\_/\ ", r"( o.o )", r" > ^ < "),
    (r" /\_/\ ", r"( -.- )", r" > ^ < "),
    (r" /\_/\ ", r"( o.o )", r" > ~ < "),
    (r" /\_/\ ", r"( ^.^ )", r" > ^ < "),
)

KAPUSTAZH_SIGNATURE = "kapustazh"
_TITLE_CALL_ME_MAYBE = "Call me maybe..."

# Log color pair ids → Rich style.
_LOG_COLOR_PAIR_TO_RICH_STYLE: dict[int, str] = {
    1: "cyan",
    2: "green",
    3: "red",
}
PAIR_INFO = 1
PAIR_OK = 2
PAIR_ERR = 3


# PipelineUIRenderer — public facade used by `pipeline.py`


class RenderError(RuntimeError):
    """Raised when the Textual renderer cannot start or run."""


class PipelineUIRenderer:
    """Stream log lines into the Textual app and block until quit.

    Public facade used by ``pipeline.py``. Methods that touch the UI hop
    to the Textual event loop via ``App.call_from_thread``; the log
    content is held as a single Rich ``Text`` accumulator.

    Attributes:
        _app: Owning :class:`PipelineApp`.
        _rendered: Accumulated styled log text.
    """

    def __init__(self, app: PipelineApp) -> None:
        """Bind the renderer to a :class:`PipelineApp`.

        Args:
            app: Hosting Textual app.
        """
        self._app = app
        self._rendered = Text()

    @classmethod
    def run_interactive(
        cls,
        worker: Callable[[PipelineUIRenderer], None],
        *,
        right_width: int = _SIDEBAR_WIDTH_CELLS_DEFAULT,
    ) -> None:
        """Run the Textual app for ``worker`` and block until it exits.

        Hooks ``SIGINT`` to mark ``quit_confirmed`` and restores the prior
        signal handler on exit.

        Args:
            worker: Callable invoked with the live renderer.
            right_width: Sidebar width in cells.

        Raises:
            RenderError: ``stdin``/``stdout`` is not a TTY, or the app
                raises while running.
        """
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise RenderError("stdin/stdout is not a TTY")
        app = PipelineApp(worker, right_width=right_width)
        prev = signal.getsignal(signal.SIGINT)

        def _sigint(_: int, __: FrameType | None) -> None:
            app.quit_confirmed.set()

        signal.signal(signal.SIGINT, _sigint)
        try:
            app.run()
        except Exception as exc:  # pragma: no cover
            raise RenderError(str(exc)) from exc
        finally:
            signal.signal(signal.SIGINT, prev)

    def _append_log(
        self,
        chunk: str,
        *,
        style: str,
        tail: bool = False,
    ) -> None:
        """Append a log chunk on the Textual event-loop thread.

        Scrolls now (newline or tail chunk) or marks ``_scroll_pending``
        when ``_LOG_SCROLL_COALESCE_MIDLINE_CHUNKS`` is enabled.

        Args:
            chunk: Text fragment to append.
            style: Rich style name applied to the chunk.
            tail: ``True`` for the final chunk of a stream; forces a
                scroll regardless of coalescing.
        """
        if not chunk:
            return
        self._rendered.append(chunk, style=style)
        self._app._sync_log_update(self._rendered.copy())
        if tail or "\n" in chunk or not _LOG_SCROLL_COALESCE_MIDLINE_CHUNKS:
            self._app._scroll_log_now()
        else:
            self._app._scroll_pending = True

    def log_stream(self, text: str, *, pair: int) -> None:
        """Stream ``text`` to the log in fixed-size batches.

        Each batch hops to the UI thread via ``call_from_thread``; batch
        size is ``_LOG_STREAM_CHARS_PER_UI_BATCH``.

        Args:
            text: Source text. Empty input is a no-op.
            pair: Color pair id; see ``_LOG_COLOR_PAIR_TO_RICH_STYLE``.
        """
        if not text:
            return
        style = _LOG_COLOR_PAIR_TO_RICH_STYLE.get(pair, "")
        n = _LOG_STREAM_CHARS_PER_UI_BATCH
        length = len(text)
        pos = 0
        while pos < length:
            end = min(pos + n, length)
            self._app.call_from_thread(
                self._append_log,
                text[pos:end],
                style=style,
                tail=end >= length,
            )
            pos = end

    def log_info_stream(self, text: str) -> None:
        """Stream ``text`` styled as info (cyan)."""
        self.log_stream(text, pair=PAIR_INFO)

    def log_ok_stream(self, text: str) -> None:
        """Stream ``text`` styled as ok (green)."""
        self.log_stream(text, pair=PAIR_OK)

    def log_err_stream(self, text: str) -> None:
        """Stream ``text`` styled as error (red)."""
        self.log_stream(text, pair=PAIR_ERR)

    def wait_until_quit(self) -> None:
        """Block until the user confirms quit (q/Esc twice or SIGINT)."""
        self._app.call_from_thread(self._app.arm_quit_wait)
        self._app.quit_confirmed.wait()


# PipelineApp (Textual TUI)


class PipelineApp(App[None]):
    """Textual app: streaming log pane plus branding sidebar.

    Layout: docked right column (``#title_box`` + ``#branding``) plus log
    pane in ``VerticalScroll``.     Cat sprite animates on
    ``_CAT_ANIMATION_TICK_SEC``. When
    ``_LOG_SCROLL_COALESCE_MIDLINE_CHUNKS`` is True, a
    ``_SCROLL_COALESCE_TICK_SEC`` interval drains deferred midline scrolls.

    Attributes:
        quit_confirmed: Set when the user confirms quit.
    """

    CSS = f"""
    #right_column {{
        dock: right;
        width: {_SIDEBAR_WIDTH_CELLS_DEFAULT};
        height: 100%;
    }}
    #title_box {{
        height: auto;
        min-height: 1;
        padding: 0 1;
        background: $surface;
        color: $foreground;
        border: solid $boost;
    }}
    #branding {{
        width: 100%;
        height: 1fr;
        background: $surface;
        color: $foreground;
        border: solid $boost;
    }}
    #log_pane {{
        width: 1fr;
        height: 100%;
        border-top: solid $boost;
        border-left: solid $boost;
        border-bottom: solid $boost;
    }}
    #log_container {{
        width: 100%;
        height: 100%;
        background: transparent;
        overflow-y: auto;
    }}
    #log {{
        width: 100%;
        height: auto;
        padding: 0 1;
        background: transparent;
    }}
    #quit_hint {{
        dock: bottom;
        height: auto;
        background: $surface;
        color: $warning;
        display: none;
    }}
    """

    def __init__(
        self,
        worker: Callable[[PipelineUIRenderer], None],
        *,
        right_width: int = _SIDEBAR_WIDTH_CELLS_DEFAULT,
    ) -> None:
        """Initialize app state; widgets are created in :meth:`compose`.

        Args:
            worker: Callable that streams content into the renderer.
            right_width: Sidebar width in cells (clamped to ``>= 18``).
        """
        super().__init__()
        self._worker_cb = worker
        self._right_width = max(18, int(right_width))
        self._renderer: PipelineUIRenderer | None = None
        self._cat_iter = cycle(_CAT_SPRITE_FRAME_LINES)
        self._cat_frame = next(self._cat_iter)
        self.quit_confirmed = threading.Event()
        self._quit_waiting = False
        self._quit_armed = False
        self._scroll_pending: bool = False

    def compose(self) -> ComposeResult:
        """Yield the widget tree (title, branding, quit hint, log pane)."""
        with Vertical(id="right_column"):
            yield Static(
                Align.right(
                    Text(_TITLE_CALL_ME_MAYBE, style="bold cyan"),
                ),
                id="title_box",
                markup=False,
            )
            yield Static(
                self._branding_renderable(
                    inner_width=max(6, self._right_width - 2),
                ),
                id="branding",
                markup=False,
            )
        yield Static(
            _QUIT_CONFIRMATION_BAR_TEXT.strip(),
            id="quit_hint",
            markup=False,
        )
        with Container(id="log_pane"):
            with VerticalScroll(id="log_container"):
                yield Static(
                    id="log",
                    markup=False,
                    expand=True,
                    shrink=True,
                )

    def on_mount(self) -> None:
        """Lock sidebar width, schedule timers, start worker."""
        rw = self._right_width
        rc = self.query_one("#right_column", Vertical)
        rc.styles.width = rw
        rc.styles.min_width = rw
        rc.styles.max_width = rw
        self._renderer = PipelineUIRenderer(self)
        if _LOG_SCROLL_COALESCE_MIDLINE_CHUNKS:
            self.set_interval(
                _SCROLL_COALESCE_TICK_SEC,
                self._tick_scroll_coalesce,
            )
        self.set_interval(_CAT_ANIMATION_TICK_SEC, self._tick_cat)
        self.call_after_refresh(self._paint_branding)
        self.call_after_refresh(self._start_worker)

    def _start_worker(self) -> None:
        """Run the user worker in an exclusive Textual thread worker."""
        self.run_worker(
            self._pipeline_worker,
            thread=True,
            exclusive=True,
        )

    def _pipeline_worker(self) -> None:
        """Invoke the worker callback and exit the app on return."""
        assert self._renderer is not None
        try:
            self._worker_cb(self._renderer)
        finally:
            self.call_from_thread(self.exit)

    def _branding_inner_width(self) -> int:
        """Return inner width of ``#branding`` in cells (``>= 6``)."""
        try:
            br = self.query_one("#branding", Static)
            cw = br.container_size.width
            if cw > 4:
                return max(6, cw)
        except NoMatches:
            pass
        return max(6, self._right_width - 2)

    def _sync_log_update(self, log_text: Text) -> None:
        """Push ``log_text`` to ``#log``."""
        try:
            self.query_one("#log", Static).update(log_text)
        except NoMatches:
            return

    def _scroll_log_now(self) -> None:
        """Scroll ``#log_container`` to bottom; clear ``_scroll_pending``."""
        self._scroll_pending = False
        try:
            sc = self.query_one("#log_container", VerticalScroll)
        except NoMatches:
            return
        sc.scroll_end(animate=False, x_axis=False)

    @staticmethod
    def _compose_brand_line(
        overlay: str | None,
        width: int,
        *,
        right_align: bool,
        fill_style: str = _SIDEBAR_BRANDING_FILL_STYLE,
    ) -> Text:
        """One sidebar row: space fill plus optional overlay (bold cyan)."""
        base = " " * width
        line = Text()
        if overlay is None:
            line.append(base, style=fill_style)
            return line
        ov = overlay if len(overlay) <= width else overlay[-width:]
        start = width - len(ov) if right_align else 0
        end = start + len(ov)
        if start > 0:
            line.append(base[:start], style=fill_style)
        cyan = "bold cyan"
        n = len(ov)
        i = 0
        while i < n:
            is_space = ov[i] == " "
            j = i + 1
            while j < n and (ov[j] == " ") == is_space:
                j += 1
            if is_space:
                line.append(base[start + i : start + j], style=fill_style)
            else:
                line.append(ov[i:j], style=cyan)
            i = j
        if end < width:
            line.append(base[end:width], style=fill_style)
        return line

    def _branding_renderable(self, *, inner_width: int | None = None) -> Align:
        """Sidebar block: signature + filler + cat (bottom-aligned).

        Args:
            inner_width: Row width in cells. Use during :meth:`compose`
                (``#branding`` not queryable yet). ``None`` → measured width.
        """
        w = (
            inner_width
            if inner_width is not None
            else self._branding_inner_width()
        )
        h = _SIDEBAR_BRANDING_INNER_HEIGHT
        cat = self._cat_frame
        cat_h = len(cat)
        cat_top = h - cat_h
        compose = self._compose_brand_line
        block = Text()
        nl = Text("\n")
        for y in range(h):
            if y == 0:
                overlay: str | None = KAPUSTAZH_SIGNATURE
            elif y >= cat_top:
                overlay = cat[y - cat_top]
            else:
                overlay = None
            block.append(compose(overlay, w, right_align=True))
            if y < h - 1:
                block.append(nl)
        return Align(block, "left", vertical="bottom")

    def _paint_branding(self) -> None:
        """Update ``#branding`` with latest sidebar block."""
        try:
            br = self.query_one("#branding", Static)
        except NoMatches:
            return
        br.update(self._branding_renderable())

    def _tick_scroll_coalesce(self) -> None:
        """Drain deferred midline scroll (interval only registered if coalescing on)."""
        if self._scroll_pending:
            self._scroll_log_now()

    def _tick_cat(self) -> None:
        """Advance cat sprite frame and repaint sidebar."""
        self._cat_frame = next(self._cat_iter)
        self._paint_branding()

    def arm_quit_wait(self) -> None:
        """Enable quit confirmation; defocus log so keys reach :meth:`on_key`."""
        self._quit_waiting = True
        self._quit_armed = False
        try:
            log = self.query_one("#log", Static)
            log.can_focus = False
        except NoMatches:
            pass
        self.set_focus(None)

    def _show_quit_bar(self) -> None:
        """Show the quit-confirmation bar."""
        try:
            self.query_one("#quit_hint", Static).display = True
        except NoMatches:
            return

    def _hide_quit_bar(self) -> None:
        """Hide the quit-confirmation bar."""
        try:
            self.query_one("#quit_hint", Static).display = False
        except NoMatches:
            return

    @staticmethod
    def _is_quit_key(event: Key) -> bool:
        """Return ``True`` if ``event`` is ``q``/``Q`` or ``Esc``."""
        if event.key == "escape":
            return True
        ch = event.character
        return ch is not None and ch.lower() == "q"

    def on_key(self, event: Key) -> None:
        """Two-step quit handler."""
        if not self._quit_waiting:
            return
        if self._is_quit_key(event):
            if not self._quit_armed:
                self._quit_armed = True
                self._show_quit_bar()
            else:
                self.quit_confirmed.set()
            event.stop()
            return
        if self._quit_armed:
            self._quit_armed = False
            self._hide_quit_bar()
