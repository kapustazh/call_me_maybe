"""Textual pipeline UI: streaming log, sidebar art, quit confirmation."""

from __future__ import annotations

import random
import signal
import sys
import threading
import time
from collections.abc import Callable
from enum import IntEnum
from itertools import cycle
from types import FrameType

from rich.align import Align
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.events import Key
from textual.widgets import Static

# Layout
_SIDEBAR_WIDTH_CELLS_DEFAULT = 20
_SIDEBAR_ART_INNER_HEIGHT = 5

# Log stream → UI thread (batches)
_LOG_STREAM_CHARS_PER_UI_BATCH = 42
_LOG_SCROLL_COALESCE_MIDLINE_CHUNKS = True

# Midline scroll coalesce: poll interval when scroll deferred (seconds)
_SCROLL_COALESCE_TICK_SEC = 0.08
_CAT_ANIMATION_TICK_SEC = 0.3

# Token-by-token TUI delay
_TOKEN_VIS_MEAN_DELAY_SEC = 0.01
_TOKEN_VIS_MAX_DELAY_SEC = 0.35

# Rich terminal styles
_RICH_STYLE_SIDEBAR_FILL = "rgb(48,48,48)"
_RICH_STYLE_BOLD_CYAN = "bold cyan"
_RICH_STYLE_LOG_INFO = "cyan"
_RICH_STYLE_LOG_OK = "green"
_RICH_STYLE_LOG_ERR = "red"
_RICH_STYLE_LOG_PLAIN = "default"

# Assets
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


class LogColorPair(IntEnum):
    """Semantic id for log stream Rich style (see :func:`log_pair_style`)."""

    PLAIN = 0
    INFO = 1
    OK = 2
    ERR = 3


_LOG_COLOR_PAIR_TO_RICH_STYLE: dict[int, str] = {
    LogColorPair.PLAIN: _RICH_STYLE_LOG_PLAIN,
    LogColorPair.INFO: _RICH_STYLE_LOG_INFO,
    LogColorPair.OK: _RICH_STYLE_LOG_OK,
    LogColorPair.ERR: _RICH_STYLE_LOG_ERR,
}


def log_pair_style(pair: LogColorPair | int) -> str:
    """Rich style string for ``pair``; unknown id → empty string."""
    return _LOG_COLOR_PAIR_TO_RICH_STYLE.get(int(pair), "")


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

    def log_stream(self, text: str, *, pair: LogColorPair | int) -> None:
        """Stream ``text`` to the log in fixed-size batches.

        Each batch hops to the UI thread via ``call_from_thread``; batch
        size is ``_LOG_STREAM_CHARS_PER_UI_BATCH``.

        Args:
            text: Source text. Empty input is a no-op.
            pair: :class:`LogColorPair` or int; see :func:`log_pair_style`.
        """
        if not text:
            return
        style = log_pair_style(pair)
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

    def log_token_visual(
        self,
        piece: str,
        *,
        pair: LogColorPair | int,
    ) -> None:
        """Append one tokenizer fragment; sleep with mean delay on worker.

        Uses exponential inter-arrival so average spacing equals
        ``_TOKEN_VIS_MEAN_DELAY_SEC``. Runs from worker thread: UI update
        first, then sleep here (does not block Textual loop).

        Args:
            piece: Text to show (often one subword); empty becomes one space.
            pair: :class:`LogColorPair` or int; see :func:`log_pair_style`.
        """
        chunk = piece if piece else " "
        style = log_pair_style(pair)
        self._app.call_from_thread(
            self._append_log,
            chunk,
            style=style,
            tail=True,
        )
        mean = _TOKEN_VIS_MEAN_DELAY_SEC
        if mean > 0:
            delay = min(
                random.expovariate(1.0 / mean),
                _TOKEN_VIS_MAX_DELAY_SEC,
            )
            time.sleep(delay)

    def log_info_stream(self, text: str) -> None:
        """Stream ``text`` styled as info (cyan)."""
        self.log_stream(text, pair=LogColorPair.INFO)

    def log_ok_stream(self, text: str) -> None:
        """Stream ``text`` styled as ok (green)."""
        self.log_stream(text, pair=LogColorPair.OK)

    def log_err_stream(self, text: str) -> None:
        """Stream ``text`` styled as error (red)."""
        self.log_stream(text, pair=LogColorPair.ERR)

    def wait_until_quit(self) -> None:
        """Block until the user confirms quit (q/Esc twice or SIGINT)."""
        self._app.call_from_thread(self._app.arm_quit_wait)
        self._app.quit_confirmed.wait()


# PipelineApp (Textual TUI)
class PipelineApp(App[None]):
    """Textual app: streaming log pane plus sidebar art.

    Layout: docked right column (``#title_box`` + ``#sidebar_art``) plus log
    pane in ``VerticalScroll``.     Cat sprite animates on
    ``_CAT_ANIMATION_TICK_SEC``. When
    ``_LOG_SCROLL_COALESCE_MIDLINE_CHUNKS`` is True, a
    ``_SCROLL_COALESCE_TICK_SEC`` interval drains deferred midline scrolls.

    Attributes:
        quit_confirmed: Set when the user confirms quit.
    """

    CSS = f"""
    /* Right sidebar: title + art panel; pinned to terminal right */
    #right_column {{
        dock: right; /* pin the sidebar to the right of the terminal */
        width: {_SIDEBAR_WIDTH_CELLS_DEFAULT};
        height: 100%;
    }}
    /* Top strip of sidebar: app title */
    #title_box {{
        height: auto;
        min-height: 1;
        padding: 0 1;
        background: $surface;
        color: $foreground;
        border: solid $boost;
    }}
    /* Below title: grows to fill column (signature + cat area) */
    #sidebar_art {{
        width: 100%;
        height: 1fr;
        background: $surface;
        color: $foreground;
        border: solid $boost;
    }}
    /* Main area left of sidebar: streaming log */
    #log_pane {{
        width: 1fr;
        height: 100%;
        border-top: solid $boost;
        border-left: solid $boost;
        border-bottom: solid $boost;
    }}
    /* Scroll viewport for log content */
    #log_container {{
        width: 100%;
        height: 100%;
        background: transparent;
        overflow-y: auto; /* allow the log container to scroll */
    }}
    /* Rich Text widget that receives log updates */
    #log {{
        width: 100%;
        height: auto;
        padding: 0 1;
        background: transparent;
    }}
    /* Quit confirmation bar; hidden until code sets display */
    #quit_hint {{
        dock: bottom; /* position the quit hint at the bottom */
        height: auto;
        background: $surface;
        color: $warning;
        display: none; /* hide the quit hint by default */
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
        """Yield the widget tree (title, sidebar art, quit hint, log pane)."""
        with Vertical(id="right_column"):
            yield Static(
                Align.right(
                    Text(_TITLE_CALL_ME_MAYBE, style=_RICH_STYLE_BOLD_CYAN),
                ),
                id="title_box",
                markup=False,
            )
            yield Static(
                self._sidebar_art_renderable(
                    inner_width=max(6, self._right_width - 2),
                ),
                id="sidebar_art",
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
        """Mount: sidebar width, renderer, timers, pipeline worker."""
        self._apply_locked_sidebar_width()
        self._attach_pipeline_renderer()
        self._schedule_mount_timers()
        self._schedule_after_first_paint()

    def _apply_locked_sidebar_width(self) -> None:
        """Pin ``#right_column`` width to the configured sidebar width."""
        rw = self._right_width
        rc = self.query_one("#right_column", Vertical)
        rc.styles.width = rw
        rc.styles.min_width = rw
        rc.styles.max_width = rw

    def _attach_pipeline_renderer(self) -> None:
        """Create the :class:`PipelineUIRenderer` bound to this app."""
        self._renderer = PipelineUIRenderer(self)

    def _schedule_mount_timers(self) -> None:
        """Register scroll coalesce (optional) and cat animation intervals."""
        if _LOG_SCROLL_COALESCE_MIDLINE_CHUNKS:
            self.set_interval(
                _SCROLL_COALESCE_TICK_SEC,
                self._tick_scroll_coalesce,
            )
        self.set_interval(_CAT_ANIMATION_TICK_SEC, self._tick_cat)

    def _schedule_after_first_paint(self) -> None:
        """After layout: sidebar art repaint, then start pipeline worker."""
        self.call_after_refresh(self._paint_sidebar_art)
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

    def _sidebar_art_inner_width(self) -> int:
        """Return inner width of ``#sidebar_art`` in cells (``>= 6``)."""
        br = self.query_one("#sidebar_art", Static)
        cw = br.container_size.width
        if cw > 4:
            return max(6, cw)
        return max(6, self._right_width - 2)

    def _sync_log_update(self, log_text: Text) -> None:
        """Push ``log_text`` to ``#log``."""
        self.query_one("#log", Static).update(log_text)

    def _scroll_log_now(self) -> None:
        """Scroll ``#log_container`` to bottom; clear ``_scroll_pending``."""
        self._scroll_pending = False
        sc = self.query_one("#log_container", VerticalScroll)
        sc.scroll_end(animate=False, x_axis=False)

    @staticmethod
    def _compose_sidebar_art_line(
        overlay: str | None,
        width: int,
        *,
        right_align: bool,
        fill_style: str = _RICH_STYLE_SIDEBAR_FILL,
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
        n = len(ov)
        i = 0
        while i < n:
            is_space = ov[i] == " "
            j = i + 1
            while j < n and (ov[j] == " ") == is_space:
                j += 1
            if is_space:
                seg_a = start + i
                seg_b = start + j
                line.append(base[seg_a:seg_b], style=fill_style)
            else:
                line.append(ov[i:j], style=_RICH_STYLE_BOLD_CYAN)
            i = j
        if end < width:
            line.append(base[end:width], style=fill_style)
        return line

    def _sidebar_art_renderable(
        self, *, inner_width: int | None = None
    ) -> Align:
        """Sidebar block: signature + filler + cat (bottom-aligned).

        Args:
            inner_width: Row width in cells. Use during :meth:`compose`
                (``#sidebar_art``). ``None`` → measured width.
        """
        w = (
            inner_width
            if inner_width is not None
            else self._sidebar_art_inner_width()
        )
        h = _SIDEBAR_ART_INNER_HEIGHT
        cat = self._cat_frame
        cat_h = len(cat)
        cat_top = h - cat_h
        compose = self._compose_sidebar_art_line
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

    def _paint_sidebar_art(self) -> None:
        """Update ``#sidebar_art`` with latest sidebar block."""
        br = self.query_one("#sidebar_art", Static)
        br.update(self._sidebar_art_renderable())

    def _tick_scroll_coalesce(self) -> None:
        """Flush deferred midline scroll when coalesce timer fires."""
        if self._scroll_pending:
            self._scroll_log_now()

    def _tick_cat(self) -> None:
        """Advance cat sprite frame and repaint sidebar."""
        self._cat_frame = next(self._cat_iter)
        self._paint_sidebar_art()

    def arm_quit_wait(self) -> None:
        """Arm quit flow; unfocus log so keys reach ``on_key``."""
        self._quit_waiting = True
        self._quit_armed = False
        log = self.query_one("#log", Static)
        log.can_focus = False
        self.set_focus(None)

    def _show_quit_bar(self) -> None:
        """Show the quit-confirmation bar."""
        self.query_one("#quit_hint", Static).display = True

    def _hide_quit_bar(self) -> None:
        """Hide the quit-confirmation bar."""
        self.query_one("#quit_hint", Static).display = False

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
