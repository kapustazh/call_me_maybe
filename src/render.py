"""Textual pipeline UI: streaming log, wave sidebar, quit confirmation."""

from __future__ import annotations

import random
import signal
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
_SIDEBAR_BRANDING_INNER_HEIGHT = 30

# Log stream → UI thread
_LOG_STREAM_CHARS_PER_UI_BATCH = 64
_LOG_SCROLL_COALESCE_MIDLINE_CHUNKS = True

# Timer cadences (seconds)
_BRANDING_TICK_SEC = 0.08
_LOG_WAVE_IDLE_TICK_SEC = 0.24
_CAT_ANIMATION_TICK_SEC = 0.3

# Log + wave merge: cap merge to last N lines (None = whole log, debug only).
_LOG_WAVE_MERGE_TAIL_LINE_COUNT: int | None = 384
_LOG_WAVE_MERGE_USE_WORKER_THREAD = True

# Phase grid shape
_LOG_GRID_HEIGHT_ROUND_TO_ROWS = 32
_LOG_GRID_MAX_HEIGHT_ROWS = 256

# Box-blur passes; per-grid override or None → use default.
_WAVE_PHASE_GRID_SMOOTHING_STEPS = 40
_BRANDING_GRID_SMOOTHING_STEPS: int | None = 12
_LOG_GRID_SMOOTHING_STEPS: int | None = None

# Wave glyph animation
_WAVE_PHASE_FRAME_COUNT = 300
_WAVE_GLYPH_CHARSET = (
    "    `~._^|',-!:}+{=\\/*;[]7oc><i?)(rlt1jsIz3vCuJ%5aYn"
    '"298e0f&L6OS$VGZxTyUhP4wkDFdgqbRpmX@QAEHK#BNWM'
)
_WAVE_RANDOM_SAMPLE_MAX = len(_WAVE_GLYPH_CHARSET) * 2 - 2

# Rich styles
_LOG_TEXT_WAVE_FILL_STYLE = "#1c221c"
_SIDEBAR_WAVE_RICH_STYLE = "rgb(48,48,48)"

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

WaveGridCacheKey = tuple[int, int, int]


# Wave helpers (pure)


def _ease_quad_in_out(t: float) -> float:
    """Apply quadratic in/out easing.

    Args:
        t: Normalized progress; clamped to ``[0, 1]``.

    Returns:
        Eased value in ``[0, 1]``.
    """
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 2.0 * t * t
    return -1.0 + (4.0 - 2.0 * t) * t


def _blur_wave_grid(
    grid: list[list[float]],
    height: int,
    width: int,
) -> list[list[float]]:
    """Box-blur a 2D float grid via a 5-point stencil.

    Edge cells reuse their own value so indices stay in bounds.

    Args:
        grid: Source grid shaped ``[height][width]``.
        height: Row count.
        width: Column count.

    Returns:
        Newly allocated blurred grid of the same shape.
    """
    out: list[list[float]] = [[0.0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            v = grid[y][x]
            xm = x - 1 if x > 0 else x
            xp = x + 1 if x < width - 1 else x
            ym = y - 1 if y > 0 else y
            yp = y + 1 if y < height - 1 else y
            out[y][x] = (
                v + grid[y][xm] + grid[y][xp] + grid[ym][x] + grid[yp][x]
            ) / 5.0
    return out


def _build_wave_phase_offsets(
    width: int,
    height: int,
    *,
    blur_steps: int | None = None,
) -> list[list[int]]:
    """Build the phase-offset grid that drives wave glyph animation.

    Random initial values are box-blurred, normalized, then quantized to
    indices into ``_WAVE_GLYPH_CHARSET``.

    Args:
        width: Grid columns.
        height: Grid rows.
        blur_steps: Override blur passes; ``None`` falls back to
            ``_WAVE_PHASE_GRID_SMOOTHING_STEPS``.

    Returns:
        Phase grid shaped ``[height][width]`` of frame indices.
    """
    grid: list[list[float]] = [
        [random.random() * _WAVE_RANDOM_SAMPLE_MAX for _ in range(width)]
        for _ in range(height)
    ]
    steps = (
        _WAVE_PHASE_GRID_SMOOTHING_STEPS
        if blur_steps is None
        else max(0, blur_steps)
    )
    for _ in range(steps):
        grid = _blur_wave_grid(grid, height, width)
    flat = [v for row in grid for v in row]
    vmin, vmax = min(flat), max(flat)
    span = vmax - vmin
    phases: list[list[int]] = []
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            if span <= 0.0:
                p = 0
            else:
                p = int(
                    (grid[y][x] - vmin) / span * (_WAVE_PHASE_FRAME_COUNT - 1)
                )
            row.append(
                max(0, min(_WAVE_PHASE_FRAME_COUNT - 1, p)),
            )
        phases.append(row)
    return phases


def _wave_cell_index(phase_off: int, tick: int) -> int:
    """Return the eased glyph index for one cell at ``tick``.

    Args:
        phase_off: Cell phase offset from the grid.
        tick: Animation tick.

    Returns:
        Index into ``_WAVE_GLYPH_CHARSET``.
    """
    nci = len(_WAVE_GLYPH_CHARSET) - 1
    denom = max(_WAVE_PHASE_FRAME_COUNT - 1, 1)
    t_frame = (phase_off + tick) % _WAVE_PHASE_FRAME_COUNT
    e = _ease_quad_in_out(t_frame / denom)
    return int(round(e * nci))


def _wave_line(phase_row: list[int], tick: int) -> str:
    """Render one row of wave glyphs as a string.

    Args:
        phase_row: Phase offsets for the row.
        tick: Animation tick.

    Returns:
        Glyph string of length ``len(phase_row)``.
    """
    nci = len(_WAVE_GLYPH_CHARSET) - 1
    return "".join(
        _WAVE_GLYPH_CHARSET[max(0, min(nci, _wave_cell_index(p, tick)))]
        for p in phase_row
    )


def _wave_frame(
    phase_offsets: list[list[int]],
    tick: int,
) -> tuple[str, ...]:
    """Render every row of ``phase_offsets`` at ``tick``.

    Args:
        phase_offsets: Phase grid built by ``_build_wave_phase_offsets``.
        tick: Animation tick.

    Returns:
        One glyph string per row.
    """
    return tuple(_wave_line(row, tick) for row in phase_offsets)


def _quantize_log_grid_height(line_count: int) -> int:
    """Round log-grid height up to ``_LOG_GRID_HEIGHT_ROUND_TO_ROWS``.

    Result is bounded by ``_SIDEBAR_BRANDING_INNER_HEIGHT`` and
    ``_LOG_GRID_MAX_HEIGHT_ROWS``.

    Args:
        line_count: Current log line count.

    Returns:
        Quantized height in rows.
    """
    need = max(
        _SIDEBAR_BRANDING_INNER_HEIGHT,
        min(line_count + 8, _LOG_GRID_MAX_HEIGHT_ROWS),
    )
    q = _LOG_GRID_HEIGHT_ROUND_TO_ROWS
    return min(
        _LOG_GRID_MAX_HEIGHT_ROWS,
        max(
            _SIDEBAR_BRANDING_INNER_HEIGHT,
            ((need + q - 1) // q) * q,
        ),
    )


# Blend wave into log whitespace


def _blend_wave_into_log_line(
    phase_row: list[int],
    tick: int,
    line: Text,
    width: int,
) -> Text:
    """Merge wave glyphs into the whitespace runs of one log line.

    The fast path (no whitespace in ``line``) keeps the original text and
    appends wave fill to ``width``. The slow path walks alternating
    whitespace and non-whitespace runs.

    Args:
        phase_row: Phase offsets for this row.
        tick: Animation tick.
        line: Source styled log line.
        width: Target output width in cells.

    Returns:
        New ``Text`` of ``width`` cells with wave filling whitespace.
    """
    plain = line.plain
    plen_raw = len(plain)
    if plen_raw > 0 and " " not in plain and "\t" not in plain:
        wave = _wave_line(phase_row, tick)
        wave += " " * max(0, width - len(wave))
        out = Text()
        take = min(plen_raw, width)
        if take:
            out.append(line[:take])
        if take < width:
            out.append(wave[take:width], style=_LOG_TEXT_WAVE_FILL_STYLE)
        return out
    wave = _wave_line(phase_row, tick)
    wave += " " * max(0, width - len(wave))
    plen = min(len(plain), width)
    out = Text()
    i = 0
    while i < plen:
        if plain[i] in " \t":
            j = i
            while j < plen and plain[j] in " \t":
                j += 1
            out.append(wave[i:j], style=_LOG_TEXT_WAVE_FILL_STYLE)
            i = j
        else:
            j = i
            while j < plen and plain[j] not in " \t":
                j += 1
            out.append(line[i:j])
            i = j
    if plen < width:
        out.append(wave[plen:width], style=_LOG_TEXT_WAVE_FILL_STYLE)
    return out


def _join_log_lines_with_wave(
    lines: list[Text],
    width: int,
    tick: int,
    grid: list[list[int]],
) -> Text:
    """Concatenate log lines with wave merged into the trailing window.

    Lines older than ``_LOG_WAVE_MERGE_TAIL_LINE_COUNT`` are emitted
    verbatim; the trailing window is passed through
    ``_blend_wave_into_log_line``. ``None`` disables the cap.

    Args:
        lines: Source log lines (Rich ``Text``).
        width: Display width in cells.
        tick: Animation tick.
        grid: Phase offset grid; rows reused via modulo.

    Returns:
        Joined ``Text`` ready for the ``#log`` widget.
    """
    if not lines:
        return Text("")
    gh = len(grid)
    cap = (
        max(8, _LOG_WAVE_MERGE_TAIL_LINE_COUNT)
        if _LOG_WAVE_MERGE_TAIL_LINE_COUNT is not None
        else None
    )
    n = len(lines)
    head_end = 0 if cap is None or n <= cap else n - cap
    nl = Text("\n")
    merged = Text()
    for i in range(head_end):
        if i:
            merged.append(nl)
        merged.append(lines[i])
    for i in range(head_end, n):
        if i:
            merged.append(nl)
        merged.append(
            _blend_wave_into_log_line(
                grid[i % gh],
                tick,
                lines[i],
                width,
            )
        )
    return merged


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
        _flush_seq: Monotonic count of ``_append_log`` calls.
        _log_dirty: Set when new content has buffered since the last
            refresh.
    """

    def __init__(self, app: PipelineApp) -> None:
        """Bind the renderer to a :class:`PipelineApp`.

        Args:
            app: Hosting Textual app.
        """
        self._app = app
        self._rendered = Text()
        self._flush_seq: int = 0
        self._log_dirty: bool = False

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

        Sets ``_log_dirty`` and either scrolls now (newline or tail
        chunk) or marks ``_scroll_pending`` for the next branding tick
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
        self._flush_seq += 1
        self._log_dirty = True
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
    """Textual app: streaming log pane with an animated wave sidebar.

    Layout: a docked right column (``#title_box`` + ``#branding``) plus
    a log pane backed by a ``VerticalScroll`` ``Static``. Two timers
    drive the animation:

    * ``_tick_brand`` paints the sidebar at ``_BRANDING_TICK_SEC``.
    * ``_tick_log_idle`` re-merges the log at
      ``_LOG_WAVE_IDLE_TICK_SEC`` so the wave drifts in whitespace
      even when output is idle.

    Heavy log merges run on ``_log_merge_executor``; concurrent
    submissions coalesce via the ``_log_merge_busy`` and
    ``_log_merge_pending`` flags.

    Attributes:
        quit_confirmed: Set when the user confirms quit.
    """

    CSS = """
    #right_column {
        dock: right;
        width: 20;
        height: 100%;
    }
    #title_box {
        height: auto;
        min-height: 1;
        padding: 0 1;
        background: $surface;
        color: $foreground;
        border: solid $boost;
    }
    #branding {
        width: 100%;
        height: 1fr;
        background: $surface;
        color: $foreground;
        border: solid $boost;
    }
    #log_pane {
        width: 1fr;
        height: 100%;
        border-top: solid $boost;
        border-left: solid $boost;
        border-bottom: solid $boost;
    }
    #log_container {
        width: 100%;
        height: 100%;
        background: transparent;
        overflow-y: auto;
    }
    #log {
        width: 100%;
        height: auto;
        padding: 0 1;
        background: transparent;
    }
    #quit_hint {
        dock: bottom;
        height: auto;
        background: $surface;
        color: $warning;
        display: none;
    }
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
        self._wave_tick: int = 0
        self._brand_phases: list[list[int]] | None = None
        self._brand_dims: WaveGridCacheKey | None = None
        self._log_phases: list[list[int]] | None = None
        self._log_dims: WaveGridCacheKey | None = None
        self._cat_iter = cycle(_CAT_SPRITE_FRAME_LINES)
        self._cat_frame = next(self._cat_iter)
        self.quit_confirmed = threading.Event()
        self._quit_waiting = False
        self._quit_armed = False
        self._log_merge_executor: ThreadPoolExecutor | None = None
        self._log_merge_seq: int = 0
        self._log_merge_busy: bool = False
        self._log_merge_pending: bool = False
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
                self._branding_renderable(),
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
        """Lock sidebar width, schedule animation timers, start worker."""
        rw = self._right_width
        rc = self.query_one("#right_column", Vertical)
        rc.styles.width = rw
        rc.styles.min_width = rw
        rc.styles.max_width = rw
        self._renderer = PipelineUIRenderer(self)
        self._ensure_brand_grid()
        self.set_interval(_BRANDING_TICK_SEC, self._tick_brand)
        self.set_interval(_LOG_WAVE_IDLE_TICK_SEC, self._tick_log_idle)
        self.set_interval(_CAT_ANIMATION_TICK_SEC, self._tick_cat)
        self._log_merge_executor = ThreadPoolExecutor(
            1,
            thread_name_prefix="logwave",
        )
        self.call_after_refresh(self._start_worker)

    def on_unmount(self) -> None:
        """Shut down the merge executor on app teardown."""
        ex = self._log_merge_executor
        if ex is not None:
            self._log_merge_executor = None
            ex.shutdown(wait=False, cancel_futures=True)

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

    def _branding_wave_width(self) -> int:
        """Return the inner width of ``#branding`` in cells.

        Falls back to ``self._right_width - 2`` when the widget has not
        been laid out yet.

        Returns:
            Width in cells, ``>= 6``.
        """
        try:
            br = self.query_one("#branding", Static)
            cw = br.container_size.width
            if cw > 4:
                return max(6, cw)
        except NoMatches:
            pass
        return max(6, self._right_width - 2)

    def _ensure_brand_grid(self) -> None:
        """Build (or reuse) the branding wave phase grid for current width."""
        w = self._branding_wave_width()
        h = _SIDEBAR_BRANDING_INNER_HEIGHT
        blur_used = (
            _BRANDING_GRID_SMOOTHING_STEPS
            if _BRANDING_GRID_SMOOTHING_STEPS is not None
            else _WAVE_PHASE_GRID_SMOOTHING_STEPS
        )
        key: WaveGridCacheKey = (w, h, blur_used)
        if self._brand_dims == key and self._brand_phases:
            return
        self._brand_phases = _build_wave_phase_offsets(
            w, h, blur_steps=_BRANDING_GRID_SMOOTHING_STEPS
        )
        self._brand_dims = key

    def _ensure_log_grid(
        self,
        width: int,
        line_count: int,
    ) -> None:
        """Build (or reuse) the log wave phase grid for the visible area.

        Args:
            width: Inner width of the ``#log`` widget in cells.
            line_count: Current log line count; drives quantized height.
        """
        w = max(6, width)
        h = _quantize_log_grid_height(line_count)
        blur_used = (
            _LOG_GRID_SMOOTHING_STEPS
            if _LOG_GRID_SMOOTHING_STEPS is not None
            else _WAVE_PHASE_GRID_SMOOTHING_STEPS
        )
        key: WaveGridCacheKey = (w, h, blur_used)
        if self._log_dims == key and self._log_phases:
            return
        self._log_phases = _build_wave_phase_offsets(
            w, h, blur_steps=_LOG_GRID_SMOOTHING_STEPS
        )
        self._log_dims = key

    def _measure_log_width(self) -> int | None:
        """Return the usable inner width of ``#log`` in cells.

        Returns:
            Width in cells, or ``None`` if ``#log`` is not laid out yet.
        """
        try:
            log_w = self.query_one("#log", Static)
        except NoMatches:
            return None
        iw = log_w.container_size.width
        return max(12, iw - 2) if iw > 4 else 80

    def _prepare_log_merge_inputs(
        self,
    ) -> tuple[int, list[Text], list[list[int]]] | None:
        """Snapshot inputs needed for one log merge pass.

        Always called on the main thread. The lists returned are owned
        by the caller and safe to hand to a worker thread.

        Returns:
            Tuple ``(width, lines, grid_snapshot)``, or ``None`` if
            ``#log`` is not yet available.
        """
        if self._renderer is None:
            return None
        width = self._measure_log_width()
        if width is None:
            return None
        lines = list(self._renderer._rendered.copy().split("\n"))
        self._ensure_log_grid(width, len(lines))
        grid = self._log_phases
        grid_snapshot = [list(row) for row in grid] if grid else []
        return (width, lines, grid_snapshot)

    def _sync_log_update(self, merged: Text) -> None:
        """Push ``merged`` to ``#log`` and drain a pending scroll if any.

        Args:
            merged: Result of :func:`_join_log_lines_with_wave`.
        """
        try:
            self.query_one("#log", Static).update(merged)
        except NoMatches:
            return
        if self._renderer is not None:
            self._renderer._log_dirty = False
        if self._scroll_pending:
            self._scroll_log_now()

    def _drain_log_merge_pending(self) -> None:
        """Schedule a refresh if one was requested while a merge was running."""
        if self._log_merge_pending:
            self._log_merge_pending = False
            self.call_after_refresh(self._refresh_log)

    def _apply_merged_log(self, merged: Text, seq: int) -> None:
        """Apply a worker-produced merge if it is still the freshest one.

        Stale results (``seq != self._log_merge_seq``) are dropped. The
        busy flag is always cleared so a queued refresh can proceed.

        Args:
            merged: Merge result produced by the worker thread.
            seq: Sequence id of the submitted merge.
        """
        try:
            if seq == self._log_merge_seq:
                self._sync_log_update(merged)
        finally:
            self._log_merge_busy = False
            self._drain_log_merge_pending()

    def _refresh_log(self) -> None:
        """Refresh ``#log`` with the current wave-merged content.

        Runs synchronously when ``_LOG_WAVE_MERGE_USE_WORKER_THREAD`` is
        ``False`` or the executor has been shut down; otherwise the
        merge is offloaded to the worker. Concurrent calls coalesce via
        ``_log_merge_busy`` and ``_log_merge_pending``.
        """
        if self._renderer is None:
            return
        ex = self._log_merge_executor
        threaded = _LOG_WAVE_MERGE_USE_WORKER_THREAD and ex is not None
        if threaded and self._log_merge_busy:
            self._log_merge_pending = True
            return
        prep = self._prepare_log_merge_inputs()
        if prep is None:
            return
        width, lines, grid_snapshot = prep
        if not grid_snapshot:
            self._sync_log_update(self._renderer._rendered.copy())
            return
        tick = self._wave_tick
        if not threaded:
            merged = _join_log_lines_with_wave(
                lines, width, tick, grid_snapshot
            )
            self._sync_log_update(merged)
            return
        assert ex is not None
        self._log_merge_busy = True
        self._log_merge_seq += 1
        seq = self._log_merge_seq

        def _run_merge() -> None:
            merged = _join_log_lines_with_wave(
                lines, width, tick, grid_snapshot
            )
            self.call_from_thread(self._apply_merged_log, merged, seq)

        ex.submit(_run_merge)

    @staticmethod
    def _compose_brand_line(
        wave_row: str,
        overlay: str | None,
        width: int,
        *,
        right_align: bool,
        wave_style: str = _LOG_TEXT_WAVE_FILL_STYLE,
    ) -> Text:
        """Compose one sidebar row: wave background plus optional overlay.

        Emits ``O(runs)`` Rich segments rather than one per cell. Overlay
        spaces show through to the wave; non-space glyphs render as
        bold cyan.

        Args:
            wave_row: Pre-rendered wave glyph row.
            overlay: Foreground glyphs (cat / signature) or ``None``.
            width: Total cells in the row.
            right_align: Right-align the overlay if ``True``, else left.
            wave_style: Rich style applied to wave glyphs.

        Returns:
            Styled ``Text`` of exactly ``width`` cells.
        """
        base = (wave_row + " " * width)[:width]
        line = Text()
        if overlay is None:
            line.append(base, style=wave_style)
            return line
        ov = overlay if len(overlay) <= width else overlay[-width:]
        start = width - len(ov) if right_align else 0
        end = start + len(ov)
        if start > 0:
            line.append(base[:start], style=wave_style)
        cyan = "bold cyan"
        n = len(ov)
        i = 0
        while i < n:
            is_space = ov[i] == " "
            j = i + 1
            while j < n and (ov[j] == " ") == is_space:
                j += 1
            if is_space:
                line.append(base[start + i : start + j], style=wave_style)
            else:
                line.append(ov[i:j], style=cyan)
            i = j
        if end < width:
            line.append(base[end:width], style=wave_style)
        return line

    def _branding_renderable(self) -> Align:
        """Build the full sidebar block (signature + filler + cat).

        Returns:
            ``Align`` wrapping the styled ``Text`` block, anchored to
            the bottom-left of the branding widget.
        """
        self._ensure_brand_grid()
        assert self._brand_phases is not None
        assert self._brand_dims is not None
        w = self._brand_dims[0]
        h = _SIDEBAR_BRANDING_INNER_HEIGHT
        wave = _wave_frame(self._brand_phases, self._wave_tick)
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
            block.append(
                compose(
                    wave[y],
                    overlay,
                    w,
                    right_align=True,
                    wave_style=_SIDEBAR_WAVE_RICH_STYLE,
                )
            )
            if y < h - 1:
                block.append(nl)
        return Align(block, "left", vertical="bottom")

    def _paint_branding(self) -> None:
        """Update the ``#branding`` widget with the latest sidebar block."""
        try:
            br = self.query_one("#branding", Static)
        except NoMatches:
            return
        br.update(self._branding_renderable())

    def _scroll_log_now(self) -> None:
        """Scroll ``#log_container`` to the bottom and clear ``_scroll_pending``."""
        self._scroll_pending = False
        try:
            sc = self.query_one("#log_container", VerticalScroll)
        except NoMatches:
            return
        sc.scroll_end(animate=False, x_axis=False)

    def _tick_brand(self) -> None:
        """Branding timer: advance wave, repaint, refresh log if dirty."""
        self._wave_tick += 1
        self._paint_branding()
        rend = self._renderer
        if rend is not None and rend._log_dirty:
            self._refresh_log()
        elif self._scroll_pending:
            self._scroll_log_now()

    def _tick_log_idle(self) -> None:
        """Idle log timer: re-merge so the wave drifts in whitespace."""
        if self._renderer is None:
            return
        self._refresh_log()

    def _tick_cat(self) -> None:
        """Cat timer: advance to the next sprite frame and repaint."""
        self._cat_frame = next(self._cat_iter)
        self._paint_branding()

    def arm_quit_wait(self) -> None:
        """Enable quit confirmation; blur the log so keys reach :meth:`on_key`."""
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
        self.query_one("#quit_hint", Static).display = True

    def _hide_quit_bar(self) -> None:
        """Hide the quit-confirmation bar."""
        self.query_one("#quit_hint", Static).display = False

    @staticmethod
    def _is_quit_key(event: Key) -> bool:
        """Return ``True`` if ``event`` is ``q``/``Q`` or ``Esc``.

        Args:
            event: Incoming key event.

        Returns:
            ``True`` if the key requests quit confirmation.
        """
        if event.key == "escape":
            return True
        ch = event.character
        return ch is not None and ch.lower() == "q"

    def on_key(self, event: Key) -> None:
        """Two-step quit handler.

        First quit-key arms the confirmation bar; the second confirms
        and sets :attr:`quit_confirmed`. Any other key cancels arming.

        Args:
            event: Incoming key event (consumed when handled).
        """
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
