from __future__ import annotations

import curses
import threading
import time
from dataclasses import dataclass
from itertools import cycle
from types import TracebackType

# --- tuning ---------------------------------------------------------------
_DOUPDATE_EVERY_N_CHARS = 12
_DELAY = 0.008
_DELAY_CAT = 0.3
_DEFAULT_RIGHT_WIDTH = 28

_QUIT_HINT_LINE = (
    " Press q or Esc again to quit. Any other key closes this bar. "
)

# --- cat art --------------------------------------------------------------
_CAT_FRAMES: tuple[tuple[str, ...], ...] = (
    (r" /\_/\ ", r"( o.o )", r" > ^ < "),
    (r" /\_/\ ", r"( -.- )", r" > ^ < "),
    (r" /\_/\ ", r"( o.o )", r" > ~ < "),
    (r" /\_/\ ", r"( ^.^ )", r" > ^ < "),
)


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Theme:
    info_pair: int = 1
    ok_pair: int = 2
    err_pair: int = 3


class CatAnimation:
    """Rotates ASCII cat frames each time you paint."""

    def __init__(
        self,
        frames: tuple[tuple[str, ...], ...] = _CAT_FRAMES,
    ) -> None:
        self._frames = cycle(frames)

    def __next__(self) -> tuple[str, ...]:
        return next(self._frames)


class Kapustazh:
    """Label above the cat."""

    SIGNATURE = "kapustazh"


class RightBrandingLayer:
    """Kapustazh line + cat art, bottom-right. Caller erases win first."""

    def __init__(self, theme: Theme, cat: CatAnimation | None = None) -> None:
        self._theme = theme
        self._cat = cat or CatAnimation()

    def paint(self, win: curses.window) -> None:
        h, w = win.getmaxyx()
        frame = next(self._cat)
        lines = list(frame)
        author = Kapustazh.SIGNATURE
        block_h = len(lines) + 2
        base_y = max(0, h - block_h)
        max_line = max(len(author), max(len(s) for s in lines))
        x0 = max(0, w - max_line - 2)
        try:
            if curses.has_colors():
                pair = curses.color_pair(self._theme.info_pair)
                win.addstr(base_y, x0, author, pair)
            else:
                win.addstr(base_y, x0, author)
            for i, line in enumerate(lines):
                yy = base_y + 1 + i
                if yy < h:
                    win.addstr(yy, x0, line[: max(0, w - x0 - 1)])
        except curses.error:
            pass


class QuitHintLayer:
    """Full-width bottom bar; second-step quit confirm text."""

    def __init__(self, line: str = _QUIT_HINT_LINE.strip()) -> None:
        self._line = line
        self.win: curses.window | None = None
        self.visible: bool = False

    def show(self) -> None:
        h, w = curses.LINES, curses.COLS
        bar_h = 2
        row = max(0, h - bar_h)
        if self.win is None:
            self.win = curses.newwin(bar_h, w, row, 0)
        else:
            self.win.mvwin(row, 0)
            self.win.resize(bar_h, w)
        self.win.erase()
        if w > 1:
            try:
                self.win.addnstr(0, 0, self._line, w - 1)
            except curses.error:
                pass
        self.visible = True

    def hide(self) -> None:
        self.visible = False
        if self.win is not None:
            self.win.erase()

    def reset(self) -> None:
        self.visible = False
        self.win = None


def composite_doupdate(
    left: curses.window | None,
    right: curses.window | None,
    hint: QuitHintLayer,
) -> None:
    """Stack: log pane, right pane, optional hint on top. Caller holds lock."""
    if left is not None:
        left.noutrefresh()
    if right is not None:
        right.noutrefresh()
    if hint.visible and hint.win is not None:
        hint.win.noutrefresh()
    curses.doupdate()


class SplitRenderer:
    """Orchestrates log pane + right branding + quit hint."""

    def __init__(self, *, right_width: int = _DEFAULT_RIGHT_WIDTH) -> None:
        self._right_width = max(18, int(right_width))
        self._stdscr: curses.window | None = None
        self._left: curses.window | None = None
        self._right: curses.window | None = None
        self._theme = Theme()
        self._branding = RightBrandingLayer(self._theme)
        self._quit_hint = QuitHintLayer()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._anim_thread: threading.Thread | None = None
        self._closed: bool = False

    def __enter__(self) -> "SplitRenderer":
        try:
            self._stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            curses.curs_set(0)
            self._stdscr.nodelay(True)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(self._theme.info_pair, curses.COLOR_CYAN, -1)
                curses.init_pair(self._theme.ok_pair, curses.COLOR_GREEN, -1)
                curses.init_pair(self._theme.err_pair, curses.COLOR_RED, -1)
            self._rebuild_windows()
            self._start_animation()
            self._closed = False
            return self
        except Exception as exc:  # pragma: no cover
            self.close()
            raise RenderError(str(exc)) from exc

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._anim_thread is not None:
            self._anim_thread.join(timeout=0.2)
        had_session = self._stdscr is not None
        self._quit_hint.reset()
        self._stdscr = None
        self._left = None
        self._right = None
        if had_session:
            try:
                curses.nocbreak()
                curses.echo()
                curses.endwin()
            except curses.error:
                pass

    def _rebuild_windows(self) -> None:
        assert self._stdscr is not None
        h, w = self._stdscr.getmaxyx()
        right_w = min(self._right_width, max(18, w // 2))
        left_w = max(10, w - right_w)
        self._left = curses.newwin(h, left_w, 0, 0)
        self._right = curses.newwin(h, right_w, 0, left_w)
        self._left.scrollok(True)
        self._left.idlok(False)
        self._left.erase()
        self._right.erase()
        self._left.noutrefresh()
        self._right.noutrefresh()
        curses.doupdate()

    def _start_animation(self) -> None:
        def _run() -> None:
            while not self._stop.is_set():
                with self._lock:
                    if self._right is not None:
                        self._right.erase()
                        self._branding.paint(self._right)
                    composite_doupdate(
                        self._left,
                        self._right,
                        self._quit_hint,
                    )
                time.sleep(_DELAY_CAT)

        self._anim_thread = threading.Thread(target=_run, daemon=True)
        self._anim_thread.start()

    @staticmethod
    def _is_quit_key(ch: int) -> bool:
        return ch in (ord("q"), ord("Q"), 27)

    def _show_quit_hint_bar(self) -> None:
        with self._lock:
            if self._stdscr is None:
                return
            self._quit_hint.show()
            composite_doupdate(
                self._left,
                self._right,
                self._quit_hint,
            )

    def _hide_quit_hint_bar(self) -> None:
        with self._lock:
            self._quit_hint.hide()
            if self._left is not None:
                try:
                    self._left.touchwin()
                except curses.error:
                    pass
                self._left.noutrefresh()
            if self._right is not None:
                self._right.noutrefresh()
            curses.doupdate()

    def wait_until_quit(self) -> None:
        """Block until user confirms quit (q/Esc twice) or cancels hint."""
        if self._left is None or self._closed:
            return
        self._left.keypad(True)
        self._left.nodelay(False)
        confirm = False
        try:
            while True:
                ch = self._left.getch()
                with self._lock:
                    composite_doupdate(
                        self._left,
                        self._right,
                        self._quit_hint,
                    )
                if not confirm:
                    if self._is_quit_key(ch):
                        confirm = True
                        self._show_quit_hint_bar()
                    continue
                if self._is_quit_key(ch):
                    break
                confirm = False
                self._hide_quit_hint_bar()
        finally:
            self._left.nodelay(True)

    def log_stream(
        self,
        text: str,
        *,
        pair: int,
        char_delay: float = _DELAY,
    ) -> None:
        """Stream text to the left pane with a short per-character delay."""
        pending_physical = 0
        for ch in text:
            with self._lock:
                if self._left is None:
                    return
                attr = curses.color_pair(pair) if curses.has_colors() else 0
                try:
                    self._left.addstr(ch, attr)
                except curses.error:
                    try:
                        self._left.scroll(1)
                        self._left.move(self._left.getmaxyx()[0] - 2, 0)
                        self._left.addstr(ch, attr)
                    except curses.error:
                        pass
                self._left.noutrefresh()
                pending_physical += 1
                if pending_physical >= _DOUPDATE_EVERY_N_CHARS:
                    curses.doupdate()
                    pending_physical = 0
            time.sleep(char_delay)
        with self._lock:
            if self._left is not None:
                curses.doupdate()

    def log_info(self, text: str) -> None:
        self._log_line(text, pair=self._theme.info_pair)

    def log_ok(self, text: str) -> None:
        self._log_line(text, pair=self._theme.ok_pair)

    def log_err(self, text: str) -> None:
        self._log_line(text, pair=self._theme.err_pair)

    def log_info_stream(self, text: str) -> None:
        self.log_stream(text, pair=self._theme.info_pair)

    def log_ok_stream(self, text: str) -> None:
        self.log_stream(text, pair=self._theme.ok_pair)

    def log_err_stream(self, text: str) -> None:
        self.log_stream(text, pair=self._theme.err_pair)

    def _log_line(self, text: str, *, pair: int) -> None:
        with self._lock:
            if self._left is None:
                return
            try:
                if curses.has_colors():
                    self._left.addstr(text + "\n", curses.color_pair(pair))
                else:
                    self._left.addstr(text + "\n")
            except curses.error:
                try:
                    self._left.scroll(1)
                    self._left.move(self._left.getmaxyx()[0] - 2, 0)
                    self._left.clrtoeol()
                    max_w = max(0, self._left.getmaxyx()[1] - 1)
                    self._left.addstr(text[:max_w])
                    self._left.addstr("\n")
                except curses.error:
                    pass
            self._left.noutrefresh()
            curses.doupdate()
