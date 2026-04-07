"""Recording indicator overlays — visual feedback while recording.

MacNativeIndicator: macOS-native floating panel via PyObjC (runs in a
    subprocess because AppKit's NSApplication event loop cannot coexist
    with Qt's event loop in the same process).

NullIndicator: no-op for CLI, headless, and unsupported platforms.
"""

from __future__ import annotations

import multiprocessing
import sys
from multiprocessing import Event, Process


class NullIndicator:
    """No-op indicator for CLI / headless / unsupported platforms."""

    def show(self) -> None:
        pass

    def hide(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# macOS native indicator (PyObjC + AppKit in a subprocess)
# ---------------------------------------------------------------------------

def _run_indicator_process(
    show_event: multiprocessing.synchronize.Event,
    hide_event: multiprocessing.synchronize.Event,
    quit_event: multiprocessing.synchronize.Event,
) -> None:
    """Entry point for the indicator subprocess. Runs an AppKit event loop."""
    try:
        import objc
        from AppKit import (
            NSApplication,
            NSBezierPath,
            NSColor,
            NSFloatingWindowLevel,
            NSMakeRect,
            NSPanel,
            NSScreen,
            NSTimer,
            NSView,
            NSWindowStyleMaskBorderless,
            NSWindowStyleMaskNonactivatingPanel,
        )
        from Foundation import NSObject
    except ImportError:
        return

    INDICATOR_SIZE = 20
    BOTTOM_MARGIN = 80

    class DotView(NSView):
        """Simple red dot that pulses."""

        _pulse_alpha = 1.0

        def drawRect_(self, rect):
            red = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.95, 0.2, 0.2, self._pulse_alpha
            )
            red.setFill()
            NSBezierPath.bezierPathWithOvalInRect_(rect).fill()

        def setPulseAlpha_(self, alpha):
            self._pulse_alpha = alpha
            self.setNeedsDisplay_(True)

    class Delegate(NSObject):
        """App delegate that manages the indicator panel and poll timer."""

        panel = objc.ivar()
        dot_view = objc.ivar()
        _pulse_up = objc.ivar()
        _current_alpha = objc.ivar()

        def applicationDidFinishLaunching_(self, notification):
            self._pulse_up = False
            self._current_alpha = 1.0

            screen = NSScreen.mainScreen().frame()
            x = (screen.size.width - INDICATOR_SIZE) / 2
            y = BOTTOM_MARGIN

            panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(x, y, INDICATOR_SIZE, INDICATOR_SIZE),
                NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
                2,  # NSBackingStoreBuffered
                False,
            )
            panel.setLevel_(NSFloatingWindowLevel)
            panel.setOpaque_(False)
            panel.setBackgroundColor_(NSColor.clearColor())
            panel.setHasShadow_(True)
            panel.setMovableByWindowBackground_(False)
            panel.setIgnoresMouseEvents_(True)
            panel.setCollectionBehavior_(1 << 0)  # canJoinAllSpaces

            dot_view = DotView.alloc().initWithFrame_(
                NSMakeRect(0, 0, INDICATOR_SIZE, INDICATOR_SIZE)
            )
            panel.setContentView_(dot_view)

            self.panel = panel
            self.dot_view = dot_view

            # Poll events from main process every 50ms
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.05, self, "pollEvents:", None, True
            )

        def pollEvents_(self, timer):
            if quit_event.is_set():
                NSApplication.sharedApplication().terminate_(None)
                return

            if show_event.is_set():
                show_event.clear()
                # Re-center on current main screen
                screen = NSScreen.mainScreen().frame()
                x = (screen.size.width - INDICATOR_SIZE) / 2
                y = BOTTOM_MARGIN
                self.panel.setFrameOrigin_((x, y))
                self.panel.orderFront_(None)
                # Start pulse animation
                NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    0.05, self, "pulseAnimation:", None, True
                )

            if hide_event.is_set():
                hide_event.clear()
                self.panel.orderOut_(None)

        def pulseAnimation_(self, timer):
            if not self.panel.isVisible():
                timer.invalidate()
                self._current_alpha = 1.0
                return

            step = 0.03
            if self._pulse_up:
                self._current_alpha = min(1.0, self._current_alpha + step)
                if self._current_alpha >= 1.0:
                    self._pulse_up = False
            else:
                self._current_alpha = max(0.5, self._current_alpha - step)
                if self._current_alpha <= 0.5:
                    self._pulse_up = True

            self.dot_view.setPulseAlpha_(self._current_alpha)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)  # NSApplicationActivationPolicyAccessory
    delegate = Delegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


class MacNativeIndicator:
    """macOS-native recording indicator using AppKit in a subprocess."""

    def __init__(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._show_event = ctx.Event()
        self._hide_event = ctx.Event()
        self._quit_event = ctx.Event()
        self._process = ctx.Process(
            target=_run_indicator_process,
            args=(self._show_event, self._hide_event, self._quit_event),
            daemon=True,
        )
        self._process.start()

    def show(self) -> None:
        if self._process.is_alive():
            self._show_event.set()

    def hide(self) -> None:
        if self._process.is_alive():
            self._hide_event.set()

    def shutdown(self) -> None:
        if self._process.is_alive():
            self._quit_event.set()
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()
