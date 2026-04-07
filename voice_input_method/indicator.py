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
            NSFont,
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

    INDICATOR_SIZE = 60
    BOTTOM_MARGIN = 80

    class MicView(NSView):
        """Custom view that draws a microphone icon on a dark circular background."""

        _pulse_alpha = 1.0

        def drawRect_(self, rect):
            # Circular dark background
            bg = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.1, 0.1, 0.85)
            bg.setFill()
            circle = NSBezierPath.bezierPathWithOvalInRect_(rect)
            circle.fill()

            # Microphone icon (drawn with basic shapes)
            cx = rect.size.width / 2
            cy = rect.size.height / 2

            white = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                1.0, 1.0, 1.0, self._pulse_alpha
            )
            white.setFill()
            white.setStroke()

            # Mic body (rounded rect)
            mic_w, mic_h = 12, 20
            mic_rect = NSMakeRect(cx - mic_w / 2, cy + 2, mic_w, mic_h)
            mic_body = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                mic_rect, mic_w / 2, mic_w / 2
            )
            mic_body.fill()

            # Mic arc (U-shape below the body)
            arc_path = NSBezierPath.alloc().init()
            arc_path.setLineWidth_(2.0)
            import math

            arc_cx = cx
            arc_cy = cy + 2
            arc_r = 11
            # Draw arc from left to right (bottom half)
            start_angle = 200  # degrees
            end_angle = 340
            arc_path.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
                (arc_cx, arc_cy), arc_r, start_angle, end_angle, True
            )
            arc_path.stroke()

            # Stand (vertical line + base)
            stand = NSBezierPath.alloc().init()
            stand.setLineWidth_(2.0)
            stand.moveToPoint_((cx, cy - 7))
            stand.lineToPoint_((cx, cy - 13))
            stand.stroke()

            base = NSBezierPath.alloc().init()
            base.setLineWidth_(2.0)
            base.moveToPoint_((cx - 8, cy - 13))
            base.lineToPoint_((cx + 8, cy - 13))
            base.stroke()

            # Red recording dot (top-right)
            red = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.95, 0.2, 0.2, self._pulse_alpha
            )
            red.setFill()
            dot_rect = NSMakeRect(rect.size.width - 18, rect.size.height - 18, 10, 10)
            dot = NSBezierPath.bezierPathWithOvalInRect_(dot_rect)
            dot.fill()

        def setPulseAlpha_(self, alpha):
            self._pulse_alpha = alpha
            self.setNeedsDisplay_(True)

    class Delegate(NSObject):
        """App delegate that manages the indicator panel and poll timer."""

        panel = objc.ivar()
        mic_view = objc.ivar()
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

            mic_view = MicView.alloc().initWithFrame_(
                NSMakeRect(0, 0, INDICATOR_SIZE, INDICATOR_SIZE)
            )
            panel.setContentView_(mic_view)

            self.panel = panel
            self.mic_view = mic_view

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

            self.mic_view.setPulseAlpha_(self._current_alpha)

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
