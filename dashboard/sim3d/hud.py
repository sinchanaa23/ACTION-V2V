# ruff: noqa: E501 - long geometry and HUD expressions
"""Glass HUD for the 3D simulation: top status, side panels, bottom controls. Owner: P4.

Panels are dark translucent quads with a thin cyan rim. Text uses Ursina's
inline ``<rgb(r,g,b)>`` tags for accents. Every value shown is read from the
live SUMO session or the pipeline's output for the selected thing.
"""

from collections.abc import Callable

from ursina import Button, Entity, Quad, Text, camera, color, window

CYAN = (0.14, 0.9, 1.0)
BLUE = (0.33, 0.52, 1.0)
ORANGE = (1.0, 0.55, 0.18)
RED = (1.0, 0.28, 0.3)
GREEN = (0.35, 1.0, 0.6)
AMBER = (1.0, 0.78, 0.25)
MUTED = (0.55, 0.6, 0.68)
WHITE = (0.92, 0.95, 0.98)

BG = color.rgba(0.035, 0.05, 0.075, 0.8)
RIM = color.rgba(0.14, 0.9, 1.0, 0.22)
BTN = color.rgba(0.08, 0.11, 0.16, 0.92)
BTN_ON = color.rgba(0.0, 0.55, 0.66, 0.95)

TEXT_SCALE = 0.85


def tint(rgb: tuple[float, float, float], text: str) -> str:
    r, g, b = rgb
    return f"<rgb({r:.2f},{g:.2f},{b:.2f},1)>{text}<default>"


def muted(text: str) -> str:
    return tint(MUTED, text)


def wrap(text: str, width: int) -> list[str]:
    """Plain text broken into lines of at most ``width`` characters."""
    import textwrap

    return textwrap.wrap(text, width) or [""]


class Glass(Entity):
    """A translucent panel anchored at its top-left corner."""

    def __init__(self, x: float, y: float, w: float, h: float, **kwargs: object) -> None:
        super().__init__(parent=camera.ui, position=(x, y), **kwargs)  # type: ignore[arg-type]
        self.w, self.h = w, h
        self.rim = Entity(
            parent=self,
            model=Quad(radius=0.012, aspect=w / h),
            color=RIM,
            scale=(w + 0.003, h + 0.003),
            origin=(-0.5, 0.5),
            position=(-0.0015, 0.0015, 0.02),
        )
        self.bg = Entity(
            parent=self,
            model=Quad(radius=0.012, aspect=w / h),
            color=BG,
            scale=(w, h),
            origin=(-0.5, 0.5),
            z=0.01,
        )

    def resize(self, h: float) -> None:
        self.h = h
        self.bg.model = Quad(radius=0.012, aspect=self.w / h)
        self.bg.scale = (self.w, h)
        self.rim.model = Quad(radius=0.012, aspect=self.w / h)
        self.rim.scale = (self.w + 0.003, h + 0.003)


def label(
    parent: Entity, text: str, x: float, y: float, scale: float = TEXT_SCALE, **kw: object
) -> Text:
    return Text(
        text or " ", parent=parent, position=(x, y, -0.01), origin=(-0.5, 0.5), scale=scale, **kw
    )  # type: ignore[arg-type]


def button(
    parent: Entity,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    on_click: Callable[[], None],
    **kw: object,
) -> Button:
    b = Button(
        text=text or " ",
        parent=parent,
        position=(x + w / 2, y - h / 2, -0.01),
        scale=(w, h),
        color=BTN,
        highlight_color=color.rgba(0.1, 0.4, 0.5, 1),
        pressed_color=BTN_ON,
        radius=0.25,
        text_size=0.7,
        **kw,  # type: ignore[arg-type]
    )
    b.on_click = on_click
    return b


class Hud:
    """Builds and refreshes every HUD element; the app wires the callbacks."""

    def __init__(self, actions: dict[str, Callable[[], None]], speeds: tuple[float, ...]) -> None:
        a = window.aspect_ratio
        m = 0.014
        self.actions = actions
        self.left_open = True
        self.right_open = True

        # top bar
        self.top = Glass(-a / 2 + m, 0.5 - m, a - 2 * m, 0.052)
        label(self.top, tint(CYAN, "A.C.T.I.O.N") + " V2V", 0.014, -0.012, 1.25)
        label(self.top, muted("V2V hazard-warning simulation  ·  live SUMO"), 0.245, -0.02, 0.68)
        self.status = label(self.top, "", 0.6, -0.017, 0.78)

        # left panel
        self.left = Glass(-a / 2 + m, 0.5 - 0.08, 0.37, 0.70)
        self.left_title = label(
            self.left, tint(CYAN, "VEHICLES NEAR THE HAZARD"), 0.016, -0.014, 0.75
        )
        self.left_toggle = button(
            self.left, "«", 0.37 - 0.036, -0.008, 0.026, 0.026, actions["toggle_left"]
        )
        self.vehicle_buttons: list[Button] = []
        for i in range(9):
            b = button(self.left, "", 0.012, -0.045 - i * 0.031, 0.346, 0.027, lambda: None)
            b.text_entity.origin = (-0.5, 0)
            b.text_entity.x = -0.46
            self.vehicle_buttons.append(b)
        self.hazard_title = label(self.left, tint(CYAN, "HAZARDS"), 0.016, -0.34, 0.75)
        self.hazard_button = button(
            self.left, "", 0.012, -0.37, 0.346, 0.04, actions["select_hazard"]
        )
        self.hazard_button.text_entity.origin = (-0.5, 0)
        self.hazard_button.text_entity.x = -0.46
        self.log_title = label(self.left, tint(CYAN, "REPORTS"), 0.016, -0.43, 0.75)
        self.log = label(self.left, "", 0.016, -0.46, 0.68)
        self.left_items = [
            self.left_title,
            *self.vehicle_buttons,
            self.hazard_title,
            self.hazard_button,
            self.log_title,
            self.log,
        ]

        # right panel
        w = 0.47
        self.right = Glass(a / 2 - m - w, 0.5 - 0.08, w, 0.70)
        self.right_toggle = button(
            self.right, "»", 0.008, -0.008, 0.026, 0.026, actions["toggle_right"]
        )
        self.detail = label(self.right, "", 0.044, -0.014, 0.66)
        self.comm = label(self.right, "", 0.016, -0.33, 0.62)
        self.right_items = [self.detail, self.comm]

        # bottom bar
        bw = min(a - 2 * m, 1.62)
        self.bottom = Glass(-bw / 2, -0.5 + m + 0.098, bw, 0.098)
        x = 0.012
        y = -0.012
        self.reset_b = button(self.bottom, "Reset", x, y, 0.07, 0.034, actions["reset"])
        x += 0.076
        self.play_b = button(self.bottom, "Play", x, y, 0.08, 0.034, actions["play"])
        x += 0.086
        self.step_b = button(self.bottom, "Step", x, y, 0.065, 0.034, actions["step"])
        x += 0.08
        self.speed_bs: list[Button] = []
        for s in speeds:
            b = button(self.bottom, f"{s:g}x", x, y, 0.046, 0.034, lambda s=s: actions["speed"](s))  # type: ignore[misc]
            self.speed_bs.append(b)
            x += 0.05
        x += 0.014
        button(self.bottom, "«", x, y, 0.028, 0.034, actions["prev_scenario"])
        x += 0.032
        self.scenario_l = label(self.bottom, "", x + 0.004, y - 0.009, 0.72)
        x += 0.25
        button(self.bottom, "»", x, y, 0.028, 0.034, actions["next_scenario"])
        x += 0.04
        button(self.bottom, "«", x, y, 0.028, 0.034, actions["prev_config"])
        x += 0.032
        self.config_l = label(self.bottom, "", x + 0.004, y - 0.009, 0.72)
        x += 0.17
        button(self.bottom, "»", x, y, 0.028, 0.034, actions["next_config"])
        x += 0.04
        self.follow_b = button(self.bottom, "Follow", x, y, 0.075, 0.034, actions["follow"])
        x += 0.08
        button(self.bottom, "Panels", x, y, 0.075, 0.034, actions["panels"])
        # timeline
        self.track_w = bw - 0.024
        self.track = Entity(
            parent=self.bottom,
            model=Quad(radius=0.5),
            color=color.rgba(1, 1, 1, 0.08),
            scale=(self.track_w, 0.008),
            origin=(-0.5, 0),
            position=(0.012, -0.066, -0.01),
        )
        self.window_bar = Entity(
            parent=self.bottom,
            model="quad",
            color=color.rgba(*RED, 0.45),
            scale=(0.0, 0.008),
            origin=(-0.5, 0),
            position=(0.012, -0.066, -0.015),
        )
        self.played = Entity(
            parent=self.bottom,
            model=Quad(radius=0.5),
            color=color.rgba(*CYAN, 0.9),
            scale=(0.001, 0.008),
            origin=(-0.5, 0),
            position=(0.012, -0.066, -0.02),
        )
        self.ticks: list[Entity] = []
        self.time_l = label(self.bottom, "", 0.012, -0.075, 0.66)
        self.hint = label(
            self.bottom,
            muted(
                "right-drag rotate  |  scroll zoom  |  WASD pan  |  click a car  |  F follow  |  space play"
            ),
            0.36,
            -0.075,
            0.62,
        )

        self.toast = Text(" ", parent=camera.ui, origin=(0, 0), position=(-0.02, 0.385), scale=0.74)
        self.toast_until = 0.0

    # --- refresh -------------------------------------------------------------------

    def set_open(self, left: bool | None = None, right: bool | None = None) -> None:
        if left is not None:
            self.left_open = left
            for e in self.left_items:
                e.enabled = left
            self.left.resize(0.70 if left else 0.042)
            self.left_toggle.text = "«" if left else "»"
        if right is not None:
            self.right_open = right
            for e in self.right_items:
                e.enabled = right
            self.right.resize(0.70 if right else 0.042)
            self.right_toggle.text = "»" if right else "«"

    def timeline(
        self, t: float, end: float, window_s: tuple[float, float] | None, reports: list[float]
    ) -> None:
        f = max(0.0, min(t / end, 1.0)) if end else 0.0
        self.played.scale_x = max(self.track_w * f, 0.001)
        if window_s is None:
            self.window_bar.scale_x = 0
        else:
            self.window_bar.x = 0.012 + self.track_w * window_s[0] / end
            self.window_bar.scale_x = self.track_w * (window_s[1] - window_s[0]) / end
        while len(self.ticks) < len(reports):
            self.ticks.append(
                Entity(
                    parent=self.bottom,
                    model="quad",
                    color=color.rgba(*ORANGE, 1),
                    scale=(0.003, 0.02),
                    position=(0, -0.066, -0.03),
                )
            )
        for i, tick in enumerate(self.ticks):
            tick.enabled = i < len(reports)
            if i < len(reports):
                tick.x = 0.012 + self.track_w * reports[i] / end

    def highlight(self, buttons: list[Button], active: int | None) -> None:
        for i, b in enumerate(buttons):
            b.color = BTN_ON if i == active else BTN
