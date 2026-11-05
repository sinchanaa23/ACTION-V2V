# ruff: noqa: E501 - long layout expressions
"""Minimal glass interface for the 3D simulation. Owner: P4.

Only the brand, a small live-status chip, the tour's chapter rail and its
caption are on screen at first. The simulation controls fold into a dock at
the bottom left, and details appear in a card only when something is
selected. Every value shown is read from the live SUMO session or the
pipeline's output for the selected thing.
"""

import math
import time
from collections.abc import Callable, Sequence

from ursina import Button, Entity, Mesh, Quad, Text, Vec3, camera, color, curve, window

CYAN = (0.14, 0.9, 1.0)
BLUE = (0.36, 0.56, 1.0)
ORANGE = (1.0, 0.56, 0.2)
RED = (1.0, 0.3, 0.32)
GREEN = (0.4, 1.0, 0.66)
AMBER = (1.0, 0.78, 0.3)
MUTED = (0.56, 0.62, 0.7)
FAINT = (0.36, 0.41, 0.48)
WHITE = (0.93, 0.95, 0.98)

GLASS = color.rgba(0.03, 0.045, 0.07, 0.72)
RIM = color.rgba(0.6, 0.85, 1.0, 0.13)
BTN = color.rgba(1, 1, 1, 0.045)
BTN_HOVER = color.rgba(0.14, 0.9, 1.0, 0.16)
BTN_ON = color.rgba(0.14, 0.9, 1.0, 0.3)

ActionFn = Callable[[], None]


def tint(rgb: tuple[float, float, float], text: str) -> str:
    r, g, b = rgb
    return f"<rgb({r:.2f},{g:.2f},{b:.2f},1)>{text}<default>"


def muted(text: str) -> str:
    return tint(MUTED, text)


def faint(text: str) -> str:
    return tint(FAINT, text)


def spaced(text: str) -> str:
    """Letter-spaced capitals for small labels."""
    return " ".join(text.upper())


def wrap(text: str, width: int) -> list[str]:
    """Plain text broken into lines of at most ``width`` characters."""
    import textwrap

    return textwrap.wrap(text, width) or [""]


class Glass(Entity):
    """A frosted panel anchored at its top-left corner."""

    def __init__(
        self, x: float, y: float, w: float, h: float, parent: Entity | None = None, **kwargs: object
    ) -> None:
        super().__init__(parent=parent or camera.ui, position=(x, y), **kwargs)  # type: ignore[arg-type]
        self.w = w
        self.rim = Entity(parent=self, color=RIM, origin=(-0.5, 0.5), z=0.02)
        self.bg = Entity(parent=self, color=GLASS, origin=(-0.5, 0.5), z=0.01)
        self.resize(w, h)

    def resize(self, w: float, h: float) -> None:
        self.w, self.h = w, h
        r = min(0.018, h / 2) / max(h, 1e-3)
        for e, (ew, eh) in ((self.bg, (w, h)), (self.rim, (w + 0.003, h + 0.003))):
            e.model = Quad(radius=r, aspect=ew / max(eh, 1e-3))
            e.origin = (-0.5, 0.5)  # a new model drops the old origin
            e.scale = (ew, eh)
        self.rim.position = (-0.0015, 0.0015, 0.02)


def label(parent: Entity, text: str, x: float, y: float, scale: float = 0.8, **kw: object) -> Text:
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
    on_click: ActionFn,
    **kw: object,
) -> Button:
    b = Button(
        text=text or " ",
        parent=parent,
        position=(x + w / 2, y - h / 2, -0.01),
        scale=(w, h),
        color=BTN,
        highlight_color=BTN_HOVER,
        pressed_color=BTN_ON,
        radius=0.5,
        text_size=0.62,
        **kw,  # type: ignore[arg-type]
    )
    b.on_click = on_click
    return b


def icon(kind: str, parent: Entity, size: float) -> Entity:
    """Play / pause glyphs as meshes (the bundled font has no such characters)."""
    if kind == "play":
        verts = [Vec3(-0.35, -0.42, 0), Vec3(-0.35, 0.42, 0), Vec3(0.42, 0, 0)]
        tris = [0, 1, 2]
    else:
        verts, tris = [], []
        for x0 in (-0.36, 0.1):
            i = len(verts)
            verts += [
                Vec3(x0, -0.4, 0),
                Vec3(x0, 0.4, 0),
                Vec3(x0 + 0.26, 0.4, 0),
                Vec3(x0 + 0.26, -0.4, 0),
            ]
            tris += [i, i + 1, i + 2, i, i + 2, i + 3]
    return Entity(
        parent=parent,
        model=Mesh(vertices=verts, triangles=tris),
        double_sided=True,
        color=color.rgba(*WHITE, 1),
        scale=size,
        z=-0.02,
    )


def scrim(aspect: float) -> Mesh:
    """A dark wash in the lower-left corner, fading out towards the centre."""
    x0, x1, y0, y1 = -aspect / 2, aspect * 0.12, -0.5, 0.05
    verts = [
        Vec3(x0, y0, 0),
        Vec3(x0, y1, 0),
        Vec3(x1, y1, 0),
        Vec3(x1, y0, 0),
        Vec3(x0, -0.17, 0),
        Vec3(x1 * 0.2, -0.17, 0),
    ]
    dark = color.rgba(0.0, 0.01, 0.025, 0.62)
    clear = color.rgba(0.0, 0.01, 0.025, 0.0)
    cols = [dark, clear, clear, clear, dark, dark]
    tris = [0, 4, 5, 0, 5, 3, 4, 1, 2, 4, 2, 5, 5, 2, 3]
    return Mesh(vertices=verts, triangles=tris, colors=cols)


class Card:
    """The contextual detail card: appears on the right when something is selected."""

    W = 0.5

    def __init__(self, on_close: ActionFn) -> None:
        a = window.aspect_ratio
        self.x = a / 2 - 0.035 - self.W
        self.y = 0.37
        self.glass = Glass(self.x, self.y, self.W, 0.2)
        g = self.glass
        self.kicker = label(g, " ", 0.024, -0.022, 0.6)
        self.title = label(g, " ", 0.024, -0.044, 1.05)
        self.close = button(g, "×", self.W - 0.05, -0.016, 0.034, 0.034, on_close)
        self.tabs: list[Button] = [button(g, " ", 0, 0, 0.1, 0.03, lambda: None) for _ in range(3)]
        self.body = label(g, " ", 0.024, -0.13, 0.64)
        self.actions: list[Button] = [
            button(g, " ", 0, 0, 0.13, 0.034, lambda: None) for _ in range(3)
        ]
        self.items: list[Button] = []
        for _ in range(6):
            b = button(g, " ", 0.018, 0, self.W - 0.036, 0.03, lambda: None)
            b.text_entity.origin = (-0.5, 0)
            b.text_entity.x = -0.47
            self.items.append(b)
        self.visible = True
        self.hide()
        self.key: tuple[str, ...] | None = None

    def hide(self) -> None:
        if self.visible:
            self.glass.enabled = False
            self.visible = False
            self.key = None

    def show(
        self,
        key: tuple[str, ...],
        kicker: str,
        title: str,
        body: str,
        tabs: Sequence[tuple[str, ActionFn, bool]] = (),
        actions: Sequence[tuple[str, ActionFn]] = (),
        items: Sequence[tuple[str, ActionFn, bool]] = (),
    ) -> None:
        """Fill and lay out the card; slides in when it shows something new."""
        g = self.glass
        if not self.visible or key != self.key:
            g.enabled = True
            g.x = self.x + 0.05
            g.animate_x(self.x, duration=0.35, curve=curve.out_expo)
        self.visible, self.key = True, key
        self.kicker.text = kicker
        self.title.text = title
        y = -0.098
        x = 0.018
        for b, spec in zip(self.tabs, [*tabs, *[None] * (3 - len(tabs))], strict=True):
            b.enabled = spec is not None
            if spec is None:
                continue
            text, fn, on = spec
            w = 0.03 + 0.0105 * len(text)
            b.text = text
            b.scale = (w, 0.03)
            b.position = (x + w / 2, y - 0.015, -0.01)
            b.on_click = fn
            b.color = BTN_ON if on else BTN
            x += w + 0.008
        if tabs:
            y -= 0.045
        self.body.text = body or " "
        self.body.y = y
        y -= self.body.height + 0.016
        for b, spec in zip(
            self.items, [*items, *[None] * (len(self.items) - len(items))], strict=True
        ):
            b.enabled = spec is not None
            if spec is None:
                continue
            text, fn, on = spec
            b.text = text
            b.text_entity.origin = (-0.5, 0)
            b.text_entity.x = -0.47
            b.y = y - 0.015
            b.on_click = fn
            b.color = BTN_ON if on else BTN
            y -= 0.034
        if items:
            y -= 0.008
        x = 0.018
        for b, spec in zip(self.actions, [*actions, *[None] * (3 - len(actions))], strict=True):
            b.enabled = spec is not None
            if spec is None:
                continue
            text, fn = spec
            w = 0.04 + 0.0105 * len(text)
            b.text = text
            b.scale = (w, 0.034)
            b.position = (x + w / 2, y - 0.017, -0.01)
            b.on_click = fn
            x += w + 0.01
        if actions:
            y -= 0.05
        g.resize(self.W, -y + 0.012)


class Dock:
    """Simulation controls folded into one pill at the bottom left."""

    def __init__(self, actions: dict[str, Callable[..., None]], speeds: tuple[float, ...]) -> None:
        a = window.aspect_ratio
        m = 0.035
        self.open = False
        self.pill = Glass(-a / 2 + m, -0.5 + m + 0.052, 0.6, 0.052)
        p = self.pill
        self.play_b = button(p, " ", 0.01, -0.008, 0.036, 0.036, actions["play"])
        self.play_icon = icon("play", self.play_b, 0.42)
        self.pause_icon = icon("pause", self.play_b, 0.42)
        self.time_l = label(p, " ", 0.058, -0.016, 0.68)
        self.more_b = button(p, "Controls", 0.29, -0.01, 0.09, 0.032, actions["dock"])
        self.mic_b = button(p, "Voice", 0.39, -0.01, 0.09, 0.032, actions["mic"])
        self.type_b = button(p, "Type", 0.49, -0.01, 0.09, 0.032, actions["type"])

        self.panel = Glass(-a / 2 + m, -0.5 + m + 0.31, 0.6, 0.25)
        q = self.panel
        x, y = 0.022, -0.02
        label(q, faint(spaced("speed")), x, y - 0.006, 0.5)
        self.speed_bs: list[Button] = []
        bx = 0.11
        for s in speeds:
            self.speed_bs.append(
                button(q, f"{s:g}×", bx, y, 0.058, 0.028, lambda s=s: actions["speed"](s))
            )  # type: ignore[misc]
            bx += 0.064
        y -= 0.044
        label(q, faint(spaced("sim")), x, y - 0.006, 0.5)
        bx = 0.11
        for text, key, w in (
            ("Step", "step", 0.07),
            ("Reset", "reset", 0.075),
            ("Skip to event", "skip", 0.13),
            ("Ground truth", "truth", 0.13),
        ):
            b = button(q, text, bx, y, w, 0.028, actions[key])
            if key == "truth":
                self.truth_b = b
            bx += w + 0.008
        y -= 0.044
        label(q, faint(spaced("scenario")), x, y - 0.006, 0.5)
        button(q, "«", 0.11, y, 0.03, 0.028, actions["prev_scenario"])
        self.scenario_l = label(q, " ", 0.15, y - 0.004, 0.66)
        button(q, "»", 0.55, y, 0.03, 0.028, actions["next_scenario"])
        y -= 0.044
        label(q, faint(spaced("config")), x, y - 0.006, 0.5)
        button(q, "«", 0.11, y, 0.03, 0.028, actions["prev_config"])
        self.config_l = label(q, " ", 0.15, y - 0.004, 0.66)
        button(q, "»", 0.55, y, 0.03, 0.028, actions["next_config"])
        y -= 0.05
        self.track_x, self.track_w, self.track_y = x, 0.6 - 2 * x, y - 0.006
        self.track = Entity(
            parent=q,
            model=Quad(radius=0.5),
            color=color.rgba(1, 1, 1, 0.07),
            scale=(self.track_w, 0.006),
            origin=(-0.5, 0),
            position=(x, self.track_y, -0.01),
        )
        self.window_bar = Entity(
            parent=q,
            model="quad",
            color=color.rgba(*RED, 0.4),
            scale=(0.0, 0.006),
            origin=(-0.5, 0),
            position=(x, self.track_y, -0.015),
        )
        self.played = Entity(
            parent=q,
            model=Quad(radius=0.5),
            color=color.rgba(*CYAN, 0.9),
            scale=(0.001, 0.006),
            origin=(-0.5, 0),
            position=(x, self.track_y, -0.02),
        )
        self.ticks: list[Entity] = []
        self.timeline_l = label(q, " ", x, y - 0.016, 0.56)
        self.panel.enabled = False

    def set_open(self, open_: bool) -> None:
        self.open = open_
        self.panel.enabled = open_
        self.more_b.text = "Close" if open_ else "Controls"
        self.more_b.color = BTN_ON if open_ else BTN

    def timeline(
        self, t: float, end: float, window_s: tuple[float, float] | None, reports: list[float]
    ) -> None:
        x, w = self.track_x, self.track_w
        f = max(0.0, min(t / end, 1.0)) if end else 0.0
        self.played.scale_x = max(w * f, 0.001)
        if window_s is None:
            self.window_bar.scale_x = 0
        else:
            self.window_bar.x = x + w * window_s[0] / end
            self.window_bar.scale_x = w * (window_s[1] - window_s[0]) / end
        while len(self.ticks) < len(reports):
            self.ticks.append(
                Entity(
                    parent=self.panel,
                    model="quad",
                    color=color.rgba(*ORANGE, 1),
                    scale=(0.0025, 0.016),
                    position=(0, self.track_y, -0.03),
                )
            )
        for i, tick in enumerate(self.ticks):
            tick.enabled = i < len(reports)
            if i < len(reports):
                tick.x = x + w * reports[i] / end

    def highlight(self, active: int | None) -> None:
        for i, b in enumerate(self.speed_bs):
            b.color = BTN_ON if i == active else BTN


class Hud:
    """Brand, status chip, chapter rail, caption, dock, card and toast."""

    def __init__(
        self, actions: dict[str, Callable[..., None]], speeds: tuple[float, ...], chapters: int
    ) -> None:
        a = window.aspect_ratio
        m = 0.035
        self.actions = actions
        self.shown = True
        ui = Entity(parent=camera.ui)
        self.root = ui

        self.brand = Text(
            tint(WHITE, "A.C.T.I.O.N") + "   " + tint(CYAN, "V2V"),
            parent=ui,
            position=(-a / 2 + m, 0.5 - m),
            origin=(-0.5, 0.5),
            scale=1.15,
        )
        self.brand_sub = Text(
            faint(spaced("hazard warnings") + "   ·   " + spaced("hebbal")),
            parent=ui,
            position=(-a / 2 + m, 0.5 - m - 0.04),
            origin=(-0.5, 0.5),
            scale=0.48,
        )

        self.status = Text(
            " ",
            parent=ui,
            position=(a / 2 - m - 0.022, 0.5 - m - 0.004),
            origin=(0.5, 0.5),
            scale=0.66,
        )
        self.live_dot = Entity(
            parent=ui,
            model="circle",
            scale=0.009,
            color=color.rgba(*GREEN, 1),
            position=(a / 2 - m - 0.005, 0.5 - m - 0.0135),
        )
        self.mode_b = button(
            ui, "Explore freely", a / 2 - m - 0.15, 0.5 - m - 0.036, 0.15, 0.03, actions["mode"]
        )

        # chapter rail
        self.rail: list[Entity] = []
        self.rail_x = a / 2 - m + 0.002
        top = 0.03 * (chapters - 1) / 2
        for i in range(chapters):
            d = Button(
                parent=ui,
                model="circle",
                color=color.rgba(1, 1, 1, 0.25),
                highlight_color=color.rgba(*CYAN, 0.8),
                scale=0.009,
                position=(self.rail_x, top - i * 0.03, -0.01),
            )
            d.on_click = lambda i=i: actions["chapter"](i)  # type: ignore[misc]
            self.rail.append(d)
        self.rail_label = Text(
            " ", parent=ui, origin=(0.5, 0), position=(self.rail_x - 0.022, 0), scale=0.55
        )

        # cinematic caption, lower left, over a soft scrim that keeps it legible
        cx = -a / 2 + m + 0.004
        self.scrim = Entity(parent=ui, model=scrim(a), z=0.05)
        self.kicker = Text(" ", parent=ui, origin=(-0.5, 0.5), position=(cx, -0.105), scale=0.55)
        self.title = Text(
            " ", parent=ui, origin=(-0.5, 0.5), position=(cx - 0.003, -0.13), scale=2.5
        )
        self.caption = Text(
            " ", parent=ui, origin=(-0.5, 0.5), position=(cx, -0.215), scale=0.78, line_height=1.35
        )
        self.scroll_hint = Text(
            faint(spaced("scroll to explore")),
            parent=ui,
            origin=(0, 0),
            position=(0, -0.5 + m + 0.012),
            scale=0.5,
        )

        # voice strip, bottom centre: shown only while the microphone is on
        vy = -0.5 + m + 0.075
        self.voice_orb = Entity(
            parent=ui, model="circle", scale=0.034, position=(0, vy, -0.01), enabled=False
        )
        self.voice_state = Text(
            " ", parent=ui, origin=(0, 0), position=(0, vy - 0.038), scale=0.62, enabled=False
        )
        self.voice_words = Text(
            " ", parent=ui, origin=(0, 0), position=(0, vy + 0.05), scale=1.05, enabled=False
        )
        self.voice_hint = Text(
            " ", parent=ui, origin=(0, 0), position=(0, vy - 0.062), scale=0.48, enabled=False
        )

        # the typed-report box: always works, whatever the microphone does
        self.type_box = Glass(-0.34, -0.5 + m + 0.23, 0.68, 0.058, parent=ui)
        self.type_text = label(self.type_box, " ", 0.02, -0.016, 0.8)
        self.type_box.enabled = False

        self.dock = Dock(actions, speeds)
        self.card = Card(actions["deselect"])
        self.toast = Text(
            " ", parent=ui, origin=(0, 0.5), position=(0, 0.5 - m - 0.004), scale=0.66
        )
        self.toast_until = 0.0

    def report_box(self, active: bool, text: str) -> None:
        """The box for typing a report: shown while ``active``, with a blinking cursor."""
        self.type_box.enabled = active and self.shown
        self.dock.type_b.color = BTN_ON if active else BTN
        if active:
            cursor = "|" if int(time.perf_counter() * 2) % 2 else " "
            self.type_text.text = (
                muted("Type a report, Enter sends, Esc cancels:  ") + tint(WHITE, text) + cursor
            )

    def voice(
        self,
        phase: str,
        words: str,
        level: float,
        vehicle: str | None,
        error: str | None,
        gated: bool = False,
    ) -> None:
        """The voice strip: an orb that breathes while waiting for "hey ACTION" and
        pulses with your voice while listening, with your words live above it."""
        on = phase != "off" or bool(error)
        self.dock.mic_b.color = BTN_ON if phase != "off" else BTN
        for e in (self.voice_orb, self.voice_state, self.voice_words, self.voice_hint):
            e.enabled = on and self.shown
        if not on:
            return
        t = time.perf_counter()
        if phase == "listening":
            colour, size = (RED, 0.034 + 0.03 * min(1.0, level * 3) + 0.004 * math.sin(t * 9))
        elif phase == "waiting":
            colour, size = (CYAN, 0.032 + 0.003 * math.sin(t * 2.2))
        else:
            colour, size = (MUTED, 0.03)
        self.voice_orb.color = color.rgba(*colour, 0.9)
        self.voice_orb.scale = size
        state = {
            "loading": "Loading the speech models…",
            "waiting": 'Say "Hey ACTION"',
            "listening": "Listening",
            "off": "Microphone problem",
        }[phase]
        self.voice_state.text = tint(colour, spaced(state) if phase == "listening" else state)
        self.voice_words.text = tint(WHITE, words) if words else " "
        self.voice_hint.text = (
            tint(
                AMBER,
                "Windows is cutting your microphone signal (noise suppression). "
                "Press Enter to type instead.",
            )
            if gated and not error
            else faint(f"{error}")
            if error
            else faint(
                f"speaking as {vehicle}" if vehicle else "no vehicle yet: press space to play"
            )
            + faint("   ·   click a car to speak as it   ·   V to talk now   ·   M to stop")
        )

    def set_shown(self, shown: bool) -> None:
        """Tab: hide every panel for a clean view (the card too)."""
        self.shown = shown
        self.root.enabled = shown
        self.dock.pill.enabled = shown
        self.dock.panel.enabled = shown and self.dock.open
        if not shown:
            self.card.hide()

    def chapter(
        self,
        progress: float,
        titles: Sequence[str],
        kicker: str,
        title: str,
        caption: str,
        show_caption: bool,
    ) -> None:
        """Rail dots and the caption, which fades between chapters."""
        current = round(progress)
        for i, d in enumerate(self.rail):
            on = i == current
            d.color = color.rgba(*CYAN, 0.95) if on else color.rgba(1, 1, 1, 0.22)
            d.scale = 0.013 if on else 0.008
        self.rail_label.text = tint(MUTED, spaced(titles[current]))
        self.rail_label.y = self.rail[current].y
        frac = abs(progress - current)
        alpha = 0.0 if not show_caption else max(0.0, 1.0 - max(0.0, frac - 0.06) / 0.22)
        for t, text in ((self.kicker, kicker), (self.title, title), (self.caption, caption)):
            if t.text != text:
                t.text = text
            t.color = color.rgba(*(WHITE if t is self.title else MUTED), alpha)
        self.kicker.color = color.rgba(*CYAN, alpha)
        self.kicker.enabled = self.title.enabled = self.caption.enabled = alpha > 0.01
        self.scrim.enabled = alpha > 0.01
        self.scrim.setColorScale(1, 1, 1, alpha)
        self.title.y = -0.13 - (1 - alpha) * 0.012  # a little rise as it fades in
        self.scroll_hint.enabled = current == 0 and show_caption
