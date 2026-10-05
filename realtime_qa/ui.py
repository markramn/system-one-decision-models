import asyncio
import logging
import math
from datetime import datetime
from html import escape
from time import time
from typing import Any, Literal

from nicegui import app, ui
from pydantic import ValidationError

from realtime_qa.audio import WaveformBin, input_devices
from realtime_qa.config import Settings
from realtime_qa.ollama import DecisionError
from realtime_qa.schemas import MetricPoint, Scorecard
from realtime_qa.scoring import build_request, load_scorecard
from realtime_qa.session import Session

logger = logging.getLogger(__name__)

BRAND = {
    "bright": "#1500FF",
    "deep": "#170035",
    "white": "#FFFFFF",
    "cyan": "#00D7D9",
    "green": "#00BC9E",
    "neutral": "#D1DDE9",
    "soft": "#DCE8F2",
    "canvas": "#EAF3F9",
    "line": "#6D73AD",
    "border": "#C6CEE0",
    "alert": "#ED4A5F",
}
COLORS = {
    "met": BRAND["green"],
    "partial": BRAND["cyan"],
    "pending": BRAND["line"],
    "missed": BRAND["alert"],
    "contradicted": BRAND["alert"],
    "review": BRAND["bright"],
    "unavailable": BRAND["line"],
}
LABELS = {
    "met": "Met",
    "partial": "Partial",
    "pending": "Pending",
    "missed": "Missed",
    "contradicted": "Contradicted",
    "review": "Review needed",
    "unavailable": "Unavailable",
}
DEMO_LINES = [
    "Good morning, thank you for calling Northstar Support. My name is Alex.",
    "This call may be recorded for training and quality assurance.",
    "Before I access your account, please confirm your name and postcode.",
    "I am sorry about the duplicate charge. I understand how frustrating that must be.",
    "I will look into this for you and help get it sorted.",
    "I will send a refund request today and email you the reference within one hour.",
    "Is there anything else I can help you with? Thank you for calling. Goodbye.",
]
STYLE = """
body {background:#FFFFFF; color:#170035; font-family:"Manrope","Segoe UI",Arial,sans-serif;}
.nicegui-content {padding:0; gap:0;}
.brandbar {width:100%; min-height:46px; padding:8px 18px; background:#EAF3F9;
 border-bottom:1px solid #C6CEE0; position:sticky; top:0; z-index:20;}
.brand {font-size:14px; font-weight:800; letter-spacing:-.3px;}
.monogram {background:#1500FF; color:#FFFFFF; border-radius:4px; padding:4px 6px;
 font-size:12px; font-weight:800;}
.brand-context {font-size:11px; color:#1500FF; border-left:1px solid #C6CEE0; padding-left:18px;}
.pill {font-size:10px; font-weight:600; border:1px solid #C6CEE0; padding:3px 8px;
 border-radius:4px; color:#170035; background:#FFFFFF;}
.side-rail {position:fixed; top:46px; left:0; bottom:0; width:46px; padding:18px 0;
 border-right:1px solid #C6CEE0; background:#FFFFFF; align-items:center; gap:22px;}
.side-rail a {color:#6D73AD; padding:5px; border-radius:4px;}
.side-rail a:hover, .side-rail a:focus-visible {color:#1500FF; background:#EAF3F9;}
.shell {margin-left:46px; width:calc(100% - 46px); padding:12px 18px 18px; gap:12px;}
.toolbar {width:100%; min-height:34px; gap:10px; justify-content:space-between;}
.page-title {font-size:15px; font-weight:600;}
.small {font-size:11px;}
.muted {color:#170035; opacity:.76;}
.metadata {display:flex; flex-wrap:wrap; gap:10px 22px; width:100%; padding:12px 14px;
 border:1px solid #C6CEE0; border-radius:5px; background:#FFFFFF; font-size:11px;}
.meta-field {gap:5px; align-items:center; min-width:0;}
.meta-key {color:#1500FF; font-size:10px; font-weight:600;}
.meta-value {font-size:11px; overflow-wrap:anywhere; font-variant-numeric:tabular-nums;}
.section-block {width:100%; border:1px solid #C6CEE0; border-radius:6px; gap:0;
 background:#FFFFFF; scroll-margin-top:62px; overflow:hidden;}
.section-heading {width:100%; min-height:35px; padding:8px 12px; gap:10px;
 justify-content:space-between; align-items:center; border-bottom:1px solid #C6CEE0;
 background:color-mix(in srgb,#EAF3F9 38%,#FFFFFF);}
.section-title {font-size:12px; font-weight:600;}
.audio-controls {padding:10px 12px; width:100%; gap:10px; align-items:center;}
.input-mode {width:145px;}
.input-device {flex:1; min-width:230px; max-width:480px;}
.wave-layout {display:grid; grid-template-columns:145px minmax(0,1fr); width:100%;
 padding:0 12px 10px; gap:10px;}
.audio-clock {border:1px solid #C6CEE0; border-radius:4px; align-items:center;
 justify-content:center; min-height:76px; gap:4px; background:#EAF3F9;}
.audio-clock-value {color:#1500FF; font-size:23px; font-weight:600;
 font-variant-numeric:tabular-nums;}
.wave-content {min-width:0; gap:4px; width:100%;}
.oscillogram {width:100%; height:76px; border:1px solid #C6CEE0; border-radius:3px;}
.oscillogram svg {display:block; width:100%; height:100%;}
.speaker-track {width:100%; background:#DCE8F2; border-radius:3px; padding:3px 8px;
 font-size:9px; color:#170035;}
.wave-caption {width:100%; justify-content:space-between; font-size:10px; gap:8px;}
.session-notice {width:100%; padding:0 12px 10px; align-items:center; gap:8px;}
.evaluation-grid {display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);
 width:100%; gap:12px; padding:12px;}
.panel {border:1px solid #C6CEE0; border-radius:5px; background:#FFFFFF;
 width:100%; min-width:0; gap:0; overflow:hidden; box-shadow:none;}
.panel-heading {width:100%; background:#EAF3F9; border-bottom:1px solid #C6CEE0;
 padding:8px 12px; min-height:36px; align-items:center; justify-content:space-between; gap:8px;}
.panel-title {font-size:12px; font-weight:600; color:#170035;}
.form-row {width:100%; padding:12px; align-items:center; justify-content:space-between; gap:8px;}
.form-name {font-size:12px; font-weight:600;}
.evaluation-summary {padding:12px; min-height:82px; justify-content:center; gap:6px; width:100%;}
.metrics-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
 width:100%; padding:12px; gap:12px;}
.metric-card {min-width:0;}
.metric-card .panel-heading {padding:8px 10px; gap:6px;}
.metric-card .panel-title {font-size:11px;}
.metric-value {font-size:12px; font-weight:800; color:#1500FF; white-space:nowrap;}
.metric-chart {width:100%; height:145px; padding:4px 8px 0;}
.metric-chart svg {width:100%; height:100%; display:block;}
.metric-caption {padding:4px 10px 9px; font-size:10px; color:#170035; min-height:42px;
 line-height:1.5; width:100%;}
.word-cloud {width:100%; height:145px; padding:12px; gap:6px 12px; overflow:auto;
 justify-content:center; align-content:center; align-items:center;}
.cloud-word {max-width:100%; overflow-wrap:anywhere; line-height:1.25;}
@media(max-width:1100px) {.metrics-grid {grid-template-columns:repeat(2,minmax(0,1fr));}}
@media(max-width:600px) {.metrics-grid {grid-template-columns:1fr;}}
.analysis-grid {display:grid; grid-template-columns:minmax(0,1.35fr) minmax(340px,1fr);
 width:100%; gap:12px; padding:12px; align-items:stretch;}
.analytics-pane {height:610px; display:flex; flex-direction:column;}
.transcript-head, .transcript-row {display:grid; grid-template-columns:70px 78px minmax(0,1fr);
 gap:10px; width:100%; align-items:start;}
.transcript-head {padding:8px 12px; border-bottom:1px solid #D1DDE9;
 color:#170035; font-size:10px; font-weight:600; background:#FFFFFF;}
.transcript-row {padding:9px 12px; font-size:12px; border-bottom:1px solid #EAF3F9;}
.transcript-speaker {color:#1500FF; font-size:11px; font-weight:600;}
.transcript-time {font-size:10px; color:#170035; opacity:.76; font-variant-numeric:tabular-nums;}
.transcript-text {line-height:1.65; white-space:pre-wrap; overflow-wrap:anywhere;}
.transcript-scroll {width:100%; flex:1; min-height:140px;}
.partial {background:#EAF3F9; border-left:2px solid #00D7D9; font-style:italic;}
.empty-state {width:100%; min-height:190px; align-items:center; justify-content:center;
 text-align:center; padding:20px; gap:8px;}
.transcript-footer {width:100%; border-top:1px solid #C6CEE0; padding:9px 12px; gap:8px;}
.text-demo {width:100%; gap:7px; padding:10px 12px; border-top:1px solid #C6CEE0;}
.qa-summary {width:100%; padding:12px; gap:6px; border-bottom:1px solid #D1DDE9;}
.score-number {font-size:28px; font-weight:800; color:#1500FF; line-height:1;}
.score-phase {font-size:9px; font-weight:600; letter-spacing:.6px; color:#1500FF;}
.qa-scroll {width:100%; flex:1; min-height:180px;}
.qa-scroll .q-scrollarea__content, .transcript-scroll .q-scrollarea__content {padding:0; gap:0;}
.question-group {padding:8px 12px; width:100%; background:#EAF3F9;
 font-size:10px; font-weight:600; color:#170035;}
.question {width:100%; border-bottom:1px solid #D1DDE9; border-left:3px solid #6D73AD;
 padding:9px 12px; gap:5px; background:#FFFFFF;}
.question-text {font-size:12px; font-weight:600; line-height:1.5;}
.status {font-size:10px; font-weight:600; color:#170035; background:#EAF3F9;
 border:1px solid #6D73AD; border-radius:3px; padding:2px 6px;}
.qa-footer {width:100%; padding:10px 12px; border-top:1px solid #C6CEE0; gap:6px;}
.telemetry {width:100%; gap:10px; align-items:center; font-size:10px; flex-wrap:wrap;}
.q-field--outlined .q-field__control:before {border-color:#6D73AD;}
.q-field__label, .q-field__native, .q-field__input {color:#170035!important; font-size:12px;}
.q-field__append {color:#170035;}
.q-btn {border-radius:4px; text-transform:none; letter-spacing:0; font-weight:600; font-size:11px;}
.q-btn:focus-visible, a:focus-visible {outline:2px solid #1500FF; outline-offset:2px;}
.q-checkbox__label {font-size:10px;}
.q-dialog .q-card, .q-menu {background:#FFFFFF; color:#170035;}
.q-dialog .q-card {border:1px solid #C6CEE0; border-radius:8px; padding:20px;}
.q-expansion-item__container>.q-item {min-height:25px; padding:2px 0; font-size:10px;}
.q-linear-progress {border-radius:3px;}
.q-linear-progress__track {background:#D1DDE9!important; opacity:1;}
.error-text {color:#170035; border-left:3px solid #ED4A5F; padding-left:10px;}
.notice-error {background:#FFFFFF; color:#170035; border:1px solid #C6CEE0;
 border-left:3px solid #ED4A5F; padding:10px 14px; border-radius:4px; font-size:12px;}
.footer {border-top:1px solid #C6CEE0; padding-top:12px; width:100%; gap:8px;}
@media(max-width:1000px) {
 .analysis-grid {grid-template-columns:minmax(0,1fr) minmax(300px,1fr);}
 .shell {padding:10px;} .brand-context {display:none;}
}
@media(max-width:760px) {
 .side-rail {display:none;} .shell {margin-left:0; width:100%; padding:10px;}
 .brandbar {padding:8px 10px;} .analysis-grid, .evaluation-grid {grid-template-columns:1fr;}
 .analytics-pane {height:570px;} .wave-layout {grid-template-columns:1fr;}
 .audio-clock {min-height:38px; flex-direction:row; gap:12px;}
 .input-device {min-width:180px; max-width:none;} .metadata {gap:10px 16px;}
 .transcript-head, .transcript-row {grid-template-columns:52px 62px minmax(0,1fr); gap:6px;}
}
"""


def clock(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def waveform_svg(bins: tuple[WaveformBin, ...]) -> str:
    valid = [
        point
        for point in bins[-400:]
        if all(math.isfinite(v) for v in (point.start, point.end, point.minimum, point.maximum))
    ]
    end = valid[-1].end if valid else 0
    start = max(0, end - 20)
    segments = []
    for point in valid:
        x = max(0, min(1200, ((point.start + point.end) / 2 - start) * 60))
        top = 40 - max(-1, min(1, point.maximum)) * 32
        bottom = 40 - max(-1, min(1, point.minimum)) * 32
        segments.append(f"M{x:.1f},{top:.1f}V{bottom:.1f}")
    path = (
        f'<path d="{" ".join(segments)}" stroke="{BRAND["bright"]}" '
        'stroke-width="1.4" stroke-linecap="round" fill="none"/>'
        if segments
        else ""
    )
    cursor = max(0, min(1200, (end - start) * 60))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 80" '
        'preserveAspectRatio="none" role="img" aria-label="Live microphone amplitude waveform">'
        f'<line x1="0" y1="40" x2="1200" y2="40" stroke="{BRAND["border"]}"/>'
        f'{path}<line x1="{cursor:.1f}" y1="0" x2="{cursor:.1f}" y2="80" '
        f'stroke="{BRAND["deep"]}" stroke-width="1"/></svg>'
    )


def metric_chart_svg(
    points: list[MetricPoint],
    *,
    label: str,
    kind: Literal["line", "bar"] = "line",
    minimum: float = 0,
    maximum: float | None = None,
    unit: str = "",
) -> str:
    points = points[-256:]
    values = [p.value for p in points if p.value is not None and math.isfinite(p.value)]
    upper = (
        maximum if maximum is not None else max(100, math.ceil(max(values, default=0) / 50) * 50)
    )
    span = max(1, upper - minimum)
    first = max(0, points[0].seconds - 5) if points else 0
    last = max(first + 5, points[-1].seconds) if points else 5

    def x(seconds: float) -> float:
        return 34 + (seconds - first) / (last - first) * 252

    def y(value: float) -> float:
        return 108 - (max(minimum, min(upper, value)) - minimum) / span * 91

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 140" '
        f'role="img" aria-label="{escape(label, quote=True)}">',
        f'<text x="34" y="10" font-size="9" fill="{BRAND["deep"]}">{escape(unit)}</text>',
    ]
    for value in (minimum, minimum + span / 2, upper):
        position = y(value)
        parts.append(
            f'<line x1="34" y1="{position:.1f}" x2="286" y2="{position:.1f}" '
            f'stroke="{BRAND["soft"]}"/><text x="29" y="{position + 3:.1f}" '
            f'text-anchor="end" font-size="9" fill="{BRAND["deep"]}">{value:g}</text>'
        )
    for position in (first, (first + last) / 2, last):
        parts.append(
            f'<text x="{x(position):.1f}" y="127" text-anchor="middle" '
            f'font-size="9" fill="{BRAND["deep"]}">{clock(position)}</text>'
        )
    path = []
    connected = False
    previous = first
    for point in points:
        if point.value is None:
            connected = False
            previous = point.seconds
            continue
        px, py = x(point.seconds), y(point.value)
        if kind == "bar":
            width = max(1, (x(point.seconds) - x(previous)) * 0.7)
            left = x(previous) + (x(point.seconds) - x(previous) - width) / 2
            parts.append(
                f'<rect x="{left:.1f}" y="{py:.1f}" width="{width:.1f}" '
                f'height="{max(0, y(minimum) - py):.1f}" fill="{BRAND["bright"]}"/>'
            )
        else:
            path.append(f"{'L' if connected else 'M'}{px:.1f},{py:.1f}")
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2" fill="{BRAND["bright"]}"/>')
            connected = True
        previous = point.seconds
    if path:
        parts.append(
            f'<path d="{" ".join(path)}" fill="none" '
            f'stroke="{BRAND["bright"]}" stroke-width="1.5"/>'
        )
    if not values:
        parts.append(
            '<text x="160" y="68" text-anchor="middle" font-size="11" '
            f'fill="{BRAND["deep"]}">No data yet</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def build_dashboard(session: Session, devices: list[dict[str, Any]]) -> None:
    ui.add_css(STYLE)
    ui.dark_mode().disable()
    ui.colors(
        primary=BRAND["bright"],
        secondary=BRAND["deep"],
        accent=BRAND["cyan"],
        positive=BRAND["green"],
        negative=BRAND["alert"],
        warning=BRAND["alert"],
        info=BRAND["bright"],
        dark=BRAND["deep"],
    )
    last_render: tuple | None = None
    last_score: tuple | None = None
    last_wave: tuple | None = None
    last_metrics: tuple | None = None
    last_terms: tuple | None = None
    question_widgets: dict[str, tuple] = {}
    demo_index = 0

    async def start() -> None:
        try:
            await session.start(mode.value, microphone.value)
        except (DecisionError, ValueError) as exc:
            ui.notify(str(exc), type="negative", color=BRAND["deep"], text_color=BRAND["white"])
        except Exception:
            logger.exception("Dashboard start failed")
            ui.notify(
                session.message, type="negative", color=BRAND["deep"], text_color=BRAND["white"]
            )
        refresh()

    async def new_session() -> None:
        nonlocal demo_index
        await session.reset()
        demo_index = 0
        refresh()

    def add_text() -> None:
        try:
            session.add_text(text.value or "")
            text.set_value("")
        except ValueError as exc:
            ui.notify(str(exc), type="warning", color=BRAND["deep"], text_color=BRAND["white"])
        refresh()

    def next_line() -> None:
        nonlocal demo_index
        if demo_index < len(DEMO_LINES):
            try:
                session.add_text(DEMO_LINES[demo_index])
                demo_index += 1
            except ValueError as exc:
                ui.notify(str(exc), type="warning", color=BRAND["deep"], text_color=BRAND["white"])
        refresh()

    def export() -> None:
        ui.download.content(
            session.export(), f"qa-{session.session_id[:8]}.json", "application/json"
        )

    def export_transcript() -> None:
        ui.download.content(
            session.transcript, f"transcript-{session.session_id[:8]}.txt", "text/plain"
        )

    def settings_dialog() -> None:
        if session.state != "idle":
            return
        with ui.dialog() as dialog, ui.card().classes("w-full").style("max-width:850px"):
            ui.label("Demo settings").classes("panel-title")
            ui.label(
                "Changes apply to this session. Download the scorecard to keep edits."
            ).classes("muted small")
            model = (
                ui.input("Ollama decision model", value=session.settings.model)
                .props("outlined")
                .classes("w-full")
            )
            silence_threshold = (
                ui.number(
                    "Silence threshold (dBFS)",
                    value=session.settings.silence_threshold_dbfs,
                    min=-80,
                    max=-10,
                    step=1,
                )
                .props("outlined")
                .classes("w-full")
            )
            ui.label(
                "RMS estimate only; adjust for microphone noise. Lower values count less silence."
            ).classes("small muted")
            editor = (
                ui.textarea("Scorecard JSON", value=session.card.model_dump_json(indent=2))
                .props("outlined")
                .classes("w-full")
                .style("max-height:460px;overflow:auto")
            )
            error = ui.label().classes("error-text small")

            def save() -> None:
                if session.state != "idle":
                    return
                try:
                    raw = editor.value or ""
                    if len(raw.encode()) > 64_000:
                        raise ValueError("Scorecard exceeds 64 KB")
                    card = Scorecard.model_validate_json(raw)
                    settings = Settings.model_validate(
                        {
                            **session.settings.model_dump(),
                            "model": model.value,
                            "silence_threshold_dbfs": silence_threshold.value,
                        }
                    )
                    build_request(settings, card, "", final=True)
                    session.settings.model = settings.model
                    session.settings.silence_threshold_dbfs = settings.silence_threshold_dbfs
                    session.card = card
                    questions.refresh()
                    dialog.close()
                    refresh()
                except (ValidationError, ValueError) as exc:
                    error.set_text(str(exc).splitlines()[0])

            with ui.row().classes("w-full justify-end"):
                ui.button(
                    "Download scorecard",
                    on_click=lambda: ui.download.content(
                        editor.value or "", "scorecard.json", "application/json"
                    ),
                ).props("flat")
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Apply", on_click=save).props("unelevated text-color=white")
        dialog.open()

    @ui.refreshable
    def transcript() -> None:
        if not session.view.utterances and not session.view.partial:
            with ui.column().classes("empty-state"):
                ui.icon("subtitles", size="28px").style("color:#6D73AD")
                ui.label("Waiting for the conversation").classes("panel-title")
                ui.label("Start a session. Recognized speech will appear here live.").classes(
                    "small muted"
                )
        for utterance in session.view.utterances:
            with ui.element("div").classes("transcript-row"):
                ui.label("Agent").classes("transcript-speaker")
                ui.label(clock(utterance.start)).classes("transcript-time")
                ui.label(utterance.text).classes("transcript-text")
        if session.view.partial:
            with ui.element("div").classes("transcript-row partial"):
                ui.label("Agent").classes("transcript-speaker")
                ui.label("LIVE").classes("transcript-time")
                with ui.column().classes("gap-1"):
                    ui.label(session.view.partial).classes("transcript-text")
                    ui.label("Provisional text; may change").classes("small muted")

    @ui.refreshable
    def questions() -> None:
        question_widgets.clear()
        for section in dict.fromkeys(q.section for q in session.card.questions):
            ui.label(section).classes("question-group")
            for question in [q for q in session.card.questions if q.section == section]:
                with ui.column().classes("question") as container:
                    with ui.row().classes("w-full justify-between items-start no-wrap gap-3"):
                        ui.label(question.text).classes("question-text").style("flex:1;min-width:0")
                        points = (
                            ui.label(f"0 / {question.weight:g}")
                            .classes("small muted")
                            .style("white-space:nowrap")
                        )
                    with ui.row().classes("w-full justify-between items-center gap-2"):
                        status = ui.label("Pending").classes("status")
                        probability = ui.label("Awaiting speech").classes("small muted")
                    with ui.expansion("Rubric & decision details").classes("w-full small muted"):
                        for key, criterion in question.criteria.items():
                            ui.label(f"{key.replace('_', ' ').capitalize()}: {criterion}").classes(
                                "small"
                            )
                        detail = (
                            ui.label("No model result yet.")
                            .classes("small")
                            .style("white-space:pre-wrap")
                        )
                    question_widgets[question.id] = (container, points, status, probability, detail)

    @ui.refreshable
    def history() -> None:
        if not session.history:
            ui.label("Decision changes will appear here.").classes("muted small")
        for event in reversed(list(session.history)[-16:]):
            ui.label(
                f"{clock(event['at_seconds'])} / {event['question'].replace('_', ' ')}"
                f" / {LABELS[event['status']]}"
            ).classes("small muted")

    @ui.refreshable
    def word_cloud(words: list[tuple[str, int]]) -> None:
        with ui.row().classes("word-cloud"):
            if not words:
                ui.label("Waiting for finalized words").classes("small muted")
            for index, (word, count) in enumerate(words):
                size = 14 + min(14, math.log2(count) * 5)
                color = BRAND["bright"] if index % 3 == 0 else BRAND["deep"]
                ui.label(word).classes("cloud-word").style(
                    f"font-size:{size:.0f}px;font-weight:600;color:{color}"
                ).tooltip(f"{count} occurrence{'s' if count != 1 else ''}")

    def metadata_field(name: str, value: str) -> ui.label:
        with ui.row().classes("meta-field"):
            ui.label(f"{name}:").classes("meta-key")
            return ui.label(value).classes("meta-value")

    with ui.row().classes("brandbar items-center justify-between"):
        with ui.row().classes("items-center gap-3"):
            ui.label("S1").classes("monogram")
            ui.label("RealtimeQA").classes("brand")
            ui.label("Recording / Call details").classes("brand-context")
        ui.label("LOCAL QA SHOWCASE").classes("pill")
    with ui.column().classes("side-rail"):
        for icon, label, target in [
            ("call", "Call information", "call-information"),
            ("graphic_eq", "Call audio", "call-audio"),
            ("assignment", "Quality assurance", "quality-assurance"),
            ("insights", "Visual analytics", "visual-analytics"),
            ("analytics", "Live analytics", "live-analytics"),
        ]:
            with ui.link(target=f"#{target}").props(f'aria-label="{label}"'):
                ui.icon(icon, size="19px")
                ui.tooltip(label)
    with ui.column().classes("shell"):
        with ui.row().classes("toolbar items-center"):
            ui.label("Call details").classes("page-title")
            with ui.row().classes("gap-2 items-center"):
                new_button = ui.button(
                    "New session", icon="restart_alt", on_click=new_session
                ).props("flat dense color=secondary")
                export_button = ui.button("Export JSON", icon="download", on_click=export).props(
                    "flat dense"
                )
                settings_button = ui.button(
                    "Settings", icon="tune", on_click=settings_dialog
                ).props("flat dense color=secondary")
        with ui.element("section").classes("metadata").props("id=call-information"):
            metadata_field("Assigned to", "Local agent")
            call_id = metadata_field("Call ID", session.session_id[:8])
            metadata_field("Phone number", "Not connected")
            metadata_field("Extension", "Not connected")
            call_started = metadata_field("Start time", "Not started")
            call_duration = metadata_field("Duration", "00:00")
            metadata_field("Direction", "Local simulation")
        with ui.column().classes("section-block").props("id=call-audio"):
            with ui.row().classes("section-heading"):
                ui.label("Live call audio").classes("section-title")
                ui.label("Microphone capture / audio is not saved").classes("small muted")
            with ui.row().classes("audio-controls"):
                mode = (
                    ui.select(
                        {"microphone": "Microphone", "text": "Text demo"},
                        value="microphone",
                        label="Input mode",
                    )
                    .props("outlined dense")
                    .classes("input-mode")
                    .mark("input-mode")
                )
                choices = {d["id"]: f"{d['name']} / {d['host']}" for d in devices}
                default = next(
                    (d["id"] for d in devices if d["default"]), next(iter(choices), None)
                )
                microphone = (
                    ui.select(choices, value=default, label="Microphone on this PC")
                    .props("outlined dense")
                    .classes("input-device")
                )
                start_button = ui.button("Start session", icon="mic", on_click=start).props(
                    "unelevated text-color=white"
                )
                end_button = ui.button("End session", icon="stop", on_click=session.end).props(
                    "outline color=secondary"
                )
            with ui.element("div").classes("wave-layout"):
                with ui.column().classes("audio-clock"):
                    timer_label = ui.label("00:00").classes("audio-clock-value")
                    audio_status = ui.label("Mic idle").classes("small muted")
                with ui.column().classes("wave-content"):
                    waveform = ui.html(waveform_svg(())).classes("oscillogram").mark("oscillogram")
                    ui.label("CHANNEL / AGENT ONLY").classes("speaker-track")
                    with ui.row().classes("wave-caption muted"):
                        wave_range = ui.label("No microphone audio yet")
                        ui.label("Amplitude envelope / recent 20 seconds")
            with ui.row().classes("session-notice"):
                state_badge = ui.badge("IDLE", color="secondary")
                message = ui.label(session.message).classes("small muted")
        error_label = ui.label().classes("notice-error w-full")
        error_label.set_visibility(False)
        with ui.column().classes("section-block").props("id=quality-assurance"):
            with ui.row().classes("section-heading"):
                ui.label("Quality assurance").classes("section-title")
                edit_card_button = ui.button("Edit scorecard", on_click=settings_dialog).props(
                    "flat dense"
                )
            with ui.element("div").classes("evaluation-grid"):
                with ui.column().classes("panel"):
                    with ui.row().classes("panel-heading"):
                        ui.label("Available evaluation forms").classes("panel-title")
                    with ui.row().classes("form-row"):
                        with ui.column().classes("gap-1"):
                            scorecard_name = ui.label(session.card.name).classes("form-name")
                            form_details = ui.label().classes("small muted")
                        ui.badge("Selected", color="primary")
                with ui.column().classes("panel"):
                    with ui.row().classes("panel-heading"):
                        ui.label("Evaluation status").classes("panel-title")
                        evaluation_badge = ui.label("NOT STARTED").classes("score-phase")
                    with ui.column().classes("evaluation-summary"):
                        evaluation_title = ui.label("No completed evaluation").classes("form-name")
                        evaluation_detail = ui.label(
                            "Start a session to evaluate this scorecard live."
                        ).classes("small muted")
        with ui.column().classes("section-block").props("id=visual-analytics"):
            with ui.row().classes("section-heading"):
                ui.label("Visual analytics").classes("section-title")
                ui.label("Agent-only / live estimates").classes("small muted")
            with ui.element("div").classes("metrics-grid"):
                with ui.column().classes("panel metric-card").props("id=sentiment-card"):
                    with ui.row().classes("panel-heading"):
                        ui.label("Agent sentiment").classes("panel-title")
                        sentiment_value = ui.label("--").classes("metric-value")
                    sentiment_chart = ui.html(
                        metric_chart_svg(
                            [],
                            label="Agent sentiment index",
                            minimum=-100,
                            maximum=100,
                            unit="Index",
                        )
                    ).classes("metric-chart")
                    sentiment_note = ui.label("Awaiting speech / Nimble").classes("metric-caption")
                    sentiment_value.tooltip(
                        "Index: 100 × (P(positive) − P(negative)). "
                        "Not a percentage or calibrated accuracy."
                    )
                with ui.column().classes("panel metric-card").props("id=cadence-card"):
                    with ui.row().classes("panel-heading"):
                        ui.label("Average cadence").classes("panel-title")
                        cadence_value = ui.label("--").classes("metric-value")
                    cadence_chart = ui.html(
                        metric_chart_svg([], label="Speaking cadence", kind="bar", unit="WPM")
                    ).classes("metric-chart")
                    cadence_note = ui.label("Waiting for microphone speech").classes(
                        "metric-caption"
                    )
                    cadence_value.tooltip(
                        "Finalized words per captured minute, including pauses. "
                        "Five-second chart bins estimate timing uniformly within utterances."
                    )
                with ui.column().classes("panel metric-card").props("id=word-cloud-card"):
                    with ui.row().classes("panel-heading"):
                        ui.label("Interaction word cloud").classes("panel-title")
                        cloud_value = ui.label("0 terms").classes("metric-value")
                    word_cloud([])
                    cloud_note = ui.label("Finalized text / filler words omitted").classes(
                        "metric-caption"
                    )
                with ui.column().classes("panel metric-card").props("id=silence-card"):
                    with ui.row().classes("panel-heading"):
                        ui.label("Call silence").classes("panel-title")
                        silence_value = ui.label("--").classes("metric-value")
                    silence_chart = ui.html(
                        metric_chart_svg(
                            [],
                            label="Estimated microphone silence",
                            maximum=5,
                            unit="Quiet seconds / 5s",
                        )
                    ).classes("metric-chart")
                    silence_note = ui.label("Waiting for microphone audio").classes(
                        "metric-caption"
                    )
                    silence_value.tooltip(
                        "Acoustic RMS threshold estimate, not speech detection. "
                        "Noise and quiet speech can affect this estimate."
                    )
        with ui.column().classes("section-block").props("id=live-analytics"):
            with ui.row().classes("section-heading"):
                ui.label("Automated analytics tools").classes("section-title")
                input_label = ui.label("SHERPA-ONNX / CPU").classes("small muted")
            with ui.element("div").classes("analysis-grid"):
                with ui.column().classes("panel analytics-pane transcript-pane"):
                    with ui.row().classes("panel-heading"):
                        ui.label("Interaction transcript").classes("panel-title")
                        with ui.row().classes("gap-2 items-center"):
                            auto_scroll = ui.checkbox("Auto-scroll", value=True).props("dense")
                            transcript_export = ui.button(
                                "Export transcript", icon="download", on_click=export_transcript
                            ).props("flat dense")
                    with ui.element("div").classes("transcript-head"):
                        for label in ["Speaker", "Time", "Live transcript"]:
                            ui.label(label)
                    with ui.scroll_area().classes("transcript-scroll") as transcript_area:
                        transcript()
                    with ui.column().classes("text-demo") as text_panel:
                        ui.label(
                            "Text demo: audio and ASR bypassed; decisions use Nimble."
                        ).classes("small muted")
                        text = (
                            ui.textarea("Agent utterance")
                            .props("outlined autogrow rows=2")
                            .classes("w-full")
                            .mark("agent-text")
                        )
                        with ui.row().classes("gap-2"):
                            add_button = ui.button("Add utterance", on_click=add_text).props(
                                "outline dense"
                            )
                            demo_button = ui.button("Next scripted line", on_click=next_line).props(
                                "flat dense"
                            )
                    with ui.column().classes("transcript-footer"):
                        ui.label(
                            "All speech is treated as the agent. No speaker identification."
                        ).classes("small muted")
                with ui.column().classes("panel analytics-pane qa-pane"):
                    with ui.row().classes("panel-heading"):
                        ui.label("Live QA scoring").classes("panel-title")
                        scoring_state = ui.label("Waiting for speech").classes("small muted")
                    with ui.column().classes("qa-summary"):
                        with ui.row().classes("w-full justify-between items-center"):
                            with ui.row().classes("items-baseline gap-2"):
                                total = ui.label("0").classes("score-number")
                                maximum = ui.label(f"/ {session.card.maximum:g}").classes(
                                    "small muted"
                                )
                            score_phase = ui.label("PROVISIONAL").classes("score-phase")
                        score_bar = (
                            ui.linear_progress(value=0, color="primary", show_value=False)
                            .classes("w-full")
                            .style("height:4px")
                        )
                        coverage = ui.label().classes("small muted")
                        score_note = ui.label("Pending questions are not failures.").classes(
                            "small muted"
                        )
                    with ui.scroll_area().classes("qa-scroll"):
                        questions()
                    with ui.column().classes("qa-footer"):
                        with ui.row().classes("telemetry"):
                            latency = ui.label("Decision: --")
                            asr_lag = ui.label("ASR: --")
                            updates = ui.label("Updates: 0")
                        model_label = ui.label(session.settings.model).classes("small muted")
                        result_age = ui.label("No decision yet").classes("small muted")
            with (
                ui.expansion("Pipeline diagnostics & decision history")
                .classes("w-full small")
                .style("padding:0 12px 10px")
            ):
                context_label = ui.label().classes("small muted")
                history()
                with ui.expansion("Exact evaluated transcript").classes("w-full small"):
                    ui.label(
                        "Evaluated context, not an explanation or evidence attribution."
                    ).classes("small muted")
                    evaluated_text = (
                        ui.label("No evaluation yet.")
                        .classes("small")
                        .style("white-space:pre-wrap")
                    )
        with ui.row().classes("footer justify-between items-center"):
            ui.label("LOCAL DEMONSTRATION / NO TELEPHONY CONNECTION").classes("small muted")
            ui.label(
                "No audio saved. Model judgments are not certified compliance results."
            ).classes("small muted")

    def refresh() -> None:
        nonlocal last_render, last_score, last_wave, last_metrics, last_terms
        session.tick()
        state_badge.set_text(session.state.upper())
        message.set_text(session.message)
        timer_label.set_text(clock(session.elapsed))
        call_id.set_text(session.session_id[:8])
        call_duration.set_text(clock(session.elapsed))
        call_started.set_text(
            datetime.fromtimestamp(session.started_wall_at)
            .astimezone()
            .strftime("%d %b %Y, %H:%M:%S")
            if session.started_wall_at
            else "Not started"
        )
        idle = session.state == "idle"
        mode.set_enabled(idle)
        microphone.set_enabled(idle and mode.value == "microphone")
        settings_button.set_enabled(idle)
        edit_card_button.set_enabled(idle)
        start_button.set_enabled(idle and (mode.value == "text" or bool(devices)))
        end_button.set_enabled(session.state == "listening")
        new_button.set_enabled(not session.active)
        export_button.set_enabled(bool(session.transcript))
        transcript_export.set_enabled(bool(session.transcript))
        is_text = (mode.value if idle else session.mode) == "text"
        text_panel.set_visibility(is_text)
        can_add = session.state == "listening" and session.mode == "text"
        add_button.set_enabled(can_add)
        demo_button.set_enabled(can_add and demo_index < len(DEMO_LINES))
        input_label.set_text("TEXT DEMO / ASR BYPASSED" if is_text else "SHERPA-ONNX / CPU")
        audio_status.set_text(
            "Audio bypassed"
            if is_text
            else "Clipping"
            if session.audio_view.clipping
            else "Listening"
            if session.state == "listening"
            else "Mic idle"
        )
        bins = () if is_text else session.audio_view.waveform
        wave_stamp = (session.session_id, is_text, len(bins), bins[-1].end if bins else 0)
        if wave_stamp != last_wave:
            waveform.set_content(waveform_svg(bins))
            wave_range.set_text(
                "Text demo: no audio waveform"
                if is_text
                else (
                    f"{clock(max(0, bins[-1].end - 20))} - {clock(bins[-1].end)}"
                    if bins
                    else "No microphone audio yet"
                )
            )
            last_wave = wave_stamp
        errors = session.scoring_error
        if session.incomplete:
            errors = "Incomplete capture: dropped audio or recognizer error. " + errors
        error_label.set_text(errors)
        error_label.set_visibility(bool(errors))
        rendered = (session.session_id, session.revision, session.state)
        if rendered != last_render:
            transcript.refresh()
            if auto_scroll.value:
                transcript_area.scroll_to(percent=1)
            last_render = rendered
        snapshot = session.snapshot
        stats = session.audio_view.stats
        metric_stamp = (
            session.session_id,
            session.revision,
            int(stats.captured_seconds * 2),
            is_text,
            snapshot.completed_at if snapshot else None,
            session.scoring_error,
            session.incomplete,
            session.settings.silence_threshold_dbfs,
        )
        if metric_stamp != last_metrics:
            metrics = session.transcript_metrics
            sentiment = snapshot.sentiment if snapshot else None
            current_index = sentiment.index if sentiment and not session.scoring_error else None
            sentiment_value.set_text(f"{current_index:+.0f}" if current_index is not None else "--")
            if session.scoring_error:
                sentiment_caption = "Unavailable / previous decisions shown"
            elif sentiment:
                sentiment_caption = {
                    "positive": "Positive",
                    "negative": "Negative",
                    "neutral": "Neutral",
                    "unclear": "Insufficient evidence",
                    "review": "Review needed",
                    "unavailable": "No valid sentiment answer",
                }[
                    sentiment.status
                ] + f" / cumulative text / revision {snapshot.revision}/{session.revision}"
            else:
                sentiment_caption = "Awaiting speech / Nimble"
            sentiment_note.set_text(sentiment_caption)
            sentiment_chart.set_content(
                metric_chart_svg(
                    list(session.sentiment_history),
                    label="Agent sentiment index",
                    minimum=-100,
                    maximum=100,
                    unit="Index",
                )
            )
            cadence_value.set_text(
                f"{metrics.cadence_wpm:.0f} wpm" if metrics.cadence_wpm is not None else "--"
            )
            cadence_chart.set_content(
                metric_chart_svg(metrics.cadence, label="Speaking cadence", kind="bar", unit="WPM")
            )
            cadence_note.set_text(
                "Microphone required / text demo"
                if is_text
                else "Unavailable / incomplete capture"
                if session.incomplete
                else f"{metrics.word_count} finalized words / 5s timing estimates"
                if metrics.word_count
                else "Waiting for finalized microphone speech"
            )
            cloud_value.set_text(f"{len(metrics.words)} terms")
            cloud_note.set_text(f"{metrics.word_count} finalized words / common words omitted")
            term_stamp = (session.session_id, tuple(metrics.words))
            if term_stamp != last_terms:
                word_cloud.refresh(metrics.words)
                last_terms = term_stamp
            audio_usable = not is_text and not session.incomplete and stats.captured_seconds > 0
            silence_value.set_text(
                f"{100 * stats.quiet_seconds / stats.captured_seconds:.0f}%"
                if audio_usable
                else "--"
            )
            silence_chart.set_content(
                metric_chart_svg(
                    [
                        MetricPoint(seconds=p.start + p.duration, value=p.quiet_seconds)
                        for p in stats.intervals
                    ]
                    if audio_usable
                    else [],
                    label="Estimated microphone silence",
                    maximum=5,
                    unit="Quiet seconds / 5s",
                )
            )
            silence_note.set_text(
                "Microphone required / text demo"
                if is_text
                else "Unavailable / incomplete capture"
                if session.incomplete
                else f"RMS below {session.settings.silence_threshold_dbfs:g} dBFS / estimate"
                if audio_usable
                else "Waiting for microphone audio"
            )
            last_metrics = metric_stamp
        scorecard_name.set_text(session.card.name)
        form_details.set_text(
            f"{len(session.card.questions)} questions / "
            f"{len({q.section for q in session.card.questions})} sections / "
            f"{session.card.maximum:g} points"
        )
        model_label.set_text(session.settings.model)
        maximum.set_text(f"/ {session.card.maximum:g}")
        total.set_text(f"{snapshot.earned:g}" if snapshot else "0")
        score_bar.set_value(snapshot.earned / session.card.maximum if snapshot else 0)
        coverage.set_text(
            f"{len(session.card.questions) - snapshot.unresolved if snapshot else 0}"
            f" / {len(session.card.questions)} assessed"
        )
        phase = (
            "FINAL"
            if session.authoritative
            else ("REVIEW REQUIRED" if session.state == "ended" else "PROVISIONAL")
        )
        score_phase.set_text(phase)
        evaluation_badge.set_text("NOT STARTED" if idle else phase)
        evaluation_title.set_text(
            "Evaluation complete"
            if session.authoritative
            else "Evaluation requires review"
            if session.state == "ended"
            else "Live evaluation in progress"
            if session.active
            else "No completed evaluation"
        )
        if session.scoring_error:
            detail_text = "Latest decision unavailable. Any earlier scores may be stale."
        elif snapshot and snapshot.final:
            detail_text = (
                f"{snapshot.earned:g} / {session.card.maximum:g} points. "
                "Review model judgments before relying on them."
            )
        elif session.state == "ended":
            detail_text = "No final evaluation is available. Check the session status."
        elif snapshot:
            detail_text = (
                f"{snapshot.earned:g} / {session.card.maximum:g} provisional points. "
                "End the session to finalize."
            )
        elif session.active:
            detail_text = "Waiting for speech and the first model decision."
        else:
            detail_text = "Start a session to evaluate this scorecard live."
        evaluation_detail.set_text(detail_text)
        score_note.set_text(
            "Final model judgment; review before relying on it."
            if session.authoritative
            else (
                "Lower-bound subtotal; unresolved questions need review."
                if snapshot and snapshot.unresolved
                else "Pending questions are not failures."
            )
        )
        latency.set_text(
            f"Decision: {snapshot.latency_ms / 1000:.2f}s" if snapshot else "Decision: --"
        )
        asr_lag.set_text(
            f"ASR: {session.audio_view.lag_ms:.0f}ms"
            if session.mode == "microphone" and session.started_at
            else "ASR: --"
        )
        updates.set_text(f"Updates: {session.updates}")
        running = (
            bool(session.request_task and not session.request_task.done())
            or session.state == "finalizing"
        )
        scoring_state.set_text(
            "Unavailable"
            if session.scoring_error
            else "Evaluating..."
            if running
            else "Up to date"
            if snapshot and snapshot.revision == session.revision
            else "Waiting for speech"
            if not snapshot
            else "Update pending"
        )
        result_age.set_text(
            (
                f"Evaluated revision {snapshot.revision}/{session.revision}"
                f" / {max(0, time() - snapshot.completed_at):.0f}s ago"
                if snapshot
                else "No decision yet"
            )
            + (" / evaluating..." if running else "")
        )
        context_label.set_text(
            f"Prompt budget: {session.prompt_fraction:.0%}"
            f" / Short demo limit: {int(session.settings.max_session_seconds)}s"
        )
        score_stamp = (
            session.session_id,
            snapshot.completed_at if snapshot else None,
            id(session.card),
        )
        if score_stamp != last_score:
            results = {r.question_id: r for r in snapshot.results} if snapshot else {}
            for question_id, (
                container,
                points,
                status,
                probability,
                detail,
            ) in question_widgets.items():
                result = results.get(question_id)
                key = result.status if result else "pending"
                color = COLORS[key]
                container.style(f"border-left-color:{color}")
                status.set_text(LABELS[key])
                status.style(f"border-color:{color}")
                if result:
                    points.set_text(f"{result.points:g} / {result.maximum:g}")
                else:
                    weight = next(q.weight for q in session.card.questions if q.id == question_id)
                    points.set_text(f"0 / {weight:g}")
                if result and result.decision:
                    decision = result.decision
                    probability.set_text(
                        f"P({decision.choice.replace('_', ' ')}) "
                        f"{decision.probabilities[decision.choice]:.0%}"
                    )
                    detail.set_text(
                        "\n".join(f"{k}: {v:.3f}" for k, v in decision.probabilities.items())
                        + f"\nDistribution concentration: {decision.confidence:.3f}"
                    )
                else:
                    probability.set_text("No valid decision" if result else "Awaiting speech")
                    detail.set_text("No model result yet.")
            evaluated_text.set_text(snapshot.transcript if snapshot else "No evaluation yet.")
            history.refresh()
            last_score = score_stamp

    mode.on_value_change(lambda: refresh())
    ui.timer(0.15, refresh)
    refresh()


def run_dashboard(settings: Settings) -> None:
    session = Session(settings, load_scorecard(settings.scorecard_path))

    @ui.page("/")
    async def page() -> None:
        client = ui.context.client
        connecting = ui.label("Connecting to the local call details demo...")
        await client.connected()
        connecting.delete()
        if session.owner_id is not None and session.owner_id != client.id:
            ui.label("This local demo is already open in another tab. Close it, then reload here.")
            return
        session.owner_id = client.id

        async def disconnect() -> None:
            if session.owner_id == client.id:
                await session.reset()
                session.owner_id = None

        client.on_disconnect(disconnect)
        try:
            devices = await asyncio.to_thread(input_devices)
        except Exception:
            logger.exception("Could not enumerate microphone devices")
            devices = []
        build_dashboard(session, devices)

    app.on_shutdown(session.close)
    ui.run(
        host="127.0.0.1",
        port=settings.port,
        title="RealtimeQA | Call details",
        dark=False,
        reload=False,
        show=False,
        on_air=False,
        favicon=(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            '<rect width="64" height="64" rx="16" fill="#1500FF"/>'
            '<path d="M16 32h8l5-13 8 27 4-14h7" stroke="#FFFFFF" '
            'stroke-width="5" fill="none"/></svg>'
        ),
    )
