import asyncio

from nicegui import ui
from nicegui.testing import User
from test_session import make_session

from realtime_qa.audio import WaveformBin
from realtime_qa.ui import BRAND, COLORS, STYLE, build_dashboard, metric_chart_svg, waveform_svg


def contrast(foreground: str, background: str) -> float:
    def luminance(color: str) -> float:
        channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in channels]
        return sum(
            value * weight for value, weight in zip(linear, [0.2126, 0.7152, 0.0722], strict=True)
        )

    first, second = sorted([luminance(foreground), luminance(background)])
    return (second + 0.05) / (first + 0.05)


def test_smarsh_palette_and_accessible_text_pairs() -> None:
    assert BRAND["bright"] == "#1500FF"
    assert BRAND["deep"] == "#170035"
    assert BRAND["canvas"] == "#EAF3F9"
    assert 'font-family:"Manrope","Segoe UI",Arial,sans-serif' in STYLE
    assert set(COLORS.values()) <= set(BRAND.values())
    assert COLORS["missed"] == COLORS["contradicted"] == BRAND["alert"]
    for foreground, background in [
        ("white", "bright"),
        ("white", "deep"),
        ("deep", "white"),
        ("deep", "canvas"),
        ("deep", "neutral"),
        ("bright", "white"),
    ]:
        assert contrast(BRAND[foreground], BRAND[background]) >= 4.5


def test_waveform_rendering_is_bounded_and_empty_without_audio() -> None:
    from xml.etree import ElementTree

    empty = ElementTree.fromstring(waveform_svg(()))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert not empty.findall("svg:path", namespace)
    rendered = ElementTree.fromstring(waveform_svg((WaveformBin(0, 0.05, -0.5, 0.75),)))
    path = rendered.find("svg:path", namespace)
    assert path is not None
    assert path.attrib["stroke"] == BRAND["bright"]
    assert "nan" not in path.attrib["d"]
    assert rendered.attrib["role"] == "img"


def test_metric_chart_empty_state_and_uncertain_gap() -> None:
    from xml.etree import ElementTree

    from realtime_qa.schemas import MetricPoint

    empty = metric_chart_svg([], label="Sentiment")
    assert "No data yet" in empty
    points = [
        MetricPoint(seconds=1, value=40),
        MetricPoint(seconds=2),
        MetricPoint(seconds=3, value=-30),
    ]
    svg = ElementTree.fromstring(
        metric_chart_svg(points, label="Sentiment", minimum=-100, maximum=100)
    )
    path = svg.find("{http://www.w3.org/2000/svg}path")
    assert path is not None
    assert path.attrib["d"].count("M") == 2
    assert "L" not in path.attrib["d"]


async def test_audio_metric_cards_and_incomplete_capture(user: User) -> None:
    from realtime_qa.audio import AudioStats, AudioView, SilenceInterval
    from realtime_qa.schemas import Utterance
    from realtime_qa.transcription import TranscriptView

    session, _ = make_session()
    session.view = TranscriptView(
        utterances=(Utterance(id=1, start=0, end=10, text="Refund account payment refund"),)
    )
    session.audio_view = AudioView(
        stats=AudioStats(
            captured_seconds=10,
            quiet_seconds=4,
            intervals=(SilenceInterval(0, 5, 4), SilenceInterval(5, 5, 0)),
        )
    )
    session.state = "ended"

    @ui.page("/")
    def page() -> None:
        build_dashboard(session, [])

    await user.open("/")
    await user.should_see("24 wpm")
    await user.should_see("40%")
    await user.should_see("refund")
    session.incomplete = True
    await asyncio.sleep(0.25)
    await user.should_see("Unavailable / incomplete capture")
    await user.should_not_see("24 wpm")
    await user.should_not_see("40%")
    await session.close()


async def test_text_demo_controls_and_scoring(user: User) -> None:
    session, _ = make_session()

    @ui.page("/")
    def page() -> None:
        build_dashboard(session, [])

    await user.open("/")
    await user.should_see("RealtimeQA")
    await user.should_see("Start session")
    await user.should_see("Call details")
    await user.should_see("Live call audio")
    await user.should_see("Available evaluation forms")
    await user.should_see("Interaction transcript")
    await user.should_see("Live QA scoring")
    await user.should_see("Visual analytics")
    await user.should_see("Agent sentiment")
    await user.should_see("Average cadence")
    await user.should_see("Interaction word cloud")
    await user.should_see("Call silence")
    user.find(marker="input-mode").click()
    user.find("Text demo").click()
    await asyncio.sleep(0.2)
    assert next(iter(user.find(marker="input-mode").elements)).value == "text"
    user.find("Start session").click()
    await asyncio.sleep(0.2)
    assert session.state == "listening"
    user.find(marker="agent-text").type("Hello, my name is Alex.")
    user.find("Add utterance").click()
    await asyncio.sleep(0.3)
    await user.should_see("Hello, my name is Alex.")
    await user.should_see("Text demo")
    user.find("End session").click()
    await asyncio.sleep(0.3)
    assert session.state == "ended"
    assert session.snapshot.final
    await user.should_see("Missed")
    await user.should_see("Evaluation complete")
    user.find("New session").click()
    await asyncio.sleep(0.2)
    assert session.state == "idle"
    assert not session.audio_view.waveform
    await user.should_see("Not started")
    await session.close()
