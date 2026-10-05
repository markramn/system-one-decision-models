import json
import os
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright

pytestmark = pytest.mark.skipif(
    os.environ.get("REALTIME_QA_BROWSER_TEST") != "1",
    reason="Opt-in test against a running local UI and real Ollama; never uses the microphone",
)


def test_real_browser_text_demo() -> None:
    base_url = os.environ.get("REALTIME_QA_BROWSER_URL", "http://127.0.0.1:8088")
    errors = []
    external_requests = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1100})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                external_requests.append(request.url)
                if not request.url.startswith(("http://127.0.0.1:", "ws://127.0.0.1:"))
                else None
            ),
        )
        page.goto(base_url)
        assert page.request.get(f"{base_url}/favicon.ico").status == 200
        expect(page.get_by_text("RealtimeQA", exact=True)).to_be_visible(timeout=15000)
        expect(page.locator("body")).to_have_css("background-color", "rgb(255, 255, 255)")
        expect(page.locator(".brandbar")).to_have_css("background-color", "rgb(234, 243, 249)")
        expect(page.locator(".page-title")).to_have_text("Call details")
        expect(page.locator(".panel").first).to_have_css("background-color", "rgb(255, 255, 255)")
        sections = [
            page.locator(f"#{name}").bounding_box()
            for name in (
                "call-information",
                "call-audio",
                "quality-assurance",
                "visual-analytics",
                "live-analytics",
            )
        ]
        assert all(sections[i]["y"] < sections[i + 1]["y"] for i in range(4))
        expect(page.locator(".metric-card")).to_have_count(4)
        cards = [
            page.locator(f"#{name}").bounding_box()
            for name in ("sentiment-card", "cadence-card", "word-cloud-card", "silence-card")
        ]
        assert max(c["y"] for c in cards) - min(c["y"] for c in cards) < 2
        left = page.locator(".transcript-pane").bounding_box()
        right = page.locator(".qa-pane").bounding_box()
        assert left["x"] < right["x"]
        assert abs(left["y"] - right["y"]) < 2
        expect(page.locator(".oscillogram svg")).to_be_visible()
        page.get_by_label("Input mode", exact=True).click()
        page.get_by_role("option", name="Text demo", exact=True).click()
        expect(page.get_by_text("Text demo: no audio waveform", exact=True)).to_be_visible()
        expect(page.locator(".oscillogram path")).to_have_count(0)
        expect(page.get_by_role("button", name="Start session", exact=True)).to_have_css(
            "background-color", "rgb(21, 0, 255)"
        )
        page.get_by_role("button", name="Start session", exact=True).click()
        expect(page.get_by_text("LISTENING", exact=True)).to_be_visible(timeout=120000)
        page.get_by_label("Agent utterance", exact=True).fill(
            "Good morning. Thank you for calling Northstar Support. My name is Alex. "
            "This call may be recorded for training and quality assurance."
        )
        page.get_by_role("button", name="Add utterance", exact=True).click()
        expect(page.get_by_text("Met", exact=True).first).to_be_visible(timeout=120000)
        page.get_by_role("button", name="End session", exact=True).click()
        expect(page.get_by_text("ENDED", exact=True)).to_be_visible(timeout=120000)
        with page.expect_download() as download_info:
            page.get_by_role("button", name="Export JSON", exact=True).click()
        exported = json.loads(Path(download_info.value.path()).read_text(encoding="utf-8"))
        assert exported["input_mode"] == "text"
        assert exported["score"]["final"]
        assert exported["score"]["revision"] == exported["transcript_revision"]
        assert len(exported["score"]["results"]) == 8
        assert exported["score"]["sentiment"]["decision"] is not None
        assert exported["visual_analytics"]["cadence_wpm"] is None
        assert exported["visual_analytics"]["audio"] is None
        assert exported["visual_analytics"]["words"]
        expect(page.locator("#word-cloud-card .cloud-word").first).to_be_visible()
        expect(page.locator("#cadence-card")).to_contain_text("Microphone required")
        expect(page.locator("#silence-card")).to_contain_text("Microphone required")
        screenshot = Path("exports") / "browser-smoke-final.png"
        screenshot.parent.mkdir(exist_ok=True)
        page.screenshot(path=str(screenshot), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        expect(page.locator(".page-title")).to_be_visible()
        assert (
            page.locator(".qa-pane").bounding_box()["y"]
            > page.locator(".transcript-pane").bounding_box()["y"]
        )
        assert (
            page.locator("#silence-card").bounding_box()["y"]
            > page.locator("#sentiment-card").bounding_box()["y"]
        )
        page.screenshot(path="exports/browser-smoke-mobile.png", full_page=True)
        page.get_by_role("button", name="New session", exact=True).click()
        expect(page.get_by_text("IDLE", exact=True)).to_be_visible(timeout=5000)
        expect(page.locator("#word-cloud-card .cloud-word")).to_have_count(0)
        expect(page.locator("#sentiment-card path")).to_have_count(0)
        context.close()
        browser.close()
    assert not errors
    assert not external_requests
