import json
import unittest

from angler.runtime.jenny_web import (
    FetchedDocument,
    JennyWebExecutor,
    WEB_AFFORDANCE_ID,
    WEB_PAGE_CONTRACT,
    WEB_SEARCH_CONTRACT,
    extract_text,
    parse_search_results,
    validate_public_https_url,
)
from angler.runtime.persistent_autonomy import AffordanceRequest

REF = "sha256:" + "0" * 64

SEARCH_HTML = """
<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fglow&amp;rut=1">Bioluminescence <b>glow</b></a>
<a class="result__snippet" href="x">Light produced by <b>living</b> organisms.</a></div>
<div class="result"><a class="result__a" href="http://insecure.example/x">Insecure</a><a class="result__snippet">nope</a></div>
"""

PAGE_HTML = """<html><head><title>Glow &amp; Light</title><style>p{}</style></head>
<body><script>var hidden = 1;</script><h1>Bioluminescence</h1><p>Light   produced by living organisms.</p>
<p>Second paragraph.</p></body></html>"""


def _request(payload: dict) -> AffordanceRequest:
    return AffordanceRequest(
        idempotency_key=REF,
        trigger_ref=REF,
        affordance_id=WEB_AFFORDANCE_ID,
        observation_ref=REF,
        state_head_ref=REF,
        action_payload=json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )


def _fake_fetch(url, *, timeout_seconds=15.0, data=None):
    if "duckduckgo" in url:
        return FetchedDocument(url, 200, "text/html", SEARCH_HTML)
    return FetchedDocument(url, 200, "text/html; charset=utf-8", PAGE_HTML)


class WebParsingTest(unittest.TestCase):
    def test_extract_text_drops_scripts_and_styles_and_keeps_title(self):
        title, text = extract_text(PAGE_HTML)
        self.assertEqual(title, "Glow & Light")
        self.assertIn("Bioluminescence", text)
        self.assertIn("Light produced by living organisms.", text)
        self.assertNotIn("hidden", text)
        self.assertNotIn("p{}", text)

    def test_search_results_unwrap_and_keep_only_https(self):
        results = parse_search_results(SEARCH_HTML)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.org/glow")
        self.assertEqual(results[0]["title"], "Bioluminescence glow")
        self.assertEqual(results[0]["snippet"], "Light produced by living organisms.")


class WebUrlPolicyTest(unittest.TestCase):
    def test_rejects_non_https_credentials_and_non_public_hosts(self):
        for bad in (
            "http://example.org/",
            "https://user:pw@example.org/",
            "https://localhost/",
            "https://127.0.0.1/",
            "https://10.0.0.5/x",
            "ftp://example.org/",
        ):
            with self.assertRaises(ValueError, msg=bad):
                validate_public_https_url(bad)


class WebExecutorTest(unittest.TestCase):
    def setUp(self):
        self.executor = JennyWebExecutor(_fake_fetch, clock=lambda: "2026-09-06T00:00:00+00:00")

    def test_search_observation_is_bounded_and_content_addressed(self):
        receipt = self.executor(_request({"operation": "search", "query": "bioluminescence", "reading_purpose": "research"}))
        self.assertEqual(receipt.status, "COMPLETED")
        payload = json.loads(receipt.observable_consequence.observation_json)
        self.assertEqual(payload["contract"], WEB_SEARCH_CONTRACT)
        self.assertEqual(payload["result_count"], 1)
        self.assertTrue(payload["content_is_untrusted_evidence"])
        self.assertEqual(receipt.observable_consequence.artifact_refs, (payload["artifact_ref"],))

    def test_read_pages_with_cursor(self):
        url = "https://en.wikipedia.org/wiki/Bioluminescence"
        first = self.executor(_request({"cursor": 0, "max_chars": 20, "operation": "read", "reading_purpose": "research", "url": url}))
        self.assertEqual(first.status, "COMPLETED")
        payload = json.loads(first.observable_consequence.observation_json)
        self.assertEqual(payload["contract"], WEB_PAGE_CONTRACT)
        self.assertEqual(len(payload["content"]), 20)
        self.assertFalse(payload["eof"])
        self.assertEqual(payload["title"], "Glow & Light")
        second = self.executor(_request({"cursor": payload["next_cursor"], "max_chars": 8192, "operation": "read", "reading_purpose": "research", "url": url}))
        self.assertTrue(json.loads(second.observable_consequence.observation_json)["eof"])

    def test_bad_payloads_fail_closed(self):
        with self.assertRaises(ValueError):
            self.executor(_request({"operation": "post", "query": "x", "reading_purpose": "y"}))
        with self.assertRaises(ValueError):
            self.executor(_request({"cursor": 0, "max_chars": 999999, "operation": "read", "reading_purpose": "r", "url": "https://en.wikipedia.org/"}))

    def test_fetch_failure_is_an_error_receipt_not_an_exception(self):
        def failing(url, *, timeout_seconds=15.0, data=None):
            raise OSError("down")
        executor = JennyWebExecutor(failing)
        receipt = executor(_request({"operation": "search", "query": "x", "reading_purpose": "y"}))
        self.assertEqual(receipt.status, "ERROR")
        self.assertIsNone(receipt.observable_consequence)


if __name__ == "__main__":
    unittest.main()
