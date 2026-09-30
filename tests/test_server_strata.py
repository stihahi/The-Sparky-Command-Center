import os
import unittest
from unittest import mock

os.environ["CC_CONFIG"] = os.path.join(os.path.dirname(__file__), "no-such-config.json")
import server  # noqa: E402

from tests.test_strata import metrics_body  # noqa: E402

STRATA_URL = "https://strata.example.ts.net"


def answering(body):
    return mock.patch.object(server, "_http_get", return_value=(True, body))


class PollModelTest(unittest.TestCase):
    def test_strata_metrics_become_a_reachable_model_card(self):
        with answering(metrics_body(state="generating", tok_s=40.0)):
            card = server.poll_model({"key": "strata-test", "label": "Strata", "endpoint": STRATA_URL})
        self.assertTrue(card["reachable"])
        self.assertEqual((card["engine"], card["decode_tps"]), ("Strata", 40.0))


class PollNodeTest(unittest.TestCase):
    node = {"key": "pc", "name": "Gaming PC", "profile": "discrete", "source": "strata", "url": STRATA_URL}

    def test_strata_source_reads_the_host_over_http(self):
        with answering(metrics_body()) as http_get:
            card = server.poll_node(self.node)
        http_get.assert_called_once_with(STRATA_URL + "/metrics", timeout=8, headers=None)
        self.assertTrue(card["reachable"])
        self.assertEqual(card["gpus"][0]["name"], "RTX 5070")

    def test_unreadable_strata_host_is_down(self):
        with mock.patch.object(server, "_http_get", return_value=(False, "")):
            card = server.poll_node(self.node)
        self.assertEqual((card["reachable"], card["err"]), (False, "no Strata /metrics"))


class TokenScrapeTest(unittest.TestCase):
    def test_strata_totals_feed_the_token_bank(self):
        with answering(metrics_body()):
            self.assertEqual(server._tok_scrape(STRATA_URL + "/metrics"), (326836.0, 3466.0))


if __name__ == "__main__":
    unittest.main()
