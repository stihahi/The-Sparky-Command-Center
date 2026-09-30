import json
import unittest

import strata

MIB = 1048576


def metrics_body(state="idle", tok_s=None, queued=0, prompt_read=None, prompt_total=None, requests=None):
    return json.dumps({
        "engine": {"model": "qwen3.8-flash-next-coder-iq1_m", "version": "0.1.25"},
        "live": {"state": state, "queued": queued, "tok_s": tok_s,
                 "prompt_read": prompt_read, "prompt_total": prompt_total},
        "requests": requests if requests is not None else [{"prompt_ms": 3203.0, "decode_tok_s": 41.5}],
        "totals": {"requests": 7, "prompt_tokens": 326836, "output_tokens": 3466},
        "hardware": {"gpu_util": 97, "gpu_mem_used": 12000 * MIB, "gpu_mem_total": 12227 * MIB,
                     "gpu_temp": 56, "gpu_power": 134.915, "gpu_power_limit": 250.0,
                     "cpu": 19.9, "ram_used": 31282 * MIB, "ram_total": 31913 * MIB},
        "hardware_static": {"gpu_name": "NVIDIA GeForce RTX 5070", "gpu_count": 1},
    })


class ParseMetricsTest(unittest.TestCase):
    def test_strata_json_is_recognised(self):
        self.assertEqual(strata.parse_metrics(metrics_body())["engine"]["version"], "0.1.25")

    def test_prometheus_text_is_not_strata(self):
        self.assertIsNone(strata.parse_metrics("vllm:num_requests_running 1\n"))

    def test_json_without_strata_shape_is_not_strata(self):
        self.assertIsNone(strata.parse_metrics('{"status": "ok"}'))


class ModelReadingTest(unittest.TestCase):
    def reading(self, previous=None, now=100.0, **kw):
        return strata.model_reading(strata.parse_metrics(metrics_body(**kw)), previous, now)

    def test_idle_server_reports_zero_speeds_and_nothing_running(self):
        fields, _ = self.reading()
        self.assertEqual(fields["engine"], "Strata")
        self.assertEqual(fields["model"], "qwen3.8-flash-next-coder-iq1_m")
        self.assertEqual((fields["decode_tps"], fields["prefill_tps"]), (0.0, 0.0))
        self.assertEqual((fields["running"], fields["waiting"]), (0, 0))

    def test_generating_uses_live_decode_speed(self):
        fields, _ = self.reading(state="generating", tok_s=38.2, queued=2)
        self.assertEqual(fields["decode_tps"], 38.2)
        self.assertEqual((fields["running"], fields["waiting"]), (1, 2))

    def test_ttft_comes_from_the_latest_request_prompt_time(self):
        fields, _ = self.reading()
        self.assertEqual(fields["ttft_ms"], 3203.0)

    def test_ttft_is_unknown_before_any_request(self):
        fields, _ = self.reading(requests=[])
        self.assertIsNone(fields["ttft_ms"])

    def test_prefill_speed_is_prompt_progress_between_polls_of_one_prompt(self):
        _, progress = self.reading(state="reading", prompt_read=1000, prompt_total=35000, now=100.0)
        fields, _ = self.reading(progress, state="reading", prompt_read=5000, prompt_total=35000, now=102.0)
        self.assertEqual(fields["prefill_tps"], 2000.0)

    def test_prefill_speed_ignores_progress_of_a_different_prompt(self):
        _, progress = self.reading(state="reading", prompt_read=1000, prompt_total=35000, now=100.0)
        fields, _ = self.reading(progress, state="reading", prompt_read=5000, prompt_total=8000, now=102.0)
        self.assertEqual(fields["prefill_tps"], 0.0)


class NodeReadingTest(unittest.TestCase):
    def setUp(self):
        self.node = strata.node_reading(strata.parse_metrics(metrics_body()))

    def test_single_gpu_is_reported_in_mib_like_nvidia_smi(self):
        gpu = self.node["gpus"][0]
        self.assertEqual(gpu["name"], "RTX 5070")
        self.assertEqual((gpu["temp"], gpu["power"], gpu["power_limit"], gpu["util"]), (56, 134.9, 250.0, 97))
        self.assertEqual((gpu["mem_used_mb"], gpu["mem_total_mb"], gpu["mem_pct"]), (12000, 12227, 98.1))

    def test_system_ram_and_served_model(self):
        self.assertEqual((self.node["mem_used_mb"], self.node["mem_total_mb"], self.node["mem_pct"]),
                         (31282, 31913, 98.0))
        self.assertEqual(self.node["models"], [{"name": "qwen3.8-flash-next-coder-iq1_m",
                                                "label": "qwen3.8-flash-next-coder-iq1_m", "up": True}])


class TokenCountsTest(unittest.TestCase):
    def test_cumulative_prompt_and_output_tokens(self):
        self.assertEqual(strata.token_counts(strata.parse_metrics(metrics_body())), (326836.0, 3466.0))


if __name__ == "__main__":
    unittest.main()
