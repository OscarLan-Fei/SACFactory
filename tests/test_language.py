import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from unittest.mock import Mock

from sacfactory.dialogue import execute_intent
from sacfactory.language import ApiConfig, CompatibleAdapter, LanguageError, post_json, shared_context, validate_intent
from sacfactory.planning import fit_model
from sacfactory.storage import Store
from test_science import sample, method


def intent(action="recommend", **kwargs):
    return {"intent": action, "sample_id": None, "batch_size": None, "pulse_count_filter": None, "topic": None, **kwargs}


def provider(value, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(value)}}]}


class LanguageTests(unittest.TestCase):
    def setUp(self):
        self.samples, self.methods = [sample()], method()
        self.context = copy.deepcopy(self.samples[0])
        self.space = {"alpha_min": 1., "alpha_max": 1., "pulses": [1], "max_total_mM": .05,
                      "verified": True, "baseline_repeat": True, "range_source": "Confirmed test baseline"}
        self.state = fit_model(self.samples, self.methods, self.context, "REAL")
        self.shared = shared_context(self.samples, self.methods, self.state, self.context, self.space, "REAL")
        self.config = ApiConfig("https://api.example.test/v1", "test-model", "test-secret-key")

    def test_disabled_or_incomplete_config_never_calls_provider(self):
        transport = Mock()
        for adapter in (CompatibleAdapter(self.config, transport=transport),
                        CompatibleAdapter(ApiConfig(self.config.base_url, "", ""), True, transport)):
            with self.assertRaises(LanguageError): adapter.interpret("Recommend", [], self.shared)
        transport.assert_not_called()

    def test_task_parser_cannot_send_to_chat_completions_for_responses_config(self):
        config = ApiConfig('https://parley.api.mit.edu/v1', 'test-model', 'test-key', 'responses')
        transport = Mock()
        with self.assertRaisesRegex(LanguageError, 'Select Research conversation'):
            CompatibleAdapter(config, True, transport).interpret('Discuss data', [], self.shared)
        transport.assert_not_called()

    def test_task_request_trims_copy_pasted_key(self):
        transport = Mock(return_value=provider(intent('summarize')))
        config = ApiConfig(self.config.base_url, 'test-model', ' test-secret-key ')
        CompatibleAdapter(config, True, transport).interpret('Summarize', [], self.shared)
        self.assertEqual(transport.call_args.args[0].get_header('Authorization'), 'Bearer test-secret-key')

    def test_endpoint_and_credential_boundaries(self):
        for base in ("http://remote.test", "https://user:password@api.test", "https://api.test?key=abc", "file:///tmp/x", "https://api.test/chat/completions"):
            with self.assertRaises(LanguageError): ApiConfig(base, "model", "secret").endpoint()
        self.assertNotIn(self.config.api_key, repr(self.config))
        self.assertEqual(ApiConfig("http://127.0.0.1:5000/v1/", "model", "secret").endpoint(), "http://127.0.0.1:5000/v1/chat/completions")

    def test_minimal_context_excludes_raw_evidence_and_other_namespace(self):
        self.samples[0].update(notes="UNTRUSTED_NOTE", import_sources=[{"file": "PRIVATE_PATH"}])
        self.samples.append(sample(1, origin="DEMO"))
        shared = shared_context(self.samples, self.methods, self.state, self.context, self.space, "REAL")
        encoded = json.dumps(shared)
        for excluded in ("UNTRUSTED_NOTE", "PRIVATE_PATH", "xrd.txt", "icp.csv", "S1", "wt%"):
            self.assertNotIn(excluded, encoded)
        self.assertEqual(shared["summary"]["total"], 1)
        self.assertEqual(shared["current_evidence"]["status"], "PASS")

    def test_request_format_and_only_validated_intent_history(self):
        transport = Mock(return_value=provider(intent(batch_size=6, pulse_count_filter=[1])))
        adapter = CompatibleAdapter(self.config, True, transport)
        result = adapter.interpret("Please plan my next experiments", [intent("summarize"), {"content": "PRIVATE_HISTORY"}], self.shared)
        request, timeout = transport.call_args.args
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret-key")
        self.assertEqual(timeout, 30)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertNotIn("PRIVATE_HISTORY", request.data.decode())
        self.assertNotIn(self.config.api_key, request.data.decode())
        answer = execute_intent(result, self.samples, self.methods, self.state, self.context, self.space, "REAL")
        self.assertEqual(len(answer.recommendations), 1)
        self.assertIsNone(answer.recommendations[0]["p_screen_pass"])
        self.assertEqual(answer.recommendations[0]["state"], "OPEN")

    def test_untrusted_or_out_of_range_output_cannot_reach_engine(self):
        for value in (intent("run_code"), intent(batch_size=100), intent(batch_size=True), intent(pulse_count_filter=[8]),
                      intent("diagnose", sample_id="S_FROM_DEMO"), intent("summarize", sample_id="S0"),
                      {**intent(), "temperature_C": 1500}, {**intent(), "sql": "DELETE FROM items"}):
            with self.subTest(value=value), self.assertRaises(LanguageError):
                CompatibleAdapter(self.config, True, Mock(return_value=provider(value))).interpret("Ignored", [], self.shared)

    def test_ambiguity_and_unsupported_edits_do_not_propose_experiments(self):
        for action in ("clarify", "unsupported"):
            answer = execute_intent(intent(action), self.samples, self.methods, self.state, self.context, self.space, "REAL")
            self.assertEqual(answer.recommendations, [])

    def test_secret_in_chat_blocked_before_transport(self):
        transport = Mock()
        with self.assertRaises(LanguageError):
            CompatibleAdapter(self.config, True, transport).interpret(self.config.api_key, [], self.shared)
        transport.assert_not_called()

    def test_invalid_empty_or_truncated_json_no_retry(self):
        for response in (provider(intent(), "length"), {"choices": []}, provider([]),
                         {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
                         {"choices": [{"finish_reason": "stop", "message": {"content": '{"intent":"recommend","intent":"summarize"}'}}]}):
            transport = Mock(return_value=response)
            with self.assertRaises(LanguageError): CompatibleAdapter(self.config, True, transport).interpret("Plan", [], self.shared)
            transport.assert_called_once()

    def test_provider_error_is_redacted_and_never_retried(self):
        transport = Mock(side_effect=HTTPError("SECRET_URL", 401, "test-secret-key", {}, None))
        with self.assertRaises(LanguageError) as caught:
            CompatibleAdapter(self.config, True, transport).interpret("Plan", [], self.shared)
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertNotIn(self.config.api_key, str(caught.exception))
        transport.assert_called_once()

    def test_trace_contains_intent_and_local_result_not_credentials(self):
        parsed = CompatibleAdapter(self.config, True, Mock(return_value=provider(intent("diagnose", sample_id="S0")))).interpret("Inspect S0", [], self.shared)
        answer = execute_intent(parsed, self.samples, self.methods, self.state, self.context, self.space, "REAL")
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder, "REAL")
            store.message("assistant", answer.text, {"language": {"intent": parsed}})
            self.assertNotIn(self.config.api_key, json.dumps(store.export()))
            self.assertEqual(store.all("recommendation"), [])
        self.assertIn("does not independently establish atomic dispersion", answer.text)

    def test_http_transport_does_not_follow_redirect(self):
        hits = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                hits.append(self.path)
                self.send_response(307)
                self.send_header("Location", "/redirected")
                self.end_headers()

            def log_message(self, *args): pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            config = ApiConfig(f"http://127.0.0.1:{server.server_port}", "model", "secret")
            with self.assertRaises(LanguageError): CompatibleAdapter(config, True).interpret("Plan", [], self.shared)
            self.assertEqual(hits, ["/chat/completions"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_loopback_http_request_to_local_calculation(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                encoded = json.dumps(provider(intent("diagnose", sample_id="S0"))).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args): pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            config = ApiConfig(f"http://127.0.0.1:{server.server_port}/v1", "fixture-model", "fixture-secret")
            parsed = CompatibleAdapter(config, True).interpret("Inspect my current experiment", [], self.shared)
            answer = execute_intent(parsed, self.samples, self.methods, self.state, self.context, self.space, "REAL")
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0]["model"], "fixture-model")
            self.assertIn("ICP=PASS", answer.text)
            self.assertEqual(answer.recommendations, [])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__": unittest.main()
