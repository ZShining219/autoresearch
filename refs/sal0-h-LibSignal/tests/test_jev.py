"""Jev's HTTP boundary and simulator action contract; all API calls are mocked."""

import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from agent.jev import JEVAgent
from agent.jev_backend import ENDPOINT, MODEL_ID, JEVClient, JEVError, parse_decision
from agent.jev_prompt import QUESTION_ID, build_jev_request
from agent.llm_tsc import LLMTSCAgent
from agent.llm_tsc_prompt import SIGNAL_ORDER, empty_signal_stats
from common.registry import Registry
from utils.logger import load_config


def response_body(signal="NTST", confidence=0.8):
    return {
        "id": "test-request",
        "model": MODEL_ID + "-20260917",
        "answers": {
            QUESTION_ID: {
                "type": "choice",
                "choice": signal,
                "confidence": confidence,
                "probabilities": {name: 0.85 if name == signal else 0.05 for name in SIGNAL_ORDER},
            }
        },
        "usage": {"input_tokens": 500, "output_tokens": 40, "cost": 0.000021},
    }


def http_response(status=200, body=None, headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = response_body() if body is None else body
    return result


class PromptTest(unittest.TestCase):
    def test_preserves_approaches_and_computes_queue_and_segment_totals(self):
        stats = empty_signal_stats()
        stats["ETWT"]["East"] = {"early_queued": 7, "segments": [2, 3, 4]}
        stats["ETWT"]["West"] = {"early_queued": 5, "segments": [6, 7, 8]}
        original = copy.deepcopy(stats)
        request = build_jev_request(stats)
        self.assertEqual(stats, original)
        signal = request["state"]["signals"]["ETWT"]
        self.assertEqual(signal["approaches"], stats["ETWT"])
        self.assertEqual(signal["totals"], {"early_queued": 12, "segments": [8, 10, 12]})
        self.assertEqual(set(request), {"state", "questions"})
        self.assertEqual(set(request["questions"][QUESTION_ID]["criteria"]), set(SIGNAL_ORDER))
        text = json.dumps(request)
        for forbidden in ("<think>", "boxed", "temperature", "max_tokens", '"role"'):
            self.assertNotIn(forbidden, text)

    def test_empty_intersection_still_has_all_four_choices(self):
        request = build_jev_request(empty_signal_stats())
        self.assertEqual(request["questions"][QUESTION_ID]["type"], "choice")
        self.assertEqual(len(request["questions"][QUESTION_ID]["criteria"]), 4)

    def test_segment_description_and_totals_match_configured_length(self):
        request = build_jev_request(empty_signal_stats(2), 2)
        self.assertIn("2 equal", request["state"]["observation_fields"]["segments"])
        with self.assertRaises(ValueError):
            build_jev_request(empty_signal_stats(), 2)
        with self.assertRaises(ValueError):
            build_jev_request(empty_signal_stats(), 0)


class DecisionTest(unittest.TestCase):
    def test_provider_rounding_is_preserved_without_rejecting_a_valid_choice(self):
        data = response_body()
        rounded = {"NTST": 0.80, "ETWT": 0.10, "ELWL": 0.06, "NLSL": 0.05}
        data["answers"][QUESTION_ID]["probabilities"] = rounded
        decision = parse_decision(data, MODEL_ID, 0.1, 1)
        self.assertEqual(decision.probabilities, rounded)
        self.assertEqual(decision.signal, "NTST")

    def test_preserves_native_decision_metadata(self):
        decision = parse_decision(response_body(), MODEL_ID, 0.12, 1)
        self.assertEqual(decision.signal, "NTST")
        self.assertEqual(decision.confidence, 0.8)
        self.assertEqual(decision.model, MODEL_ID + "-20260917")
        self.assertEqual(decision.usage["cost"], 0.000021)

    def test_optional_confidence_and_probabilities_are_not_invented(self):
        data = response_body()
        del data["answers"][QUESTION_ID]["confidence"]
        del data["answers"][QUESTION_ID]["probabilities"]
        del data["usage"]["cost"]
        decision = parse_decision(data, MODEL_ID, 0.1, 1)
        self.assertIsNone(decision.confidence)
        self.assertIsNone(decision.probabilities)
        self.assertNotIn("cost", decision.usage)

    def test_invalid_answers_cannot_become_actions(self):
        changes = [
            lambda d: d.pop("answers"),
            lambda d: d.update(model="typesafe/jev-latest"),
            lambda d: d["answers"].clear(),
            lambda d: d["answers"][QUESTION_ID].update(type="score"),
            lambda d: d["answers"][QUESTION_ID].update(choice="invented"),
            lambda d: d["answers"][QUESTION_ID].update(choice=["NTST"]),
            lambda d: d["answers"][QUESTION_ID].update(confidence=float("nan")),
            lambda d: d["answers"][QUESTION_ID].update(confidence=True),
            lambda d: d["answers"][QUESTION_ID].update(confidence=1.1),
            lambda d: d["answers"][QUESTION_ID]["probabilities"].pop("ETWT"),
            lambda d: d["answers"][QUESTION_ID]["probabilities"].update(NTST=0.1),
            lambda d: d["answers"][QUESTION_ID]["probabilities"].update(NTST=float("inf")),
            lambda d: d["usage"].update(cost=-1),
            lambda d: d.pop("usage"),
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                data = response_body()
                change(data)
                with self.assertRaises(JEVError):
                    parse_decision(data, MODEL_ID, 0.1, 1)


class ClientTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        session = patch("requests.Session")
        self.session = session.start().return_value
        self.addCleanup(session.stop)
        sleep = patch("agent.jev_backend.time.sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        self.request = build_jev_request(empty_signal_stats())

    def test_calls_only_openrouter_decisions_with_the_requested_model(self):
        self.session.post.return_value = http_response()
        decision = JEVClient({}).decide(self.request)
        args, kwargs = self.session.post.call_args
        self.assertEqual(args, (ENDPOINT,))
        self.assertEqual(set(kwargs["json"]), {"model", "state", "questions"})
        self.assertEqual(kwargs["json"]["model"], MODEL_ID)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertFalse(kwargs["allow_redirects"])
        self.assertGreater(kwargs["timeout"], 0)
        self.assertEqual(decision.attempts, 1)

    def test_rate_limit_retries_same_request_and_honors_retry_after(self):
        self.session.post.side_effect = [
            http_response(429, {"error": {"message": "rate limited"}}, {"Retry-After": "2"}),
            http_response(),
        ]
        result = JEVClient({}).decide(self.request)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(self.session.post.call_args_list[0], self.session.post.call_args_list[1])
        self.sleep.assert_called_once_with(2.0)

    def test_timeouts_are_bounded_and_do_not_leak_exception_details(self):
        self.session.post.side_effect = requests.Timeout("sensitive test-key")
        with self.assertRaises(JEVError) as caught:
            JEVClient({"request_attempts": 2}).decide(self.request)
        self.assertEqual(self.session.post.call_count, 2)
        self.assertNotIn("test-key", str(caught.exception))

    def test_auth_credit_validation_and_redirect_errors_are_not_retried(self):
        for status in (400, 401, 402, 403, 404, 413, 302):
            with self.subTest(status=status):
                self.session.post.reset_mock()
                self.session.post.return_value = http_response(status, {"error": {"message": "test-key"}})
                with self.assertRaises(JEVError) as caught:
                    JEVClient({}).decide(self.request)
                self.assertNotIn("test-key", str(caught.exception))
                self.assertEqual(self.session.post.call_count, 1)

    def test_malformed_success_is_not_retried_as_a_chat_completion(self):
        self.session.post.return_value = http_response(body={"choices": []})
        with self.assertRaises(JEVError):
            JEVClient({}).decide(self.request)
        self.assertEqual(self.session.post.call_count, 1)

    def test_invalid_json_is_a_clear_failure(self):
        response = http_response()
        response.json.side_effect = ValueError("not JSON")
        self.session.post.return_value = response
        with self.assertRaisesRegex(JEVError, "invalid JSON"):
            JEVClient({}).decide(self.request)

    def test_long_retry_delay_fails_without_sleeping(self):
        self.session.post.return_value = http_response(429, {}, {"Retry-After": "120"})
        with self.assertRaisesRegex(JEVError, "over 30 seconds"):
            JEVClient({}).decide(self.request)
        self.sleep.assert_not_called()

    def test_dotenv_loading_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text('OPENROUTER_API_KEY="file-key"\nUNRELATED=value\n')
            with patch("agent.jev_backend.ENV_FILE", env_path):
                self.assertEqual(JEVClient({})._api_key, "test-key")
                del os.environ["OPENROUTER_API_KEY"]
                self.assertEqual(JEVClient({})._api_key, "file-key")
                self.assertNotIn("UNRELATED", os.environ)
                env_path.write_text("UNRELATED=value\n")
                with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                    JEVClient({})


def stub_agent(intersection, signal="NTST", enabled=True):
    agent = object.__new__(JEVAgent)
    agent.inter_obj = SimpleNamespace(id=intersection, phases=list(range(6)))
    agent.world = SimpleNamespace(get_current_time=lambda: 20)
    agent.param = {}
    agent.n_segments = 3
    agent.parallelism = 2
    agent._llm_enabled = enabled
    agent._signal_to_phase = {"NTST": 0, "NLSL": 1, "ETWT": 4, "ELWL": 5}
    agent.last_action = 1
    agent.last_signal = "NLSL"
    agent.last_decision = None
    agent._decision_log = None
    decision = parse_decision(response_body(signal), MODEL_ID, 0.1, 1)
    agent._jev_client = Mock()
    agent._jev_client.decide.return_value = decision
    agent._collect_signal_stats = Mock(return_value=empty_signal_stats())
    return agent


class AgentTest(unittest.TestCase):
    def test_concurrent_results_keep_agent_order_and_sumo_reads_on_main_thread(self):
        agents = [stub_agent("A", "ETWT"), stub_agent("B", "NLSL")]
        main_thread = threading.get_ident()
        second_finished = threading.Event()

        def observe():
            self.assertEqual(threading.get_ident(), main_thread)
            return empty_signal_stats()

        def first(request):
            self.assertNotEqual(threading.get_ident(), main_thread)
            self.assertTrue(second_finished.wait(timeout=5))
            return parse_decision(response_body("ETWT"), MODEL_ID, 0.1, 1)

        def second(request):
            second_finished.set()
            return parse_decision(response_body("NLSL"), MODEL_ID, 0.1, 1)

        for agent in agents:
            agent._collect_signal_stats.side_effect = observe
        agents[0]._jev_client.decide.side_effect = first
        agents[1]._jev_client.decide.side_effect = second
        self.assertEqual(JEVAgent.get_actions_batch(agents, [], []), [4, 1])
        self.assertEqual([a.last_signal for a in agents], ["ETWT", "NLSL"])

    def test_single_phase_junction_never_calls_api(self):
        agent = stub_agent("fringe", enabled=False)
        agent._jev_client = None
        with patch("agent.jev.JEVClient") as client:
            self.assertEqual(JEVAgent.get_actions_batch([agent], [], []), [0])
        client.assert_not_called()
        self.assertIsNone(agent.last_signal)
        self.assertEqual(JEVAgent.get_actions_batch([], [], []), [])

    def test_low_confidence_choice_is_applied_and_logged(self):
        agent = stub_agent("A")
        agent._jev_client.decide.return_value = parse_decision(response_body("ELWL", 0), MODEL_ID, 0.1, 1)
        with tempfile.TemporaryDirectory() as directory:
            agent._decision_log = Path(directory) / "logger" / "jev_decisions.jsonl"
            self.assertEqual(agent.get_action(None, None), 5)
            record = json.loads(agent._decision_log.read_text())
        self.assertEqual(record["confidence"], 0)
        self.assertEqual(record["signal"], "ELWL")
        self.assertEqual(record["phase_index"], 5)
        self.assertEqual(record["simulation_time"], 20)
        self.assertEqual(record["intersection_id"], "A")

    def test_api_failure_does_not_apply_a_partial_batch_or_fallback(self):
        agents = [stub_agent("A", "ETWT"), stub_agent("B")]
        agents[1]._jev_client.decide.side_effect = JEVError("unavailable")
        with self.assertRaisesRegex(JEVError, "intersection B"):
            JEVAgent.get_actions_batch(agents, [], [])
        self.assertEqual([a.last_action for a in agents], [1, 1])

    def test_invalid_explicit_phase_maps_fail_before_api_calls(self):
        agent = stub_agent("A")
        for mapping in (
            {"NTST": 0},
            {"NTST": -1, "NLSL": 1, "ETWT": 4, "ELWL": 5},
            {"NTST": 0, "NLSL": 1, "ETWT": 4, "ELWL": 6},
            {"NTST": 0, "NLSL": 1, "ETWT": 4, "ELWL": 4},
        ):
            with self.subTest(mapping=mapping):
                with self.assertRaises(ValueError):
                    agent._resolve_phase_map({"phase_name_to_index": mapping})

    def test_reset_discards_previous_answer_and_retains_client(self):
        agent = stub_agent("A")
        client = agent._jev_client
        agent.last_decision = client.decide.return_value
        with patch.object(LLMTSCAgent, "reset"):
            agent.reset()
        self.assertIsNone(agent.last_decision)
        self.assertIs(agent._jev_client, client)

    def test_registered_config_matches_traffic_r1_evaluation_protocol(self):
        jev, _ = load_config("configs/tsc/jev.yml")
        traffic, _ = load_config("configs/tsc/traffic_r1.yml")
        self.assertIs(Registry.mapping["model_mapping"]["jev"], JEVAgent)
        for section in ("trainer", "world", "logger"):
            self.assertEqual(jev[section], traffic[section])
        for name in ("v_stop", "n_segments", "obs_distance", "train_model", "test_model"):
            self.assertEqual(jev["model"][name], traffic["model"][name])
        self.assertEqual(jev["model"]["model_id"], MODEL_ID)
        self.assertNotIn("backend", jev["model"])


if __name__ == "__main__":
    unittest.main()
