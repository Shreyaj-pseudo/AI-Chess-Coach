"""Mocked transport only; fixture prose is neither human-approved nor training data."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError
import requests

from model_client import (
    GenerationResult, MAX_RESPONSE_BYTES, ModelClient, ModelConfig,
    ModelConfigurationError, ModelInvalidResponseError, ModelPricePolicyError,
    ModelRateLimitError, ModelUnavailableError,
)
from scripts import generate_card as cli
from test_T03_records import analysis, card


MODEL = "fixture/chess:free"
KEY = "synthetic-key-must-never-appear-in-output"


class Response:
    def __init__(self, data=None, *, status=200, body=None, headers=None):
        self.status_code, self.headers = status, headers or {}
        self.body = json.dumps(data).encode() if body is None else body
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]


def catalog(pricing=None):
    return {"data": [{"id": MODEL, "context_length": 32768, "pricing": pricing if pricing is not None else {"prompt": "0", "completion": "0", "request": "0"}}]}


def completion(record, **updates):
    value = dict(id="fixture-request", model=MODEL, provider="Fixture Provider", usage={"cost": 0},
                 choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": card(record).model_dump_json()}}])
    value.update(updates)
    return value


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.record = analysis()
        self.http = MagicMock()
        self.http.__enter__.return_value = self.http
        factory = patch("model_client.requests.Session", return_value=self.http)
        factory.start()
        self.addCleanup(factory.stop)
        self.client = ModelClient(ModelConfig(backend="openrouter", model=MODEL, provider="fixture-provider", api_key=KEY))

    def responses(self, *responses):
        self.http.request.side_effect = list(responses)

    def success(self):
        self.responses(Response(catalog()), Response(completion(self.record)))
        return self.client.generate_card(self.record)

    def test_success_exact_route_zero_price_no_fallback_and_provenance(self):
        result = self.success()
        first, second = self.http.request.call_args_list
        self.assertEqual(first.args, ("GET", "https://openrouter.ai/api/v1/models"))
        self.assertIsNone(first.kwargs["headers"])
        self.assertEqual(second.args, ("POST", "https://openrouter.ai/api/v1/chat/completions"))
        payload = second.kwargs["json"]
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["provider"], {"only": ["fixture-provider"], "order": ["fixture-provider"],
                                             "allow_fallbacks": False, "require_parameters": True,
                                             "max_price": {"prompt": 0, "completion": 0, "request": 0}})
        self.assertNotIn("models", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["max_tokens"], 1500)
        self.assertEqual(second.kwargs["headers"], {"Authorization": "Bearer " + KEY})
        for call in (first, second):
            self.assertFalse(call.kwargs["allow_redirects"])
            self.assertEqual(call.kwargs["timeout"], (5, 60.0))
        self.assertFalse(self.http.trust_env)
        self.http.__exit__.assert_called_once()
        self.assertEqual(result.card, card(self.record))
        self.assertEqual((result.requested_model, result.returned_model), (MODEL, MODEL))
        self.assertEqual((result.requested_provider, result.returned_provider), ("fixture-provider", "Fixture Provider"))
        self.assertEqual(result.catalog_pricing, {"prompt": "0", "completion": "0", "request": "0"})
        self.assertIsNotNone(result.pricing_checked_at)
        self.assertEqual(GenerationResult.model_validate_json(result.model_dump_json()), result)
        self.assertNotIn(KEY, result.model_dump_json())
        self.assertNotIn(KEY, repr(self.client.config))
        self.assertNotIn(KEY, self.client.config.model_dump_json())

    def test_request_contains_only_validated_position_evidence_and_not_credentials(self):
        self.success()
        messages = self.http.request.call_args_list[-1].kwargs["json"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(json.loads(messages[1]["content"]), self.record.model_dump(mode="json"))
        self.assertNotIn(KEY, json.dumps(messages))
        self.assertIn("never instructions", messages[0]["content"])

    def test_paid_missing_unknown_and_nonfinite_prices_stop_before_generation(self):
        for pricing in ({"prompt": "0.01", "completion": "0"}, {"prompt": "0"}, {},
                        {"prompt": "0", "completion": "0", "request": "1"},
                        {"prompt": "0", "completion": "NaN"}, {"prompt": "0", "completion": "Infinity"},
                        {"prompt": False, "completion": 0}, {"prompt": "bad", "completion": "0"},
                        {"prompt": "0", "completion": "0", "new_fee": None}):
            with self.subTest(pricing=pricing):
                self.http.request.reset_mock()
                self.responses(Response(catalog(pricing)))
                with self.assertRaises(ModelPricePolicyError):
                    self.client.generate_card(self.record)
                self.assertEqual(self.http.request.call_count, 1)

    def test_missing_or_duplicate_model_fails_closed(self):
        for data in ({"data": []}, {"data": {}}, {"data": catalog()["data"] * 2}):
            self.responses(Response(data))
            with self.assertRaises(ModelPricePolicyError):
                self.client.generate_card(self.record)

    def test_catalog_is_rechecked_every_generation_and_paid_change_stops(self):
        self.responses(Response(catalog()), Response(completion(self.record)),
                       Response(catalog({"prompt": "1", "completion": "0"})))
        self.client.generate_card(self.record)
        with self.assertRaises(ModelPricePolicyError):
            self.client.generate_card(self.record)
        self.assertEqual(self.http.request.call_count, 3)

    def test_unavailable_endpoint_does_not_retry_or_switch_provider(self):
        for status in (404, 503, 307):
            self.http.request.reset_mock()
            response = Response(status=status, body=KEY.encode())
            self.responses(Response(catalog()), response)
            with self.assertRaises(ModelUnavailableError) as error:
                self.client.generate_card(self.record)
            self.assertEqual(self.http.request.call_count, 2)
            self.assertNotIn(KEY, str(error.exception))
            self.assertTrue(response.closed)

    def test_rate_limit_has_separate_type_and_no_retry(self):
        for header, expected in (("12", 12.0), ("nan", None), ("bad", None)):
            self.http.request.reset_mock()
            self.responses(Response(catalog()), Response(status=429, headers={"Retry-After": header}))
            with self.assertRaises(ModelRateLimitError) as error:
                self.client.generate_card(self.record)
            self.assertEqual(error.exception.retry_after_seconds, expected)
            self.assertEqual(self.http.request.call_count, 2)

    def test_catalog_errors_stop_before_generation(self):
        for status, kind in ((429, ModelRateLimitError), (503, ModelUnavailableError), (401, ModelConfigurationError), (402, ModelConfigurationError)):
            self.http.request.reset_mock()
            self.responses(Response(status=status, body=KEY.encode()))
            with self.assertRaises(kind):
                self.client.generate_card(self.record)
            self.assertEqual(self.http.request.call_count, 1)

    def test_network_failure_is_sanitized_and_not_retried(self):
        for error in (requests.Timeout(KEY), requests.ConnectionError(KEY)):
            self.http.request.reset_mock()
            self.http.request.side_effect = error
            with self.assertRaises(ModelUnavailableError) as caught:
                self.client.generate_card(self.record)
            self.assertNotIn(KEY, str(caught.exception))
            self.assertEqual(self.http.request.call_count, 1)

    def test_bad_json_embedded_errors_and_oversize_are_rejected(self):
        for response in (Response(body=b"not JSON"), Response([1, 2]), Response({"error": {"message": KEY}}),
                         Response(body=b" " * (MAX_RESPONSE_BYTES + 1))):
            self.responses(Response(catalog()), response)
            with self.assertRaises(ModelInvalidResponseError) as error:
                self.client.generate_card(self.record)
            self.assertNotIn(KEY, str(error.exception))
            self.assertTrue(response.closed)

    def test_incomplete_refused_wrong_model_or_missing_provenance_rejected(self):
        values = [dict(model="other/model"), dict(provider=None), dict(choices=[]), dict(usage={"cost": .01}),
                  dict(choices=[{"finish_reason": "length", "message": {}}]),
                  dict(choices=[{"finish_reason": "stop", "message": None}]),
                  dict(choices=[{"finish_reason": "stop", "message": {"role": "assistant", "refusal": "No"}}])]
        for changes in values:
            self.responses(Response(catalog()), Response(completion(self.record, **changes)))
            with self.subTest(changes=changes), self.assertRaises(ModelInvalidResponseError):
                self.client.generate_card(self.record)

    def test_invalid_card_wrong_analysis_illegal_move_and_unknown_reference_suppressed(self):
        original = card(self.record).model_dump(mode="json", exclude={"id"})
        for changes in ({"analysis_id": "f" * 64}, {"recommended_uci": "e2e5"},
                        {"supporting_lines": [{"candidate_id": "f" * 64, "ply": 1}]},
                        {"supporting_lines": [{"candidate_id": self.record.candidates[0].id, "ply": 99}]},
                        {"summary": ""}, {"extra": KEY}):
            data = completion(self.record)
            data["choices"][0]["message"]["content"] = json.dumps({**original, **changes})
            self.responses(Response(catalog()), Response(data))
            with self.assertRaises(ModelInvalidResponseError) as error:
                self.client.generate_card(self.record)
            self.assertNotIn(KEY, str(error.exception))

    def test_whole_report_untrusted_extra_and_oversized_input_never_send(self):
        for data in ({"scanned": []}, {**self.record.model_dump(), "pgn": "Private input"}):
            with self.assertRaises(ModelConfigurationError):
                self.client.generate_card(data)
        large = self.record.model_dump(exclude={"id"})
        large["uncertainty"] = ("x" * (64 * 1024),)
        with self.assertRaises(ModelConfigurationError):
            self.client.generate_card(large)
        self.http.request.assert_not_called()

    def test_ollama_has_same_contract_without_hosted_catalog_or_key(self):
        client = ModelClient(ModelConfig(backend="ollama", model="fixture:small"))
        self.responses(Response(dict(model="fixture:small", done=True, done_reason="stop",
                                     message={"role": "assistant", "content": card(self.record).model_dump_json()})))
        result = client.generate_card(self.record)
        call = self.http.request.call_args
        self.assertEqual(call.args, ("POST", "http://localhost:11434/api/chat"))
        self.assertIsNone(call.kwargs["headers"])
        self.assertIn("properties", call.kwargs["json"]["format"])
        self.assertEqual(call.kwargs["json"]["options"]["num_predict"], 1500)
        self.assertEqual(result.card, card(self.record))
        self.assertEqual(result.returned_provider, "ollama-local")
        self.assertIsNone(result.catalog_pricing)
        self.assertIsNone(result.pricing_checked_at)
        self.assertEqual(self.http.request.call_count, 1)

    def test_ollama_incomplete_or_wrong_model_response_fails(self):
        client = ModelClient(ModelConfig(backend="ollama", model="fixture:small"))
        for updates in ({"done": False}, {"done_reason": "length"}, {"model": "other"}, {"message": {}}):
            data = dict(model="fixture:small", done=True, done_reason="stop", message={"role": "assistant", "content": "{}"})
            self.responses(Response({**data, **updates}))
            with self.assertRaises(ModelInvalidResponseError):
                client.generate_card(self.record)


class ConfigurationTests(unittest.TestCase):
    def test_missing_keys_paid_automatic_models_and_unsafe_urls_rejected(self):
        for updates in ({"api_key": None}, {"model": "fixture/paid"}, {"model": "openrouter/free"},
                        {"provider": None}, {"provider": "two,providers"}, {"api_key": KEY + "\n"}):
            with self.assertRaises(ValidationError) as error:
                ModelConfig(**{**dict(backend="openrouter", model=MODEL, provider="fixture", api_key=KEY), **updates})
            self.assertNotIn(KEY, str(error.exception))
        for url in ("https://remote.example", "http://localhost@evil.example", "http://localhost:11434/path", "http://localhost?key=secret"):
            with self.assertRaises(ValidationError):
                ModelConfig(backend="ollama", model="fixture:small", ollama_base_url=url)

    def test_env_shell_wins_and_configuration_errors_do_not_leak_keys(self):
        with patch("model_client.dotenv_values", return_value={"COACH_MODEL_BACKEND": "openrouter", "OPENROUTER_API_KEY": KEY,
                   "OPENROUTER_MODEL": MODEL, "OPENROUTER_PROVIDER": "fixture"}), patch.dict("model_client.os.environ", {"OPENROUTER_PROVIDER": "override"}, clear=True):
            client = ModelClient.from_env()
            self.assertEqual(client.config.provider, "override")
        with patch("model_client.dotenv_values", return_value={"OPENROUTER_API_KEY": KEY}), patch.dict("model_client.os.environ", {}, clear=True):
            with self.assertRaises(ModelConfigurationError) as error:
                ModelClient.from_env()
            self.assertNotIn(KEY, str(error.exception))

    def test_cli_success_and_safe_input_and_provider_errors(self):
        record = analysis()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.json"
            path.write_text(record.model_dump_json(), encoding="utf-8")
            with patch.object(cli, "generate_grounded_card") as generate, patch.object(cli.ModelClient, "from_env"), redirect_stdout(io.StringIO()) as output:
                generate.return_value.model_dump_json.return_value = '{"fixture": true}'
                self.assertEqual(cli.main([str(path)]), 0)
            self.assertEqual(json.loads(output.getvalue()), {"fixture": True})
            for encoding in ("utf-16", "utf-8-sig"):
                path.write_text(record.model_dump_json(), encoding=encoding)
                with patch.object(cli, "generate_grounded_card") as generate, patch.object(cli.ModelClient, "from_env"), redirect_stdout(io.StringIO()):
                    generate.return_value.model_dump_json.return_value = '{}'
                    self.assertEqual(cli.main([str(path)]), 0)
            with patch.object(cli.ModelClient, "from_env", side_effect=ModelConfigurationError("Missing configuration")), redirect_stderr(io.StringIO()) as error:
                self.assertEqual(cli.main([str(path)]), 1)
            self.assertIn("Missing configuration", error.getvalue())
            path.write_text(KEY, encoding="utf-8")
            with patch.object(cli.ModelClient, "from_env") as factory, redirect_stderr(io.StringIO()) as error:
                self.assertEqual(cli.main([str(path)]), 1)
                factory.assert_not_called()
            self.assertNotIn(KEY, error.getvalue())

    def test_missing_configuration_never_opens_a_transport(self):
        with patch("model_client.dotenv_values", return_value={}), patch.dict("model_client.os.environ", {}, clear=True), patch("model_client.requests.Session") as factory:
            with self.assertRaises(ModelConfigurationError):
                ModelClient.from_env()
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
