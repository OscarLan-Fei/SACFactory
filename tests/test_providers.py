import unittest
import json
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

from sacfactory.language import ApiConfig, LanguageError
from sacfactory.providers import MIT_PARLEY_BASE, environment_config, list_available_models, probe_chat_connection, setup_candidate


class ProviderTests(unittest.TestCase):
    def test_catalog_and_inference_use_the_same_trimmed_bearer_key(self):
        config = ApiConfig(MIT_PARLEY_BASE, ' model ', ' \ttest-key\t ', 'responses')
        catalog = Mock(return_value={'data': [{'id': 'model'}]})
        probe = Mock(return_value={'status': 'completed', 'output_text': 'OK'})
        list_available_models(config, catalog)
        result = probe_chat_connection(config, probe)
        self.assertTrue(result['complete_text_reply'])
        self.assertEqual(catalog.call_args.args[0].get_header('Authorization'), 'Bearer test-key')
        self.assertEqual(probe.call_args.args[0].get_header('Authorization'), 'Bearer test-key')
        request, timeout = probe.call_args.args
        body = json.loads(request.data)
        self.assertEqual(request.full_url, MIT_PARLEY_BASE + '/responses')
        self.assertEqual(body, {'model': 'model', 'input': 'Reply with OK.', 'store': False, 'max_output_tokens': 256})
        self.assertEqual(timeout, 30)
        self.assertNotIn('test-key', request.data.decode())

    def test_line_breaks_still_rejected_after_space_normalization(self):
        for key in ('test\nkey', '\ntest-key', 'test-key\r\n'):
            with self.assertRaises(LanguageError):
                ApiConfig(MIT_PARLEY_BASE, 'model', key).endpoint()

    def test_probe_http_error_contains_status_and_route_without_secrets(self):
        transport = Mock(side_effect=HTTPError('PRIVATE_URL', 401, 'PRIVATE_KEY', {}, None))
        with self.assertRaises(LanguageError) as caught:
            probe_chat_connection(ApiConfig(MIT_PARLEY_BASE, 'model', 'PRIVATE_KEY', 'responses'), transport)
        self.assertIn('HTTP 401, /responses', str(caught.exception))
        self.assertNotIn('PRIVATE_', str(caught.exception))
        transport.assert_called_once()

    def test_probe_incomplete_reply_does_not_claim_complete_chat(self):
        config = ApiConfig(MIT_PARLEY_BASE, 'model', 'key', 'responses')
        for value in ({'status': 'incomplete', 'output': []}, {'status': 'completed', 'output': None},
                      {'status': 'completed', 'output': [{'type': 'message', 'content': None}]}):
            result = probe_chat_connection(config, Mock(return_value=value))
            self.assertTrue(result['request_accepted'])
            self.assertFalse(result['complete_text_reply'])

    def test_parley_key_and_responses_endpoint_supported(self):
        config = ApiConfig(MIT_PARLEY_BASE, 'institution-model', 'sk-parley-v1-test-only', 'responses')
        self.assertEqual(config.research_endpoint(), MIT_PARLEY_BASE + '/responses')
        self.assertNotIn(config.api_key, repr(config))

    def test_model_discovery_requires_key_but_not_model(self):
        config = ApiConfig(MIT_PARLEY_BASE, '', 'test-key', 'responses')
        transport = Mock(return_value={'data': [{'id': 'z-model'}, {'id': 'a-model'}, {'id': 'z-model'}, {},
                                              {'id': 'test-key'}, {'id': 'bad\nmodel'}]})
        self.assertEqual(list_available_models(config, transport), ['a-model', 'z-model'])
        request, timeout = transport.call_args.args
        self.assertEqual(request.full_url, MIT_PARLEY_BASE + '/models')
        self.assertEqual(request.get_method(), 'GET')
        self.assertIsNone(request.data)
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-key')
        self.assertEqual(timeout, 30)

    def test_missing_fields_have_specific_errors_and_no_requests(self):
        transport = Mock()
        with self.assertRaisesRegex(LanguageError, 'API key is missing'):
            list_available_models(ApiConfig(MIT_PARLEY_BASE, '', ''), transport)
        transport.assert_not_called()
        with self.assertRaisesRegex(LanguageError, 'Model ID is missing'):
            ApiConfig(MIT_PARLEY_BASE, '', 'test-key').endpoint()

    def test_model_errors_are_sanitized_and_never_retried(self):
        for error in (HTTPError('SECRET_URL', 401, 'PRIVATE_KEY', {}, None),
                      HTTPError('SECRET_URL', 404, 'PRIVATE_KEY', {}, None),
                      HTTPError('SECRET_URL', 429, 'PRIVATE_KEY', {}, None),
                      URLError('PRIVATE_KEY'), TimeoutError('PRIVATE_KEY'), ValueError('PRIVATE_KEY')):
            transport = Mock(side_effect=error)
            with self.assertRaises(LanguageError) as caught:
                list_available_models(ApiConfig(MIT_PARLEY_BASE, '', 'PRIVATE_KEY'), transport)
            self.assertNotIn('PRIVATE_KEY', str(caught.exception))
            self.assertNotIn('SECRET_URL', str(caught.exception))
            transport.assert_called_once()

    def test_invalid_or_empty_catalog_is_not_a_model_guess(self):
        for result in (None, [], {'data': []}, {'data': {}}, {'data': [{'id': 1}]}):
            with self.assertRaises(LanguageError):
                list_available_models(ApiConfig(MIT_PARLEY_BASE, '', 'test-key'), Mock(return_value=result))

    def test_openai_environment_aliases_and_coherent_precedence(self):
        config = environment_config({'OPENAI_BASE_URL': MIT_PARLEY_BASE, 'OPENAI_API_KEY': 'school-key', 'OPENAI_MODEL_ID': 'school-model'})
        self.assertEqual((config.base_url, config.model, config.api_key, config.api_format),
                         (MIT_PARLEY_BASE, 'school-model', 'school-key', 'responses'))
        config = environment_config({'SACFACTORY_API_BASE_URL': 'https://other.test/v1',
                                     'OPENAI_BASE_URL': MIT_PARLEY_BASE, 'OPENAI_API_KEY': 'school-key'})
        self.assertEqual(config.api_key, '')
        self.assertEqual(environment_config({}).api_format, 'chat_completions')
        self.assertEqual(environment_config({'OPENAI_API_KEY': 'test'}).base_url, 'https://api.openai.com/v1')

    def test_pending_key_reused_for_model_correction_only_same_base(self):
        pending = ApiConfig(MIT_PARLEY_BASE, '', 'school-key', 'responses')
        config = setup_candidate(MIT_PARLEY_BASE + '/', 'selected-model', '', 'responses', pending=pending, environ={})
        self.assertEqual(config.api_key, 'school-key')
        config.endpoint()
        changed = setup_candidate('https://other.test', 'model', '', 'responses', pending=pending,
                                  environ={'OPENAI_BASE_URL': MIT_PARLEY_BASE, 'OPENAI_API_KEY': 'school-key'})
        self.assertEqual(changed.api_key, '')
        with self.assertRaises(LanguageError):
            changed.endpoint()

    def test_matching_environment_key_only_and_explicit_key_priority(self):
        env = {'OPENAI_BASE_URL': MIT_PARLEY_BASE, 'OPENAI_API_KEY': 'school-key'}
        self.assertEqual(setup_candidate(MIT_PARLEY_BASE, 'model', '', 'responses', environ=env).api_key, 'school-key')
        self.assertEqual(setup_candidate(MIT_PARLEY_BASE, 'model', 'new-key', 'responses', environ=env).api_key, 'new-key')
