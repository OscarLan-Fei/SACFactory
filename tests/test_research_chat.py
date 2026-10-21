"""Offline research conversation and read-only scientific tool boundary tests."""
import copy
import json
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError

from sacfactory.language import ApiConfig, LanguageError
from sacfactory.planning import fit_model
from sacfactory.research_chat import ResearchAdapter, ResearchTools, research_context
from test_science import sample, method


def response(text='Evidence-based research discussion.'):
    return {'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]}], 'usage': {'input_tokens': 100, 'output_tokens': 20}}


def function_response(name='summarize_experiments', args=None):
    return {'status': 'completed', 'output': [{'type': 'reasoning', 'id': 'REASON1', 'summary': [], 'encrypted_content': 'OPAQUE_REASONING'}, {'type': 'function_call', 'call_id': 'CALL1', 'name': name, 'arguments': json.dumps(args or {})}], 'usage': {'input_tokens': 100, 'output_tokens': 20}}


class ResearchChatTests(unittest.TestCase):
    def setUp(self):
        self.samples, self.methods = [sample()], method()
        self.context = self.samples[0]
        self.space = {'alpha_min': 1., 'alpha_max': 1., 'pulses': [1], 'max_total_mM': .05, 'verified': True, 'baseline_repeat': True, 'range_source': 'Test baseline'}
        self.state = fit_model(self.samples, self.methods, self.context, 'REAL')
        self.config = ApiConfig('https://api.openai.com/v1', 'test-model', 'test-secret', 'responses')
        self.shared = research_context(self.samples, self.methods, self.state, self.context, self.space, 'REAL', ['S0'])
        self.tools = ResearchTools(self.samples, self.methods, self.state, self.context, self.space, 'REAL', ['S0'])

    def test_disabled_research_sends_no_request(self):
        transport = Mock()
        with self.assertRaises(LanguageError):
            ResearchAdapter(self.config, transport=transport).converse('Discuss my data', [], self.shared, self.tools)
        transport.assert_not_called()

    def test_context_shares_only_selected_records_without_notes_files_or_other_namespace(self):
        self.samples[0]['notes'] = 'PRIVATE_NOTE_INJECTION'
        self.samples[0]['icp'][0]['source'] = 'PRIVATE_SOURCE_PATH'
        samples = self.samples + [sample(1), sample(2, origin='DEMO')]
        shared = research_context(samples, self.methods, self.state, self.context, self.space, 'REAL', ['S0'])
        raw = json.dumps(shared)
        self.assertNotIn('PRIVATE_NOTE', raw)
        self.assertNotIn('PRIVATE_SOURCE', raw)
        self.assertEqual([r['sample_id'] for r in shared['shared_records']], ['S0'])
        self.assertEqual(shared['shared_records'][0]['ICP_W_wt_percent']['Fe']['value'], '12')
        with self.assertRaises(LanguageError):
            research_context(samples, self.methods, self.state, self.context, self.space, 'REAL', ['S2'])

    def test_free_conversation_uses_responses_and_bounded_history(self):
        transport = Mock(return_value=response())
        result = ResearchAdapter(self.config, True, transport).converse('Discuss my experiment', [{'role': 'user', 'content': 'Previous question'}, {'role': 'assistant', 'content': 'Previous answer'}], self.shared, self.tools)
        request, timeout = transport.call_args.args
        body = json.loads(request.data)
        self.assertEqual(request.full_url, 'https://api.openai.com/v1/responses')
        self.assertEqual(timeout, 60)
        self.assertFalse(body['store'])
        self.assertIn('Previous question', json.dumps(body['input']))
        self.assertNotIn(self.config.api_key, request.data.decode())
        self.assertEqual(result.api_calls, 1)
        self.assertEqual(result.usage, {'input_tokens': 100, 'output_tokens': 20})

    def test_selection_and_inspection_accept_more_than_thirty_records(self):
        samples = [sample(i) for i in range(80)]
        ids = [s['sample_id'] for s in samples]
        shared = research_context(samples, self.methods, self.state, self.context, self.space, 'REAL', ids)
        self.assertEqual(shared['selected_record_count'], 80)
        self.assertEqual(len(shared['shared_records']), 80)
        self.assertEqual(shared['sharing_mode'], 'FULL_DETAILS')
        self.assertEqual(shared['available_sample_ids'], ids)
        tools = ResearchTools(samples, self.methods, self.state, self.context, self.space, 'REAL', ids)
        self.assertEqual(len(tools.run('inspect_samples', {'sample_ids': ids})['records']), 80)

    def test_large_selection_pages_all_details_and_analyzes_every_selected_sample(self):
        samples = [sample(i) for i in range(400)] + [sample(999)]
        ids = [s['sample_id'] for s in samples[:400]]
        shared = research_context(samples, self.methods, self.state, self.context, self.space, 'REAL', ids)
        self.assertEqual(shared['selected_record_count'], 400)
        self.assertEqual(shared['selection_summary']['total'], 400)
        self.assertEqual(shared['sharing_mode'], 'LOCAL_ANALYSIS_AND_PAGED_DETAILS')
        self.assertLess(len(shared['shared_records']), 400)
        tools = ResearchTools(samples, self.methods, self.state, self.context, self.space, 'REAL', ids)
        seen, offset = [], 0
        while offset is not None:
            page = tools.run('browse_shared_records', {'offset': offset, 'batch_id': None})
            seen.extend(r['sample_id'] for r in page['records'])
            self.assertEqual(page['total_records'], 400)
            self.assertLess(len(json.dumps(page)), 160000)
            offset = page['next_offset']
        self.assertEqual(seen, ids)
        result = tools.run('analyze_experiments', {'scope': 'shared_selection'})
        self.assertEqual(result['metrics']['samples'], 400)
        self.assertEqual(result['metrics']['passed'], 400)
        inspected = tools.run('inspect_samples', {'sample_ids': ids})
        self.assertEqual(len(inspected['records']) + len(inspected['remaining_sample_ids']), 400)
        with self.assertRaises(LanguageError):
            tools.run('inspect_samples', {'sample_ids': ['S999']})

    def test_paged_tool_rejects_invalid_offsets_and_keeps_batch_filter_in_selection(self):
        samples = [sample(i) for i in range(40)]
        ids = [s['sample_id'] for s in samples[:35]]
        tools = ResearchTools(samples, self.methods, self.state, self.context, self.space, 'REAL', ids)
        for offset in (-1, True, 36):
            with self.assertRaises(LanguageError):
                tools.run('browse_shared_records', {'offset': offset, 'batch_id': None})
        self.assertEqual(tools.run('browse_shared_records', {'offset': 0, 'batch_id': 'B0'})['records'][0]['sample_id'], 'S0')
        self.assertEqual(tools.run('browse_shared_records', {'offset': 0, 'batch_id': 'B39'})['records'], [])
        empty = ResearchTools(samples, self.methods, self.state, self.context, self.space, 'REAL', [])
        self.assertFalse(empty.run('analyze_experiments', {'scope': 'shared_selection'})['available'])

    def test_large_shared_context_reaches_provider_without_silent_selection_drop(self):
        samples = [sample(i) for i in range(400)]
        ids = [s['sample_id'] for s in samples]
        shared = research_context(samples, self.methods, self.state, self.context, self.space, 'REAL', ids)
        tools = ResearchTools(samples, self.methods, self.state, self.context, self.space, 'REAL', ids)
        transport = Mock(side_effect=[function_response('analyze_experiments', {'scope': 'shared_selection'}), response()])
        result = ResearchAdapter(self.config, True, transport).converse('Analyze all 400 shared records', [], shared, tools)
        self.assertEqual(result.trace[0]['result']['metrics']['samples'], 400)
        self.assertEqual(result.api_calls, 2)
        for call in transport.call_args_list:
            self.assertLess(len(call.args[0].data), 250000)

    def test_local_tool_outputs_and_encrypted_reasoning_are_replayed_in_order(self):
        bodies = []
        def transport(request, timeout):
            bodies.append(json.loads(request.data))
            return function_response() if len(bodies) == 1 else response('One recorded sample passed screening.')
        result = ResearchAdapter(self.config, True, transport).converse('Summarize my data', [], self.shared, self.tools)
        self.assertEqual(result.api_calls, 2)
        self.assertEqual(result.trace[0]['result']['summary']['total'], 1)
        items = bodies[1]['input']
        self.assertEqual([item.get('type') for item in items[-3:]], ['reasoning', 'function_call', 'function_call_output'])
        self.assertEqual(items[-3]['encrypted_content'], 'OPAQUE_REASONING')

    def test_tool_cannot_read_unselected_sample_or_run_arbitrary_code(self):
        before = copy.deepcopy(self.samples)
        for name, args in [('run_code', {'code': 'DELETE ALL'}), ('inspect_samples', {'sample_ids': ['S1']}), ('summarize_experiments', {'sql': 'DELETE'}), ('preview_next_batch', {'batch_size': 100, 'pulse_counts': None}), ('preview_next_batch', {'batch_size': 2, 'pulse_counts': [8]})]:
            with self.assertRaises(LanguageError):
                self.tools.run(name, args)
        self.assertEqual(before, self.samples)

    def test_preview_proposals_remain_unmeasured_and_unreserved(self):
        result = self.tools.run('preview_next_batch', {'batch_size': 2, 'pulse_counts': [1]})
        self.assertTrue(result['preview_only'])
        self.assertEqual(len(self.tools.previews), 1)
        self.assertIsNone(self.tools.previews[0]['p_screen_pass'])
        self.assertEqual(len(self.samples), 1)
        self.assertEqual(self.tools.previews[0]['state'], 'OPEN')

    def test_analysis_and_parameter_tools_have_explicit_denominators(self):
        result = self.tools.run('analyze_experiments', {'scope': 'current_cohort'})
        self.assertEqual(result['metrics']['labeled'], 1)
        result = self.tools.run('analyze_parameter', {'parameter': 'element_concentration_mM', 'element': 'Fe'})
        self.assertEqual(result['rates'][0]['labeled'], 1)
        self.assertIsNone(result['association_with_screening'][0]['spearman_rho'])

    def test_compatible_chat_completion_tool_loop(self):
        transport = Mock(side_effect=[{'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'C1', 'type': 'function', 'function': {'name': 'inspect_samples', 'arguments': '{"sample_ids":["S0"]}'}}]}}]}, {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'I inspected the recorded Fe sample.'}}]}])
        config = ApiConfig('https://api.example.test/v1', 'test-model', 'test-secret')
        result = ResearchAdapter(config, True, transport).converse('Discuss Fe', [], self.shared, self.tools)
        self.assertEqual(result.api_calls, 2)
        body = json.loads(transport.call_args.args[0].data)
        self.assertEqual(body['messages'][-1]['role'], 'tool')
        self.assertEqual(body['messages'][-1]['tool_call_id'], 'C1')

    def test_unknown_tools_incomplete_replies_and_budget_exhaustion_stop(self):
        for value in [function_response('delete_records'), {'status': 'incomplete', 'output': []}, {'status': 'completed', 'output': []}]:
            transport = Mock(return_value=value)
            with self.assertRaises(LanguageError):
                ResearchAdapter(self.config, True, transport).converse('Discuss', [], self.shared, self.tools)
            transport.assert_called_once()
        transport = Mock(return_value=function_response())
        with self.assertRaises(LanguageError):
            ResearchAdapter(self.config, True, transport).converse('Discuss', [], self.shared, self.tools)
        self.assertEqual(transport.call_count, 3)

    def test_credentials_are_blocked_in_user_history_and_redacted_in_reply(self):
        for request, history in [('test-secret', []), ('Discuss', [{'role': 'user', 'content': 'test-secret'}])]:
            transport = Mock()
            with self.assertRaises(LanguageError):
                ResearchAdapter(self.config, True, transport).converse(request, history, self.shared, self.tools)
            transport.assert_not_called()
        result = ResearchAdapter(self.config, True, Mock(return_value=response('test-secret'))).converse('Discuss', [], self.shared, self.tools)
        self.assertNotIn('test-secret', result.answer.text)

    def test_provider_errors_are_sanitized_and_not_retried(self):
        transport = Mock(side_effect=HTTPError('SECRET_URL', 401, 'test-secret', {}, None))
        with self.assertRaises(LanguageError) as error:
            ResearchAdapter(self.config, True, transport).converse('Discuss', [], self.shared, self.tools)
        self.assertNotIn('test-secret', str(error.exception))
        self.assertIn('HTTP 401, /responses', str(error.exception))
        transport.assert_called_once()

    def test_research_authentication_matches_catalog_normalization(self):
        config = ApiConfig(self.config.base_url, ' test-model ', ' test-secret ', 'responses')
        transport = Mock(return_value=response())
        ResearchAdapter(config, True, transport).converse('Discuss', [], self.shared, self.tools)
        request = transport.call_args.args[0]
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-secret')
        self.assertEqual(json.loads(request.data)['model'], 'test-model')


if __name__ == '__main__':
    unittest.main()
