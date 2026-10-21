"""Chemistry provenance, state gates, transfer ablations and recipe identity."""
import copy
import json
import unittest

import numpy as np

from sacfactory.adaptive import HierarchicalKernel, eligible_records
from sacfactory.chemical_descriptors import descriptor_matrix, coverage
from sacfactory.chemistry_validation import chemistry_ablation
from sacfactory.composition import transfer_feature, model_feature
from sacfactory.domain import recipe_key, validate_sample
from sacfactory.importers import sample_tables, samples_from_rows, recommendation_workbook, read_table
from sacfactory.precursors import normalize_precursors, precursor_features, precursor_coverage, ionic_reference, state_reference
from sacfactory.planning import fit_model, recommend
from sacfactory.queueing import controlled_recipe_key
from test_transfer import source, context


def precursor(compound='FeCl2', kind='CHLORIDE', oxidation=2):
    return {'compound': compound, 'type': kind, 'oxidation_state': oxidation,
            'source': 'Synthetic test reagent record', 'coordination': 'UNKNOWN', 'spin': 'UNKNOWN', 'state_source': ''}


def campaign():
    rows = []
    for element in ['Fe', 'Co']:
        for i in range(18):
            row = source(i, element, passed=i >= 8)
            row.update(sample_id=f'{element}_{i}', batch_id=f'{element}_B{i}', precursor_family='SALTS',
                       precursors={element: precursor(element+'Cl2', 'CHLORIDE')})
            rows.append(validate_sample(row))
    for i in range(9):
        row = source(i, 'Fe', passed=i >= 4)
        row.update(sample_id=f'MIX_{i}', batch_id=f'MIX_B{i}', precursor_family='SALTS', elements=['Fe', 'Co'],
                   concentrations_mM={'Fe': .02+i*.006, 'Co': .035},
                   precursors={'Fe': precursor(), 'Co': precursor('Co(NO3)2', 'NITRATE')})
        rows.append(validate_sample(row))
    return rows


class ChemistryTests(unittest.TestCase):
    def test_reference_units_nulls_and_exact_state_match(self):
        data = state_reference()
        self.assertEqual(len(data['elements']), 118)
        self.assertEqual(len(data['ionic_radii']), 507)
        self.assertEqual(data['units']['ionic_radius'], 'pm')
        self.assertTrue(any(r['atomic_radius_rahm'] is None for r in data['elements']))
        self.assertEqual(coverage(['Fe'])[0]['en_pauling'], 1.83)
        entry = precursor()
        self.assertIsNone(ionic_reference('Fe', entry)[0])
        entry.update(coordination='VI', spin='HS', state_source='Test-only explicit state evidence')
        radius, status = ionic_reference('Fe', entry)
        self.assertEqual(status, 'EXACT_REFERENCE_MATCH')
        self.assertEqual(radius['ionic_radius'], 78.)
        entry['spin'] = 'LS'
        radius, _ = ionic_reference('Fe', entry)
        self.assertEqual(radius['ionic_radius'], 61.)
        self.assertIn('estimated', radius['origin'])
        for key in ['oxidation_state', 'coordination', 'spin', 'state_source']:
            missing = dict(entry)
            missing[key] = None
            self.assertIsNone(ionic_reference('Fe', missing)[0])

    def test_unknown_chemistry_has_masks_and_never_uses_common_oxidation_state(self):
        row = source(3)
        row['precursors'] = {'Fe': precursor()}
        features = precursor_features(row, True)
        self.assertEqual(features[-3:], [0., 0., 0.])
        self.assertEqual(precursor_coverage(row)[0]['ionic_status'], 'STATE_UNRESOLVED')
        row['precursors']['Fe']['source'] = ''
        encoded = precursor_features(row)
        self.assertEqual(encoded[0], 1.)
        self.assertEqual(encoded[1], 0.)
        self.assertEqual(encoded[9:12], [0., 0., 0.])
        self.assertIsNone(normalize_precursors({'Fe': {'oxidation_state': float('nan')}}, ['Fe'])['Fe']['oxidation_state'])
        for value in [float('inf'), 1.5, 9, 'guess']:
            with self.assertRaises(ValueError):
                normalize_precursors({'Fe': {'oxidation_state': value}}, ['Fe'])

    def test_permutation_invariant_precursor_features_and_element_conditioned_types(self):
        row = campaign()[-1]
        original = precursor_features(row)
        reordered = copy.deepcopy(row)
        reordered['elements'] = list(reversed(row['elements']))
        np.testing.assert_allclose(original, precursor_features(reordered))
        swapped = copy.deepcopy(row)
        swapped['precursors']['Fe']['type'], swapped['precursors']['Co']['type'] = 'NITRATE', 'CHLORIDE'
        self.assertFalse(np.allclose(original, precursor_features(swapped)))
        outcomes = copy.deepcopy(row)
        outcomes['icp_review']['state'] = 'FAIL'
        outcomes['temperature_C'] = 1600
        np.testing.assert_allclose(original, precursor_features(outcomes))

    def test_chemistry_identity_separates_salts_and_pending_forecasts(self):
        row = source(0)
        unknown = dict(row, precursors={'Fe': {}})
        self.assertEqual(recipe_key(row), recipe_key(unknown))
        self.assertEqual(controlled_recipe_key(row), controlled_recipe_key(unknown))
        known = dict(row, precursors={'Fe': precursor()})
        changed = copy.deepcopy(known)
        changed['precursors']['Fe']['compound'] = 'Fe(NO3)2'
        changed['precursors']['Fe']['type'] = 'NITRATE'
        self.assertNotEqual(recipe_key(known), recipe_key(changed))
        self.assertNotEqual(controlled_recipe_key(known), controlled_recipe_key(changed))
        changed = copy.deepcopy(known)
        changed['precursors']['Fe']['source'] = 'Additional documentation'
        self.assertEqual(recipe_key(known), recipe_key(changed))

    def test_chemical_roundtrip_legacy_imports_and_reporting_workbook(self):
        row = validate_sample(dict(source(4), precursors={'Fe': precursor()}))
        samples, details = sample_tables([row])
        restored = samples_from_rows(samples, details)[0]
        self.assertEqual(restored['precursors'], row['precursors'])
        legacy_details = [{k: v for k, v in r.items() if not k.startswith('precursor_') or k == 'precursor_mM'} for r in details]
        legacy = samples_from_rows(samples, legacy_details)[0]
        self.assertEqual(legacy.get('precursors', {}), {})
        proposal = {'recommendation_id': 'TEST_PLAN', 'origin': 'REAL', 'recipe': row}
        book = recommendation_workbook([proposal])
        restored = samples_from_rows(read_table(book, 'batch.xlsx', 'samples'), read_table(book, 'batch.xlsx', 'icp'))[0]
        self.assertEqual(restored['precursors'], row['precursors'])

    def test_new_kernel_psd_and_precursor_schema_reaches_acquisition(self):
        rows = campaign()
        for profile in ['chemistry_precursor', 'chemistry_state']:
            x = np.array([transfer_feature(r, profile) for r in rows[::3]])
            kernel = HierarchicalKernel(positions=descriptor_matrix(profile))
            self.assertGreaterEqual(np.linalg.eigvalsh(kernel(x)).min(), -1e-8)
            np.testing.assert_allclose(np.diag(kernel(x)), kernel.diag(x))
        target = copy.deepcopy(rows[-1])
        state = fit_model(rows, {}, target, 'REAL', True, 'reliability',
                          {'learning_iterations': 2, 'descriptor_profile': 'chemistry_precursor'})
        space = {'concentration_bounds_mM': {e: {'min_mM': .01, 'max_mM': .15} for e in target['elements']},
                 'pulses': [1, 2], 'pulse_limit_min_C': 900, 'pulse_limit_max_C': 1100, 'max_total_mM': .3,
                 'verified': True, 'baseline_repeat': False, 'repeat_strategy': 'none', 'candidate_budget': 256}
        proposals, _ = recommend(state, rows, target, space, 'REAL', 2)
        self.assertEqual(len(proposals), 2)
        for proposal in proposals:
            self.assertEqual(proposal['recipe']['precursors'], target['precursors'])
            self.assertTrue(0 <= proposal['p_screen_pass'] <= 1)
            self.assertEqual(len(model_feature(proposal['recipe'], state.model)), state.model.gp.x.shape[1])
        self.assertIn('state_descriptor_snapshot_sha256', state.info)

    def test_cross_salt_transfer_keeps_support_protocol_and_namespace_gates(self):
        row = source(0)
        target = copy.deepcopy(row)
        target['precursor_family'] = 'CHLORIDES'
        row['precursor_family'] = 'NITRATES'
        self.assertEqual(len(eligible_records([row], {}, target, 'REAL', True, 'chemistry_precursor')[0]), 0)
        row['precursors'] = {'Fe': precursor('Fe(NO3)2', 'NITRATE')}
        target['precursors'] = {'Fe': precursor()}
        self.assertEqual(len(eligible_records([row], {}, target, 'REAL', True)[0]), 0)
        self.assertEqual(len(eligible_records([row], {}, target, 'REAL', True, 'chemistry_precursor')[0]), 1)
        for key, value in [('support_batch', 'OTHER_SUPPORT'), ('protocol_id', 'OTHER_PROTOCOL'), ('origin', 'DEMO')]:
            changed = dict(row, **{key: value})
            self.assertEqual(len(eligible_records([changed], {}, target, 'REAL', True, 'chemistry_precursor')[0]), 0)

    def test_direct_adaptive_chemical_validation_uses_matching_feature_schema(self):
        rows = campaign()
        state = fit_model(rows, {}, rows[-1], 'REAL', True, 'adaptive',
                          {'learning_iterations': 2, 'descriptor_profile': 'chemistry_state'})
        self.assertIsNotNone(state.model)
        self.assertIsNotNone(state.info['validation'])
        self.assertEqual(state.model.descriptor_profile, 'chemistry_state')
        self.assertEqual(len(state.info['model_comparison']), 3)
        self.assertIn('state_descriptor_snapshot_sha256', state.info)

    def test_ablation_identical_rows_single_to_mixture_purge_and_missing_state_equivalence(self):
        rows = campaign()
        rows[0]['batch_id'] = rows[-1]['batch_id']
        frame, info = chemistry_ablation(rows, {}, rows[-1], 'REAL', iterations=2)
        self.assertEqual(frame.profile.nunique(), 9)
        evaluated = [set(group.sample_id) for _, group in frame.groupby('profile')]
        self.assertTrue(all(s == evaluated[0] for s in evaluated))
        for fold in info['folds']:
            self.assertEqual(fold['train_stages'], [1])
            self.assertEqual(fold['test_stages'], [2])
            self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))
            self.assertNotIn(rows[0]['sample_id'], fold['train_sample_ids'])
            for model in fold['models']:
                self.assertFalse(set(model['kernel_fit_ids']) & set(fold['test_sample_ids']))
                self.assertFalse(set(model['calibration_ids']) & set(fold['test_sample_ids']))
        wide = frame.pivot(index='sample_id', columns='profile', values='p_screen_pass')
        np.testing.assert_allclose(wide.chemistry_precursor, wide.chemistry_state, atol=1e-7)
        self.assertEqual(info['ionic_rows_with_exact_state_match'], 0)
        self.assertEqual(len(info['paired_effects']), 8)
        self.assertTrue(all(row['bootstrap_lower_95'] is not None for row in info['paired_effects']))
        json.dumps(info, allow_nan=False)

    def test_ablation_strict_unseen_element_purges_all_containing_compositions(self):
        rows = campaign()
        frame, info = chemistry_ablation(rows, {}, rows[-1], 'REAL', protocol='unseen_element', element='Co', iterations=2)
        self.assertFalse(frame.empty)
        by_id = {s['sample_id']: s for s in rows}
        for fold in info['folds']:
            self.assertTrue(all('Co' not in by_id[s]['elements'] for s in fold['train_sample_ids']))
            self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))


if __name__ == '__main__':
    unittest.main()
