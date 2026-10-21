"""Sourced candidate boundaries and the default ion/optional salt hypotheses."""
import copy
import unittest
from urllib.parse import urlsplit

import numpy as np

from sacfactory.domain import TARGETS
from sacfactory.precursor_catalog import catalog, catalog_rows, planning_draft, preference_key, reagent_defaults, register_defaults
from sacfactory.precursors import precursor_features, confirm_catalog_precursors, precursor_coverage
from sacfactory.composition import transfer_feature
from test_transfer import source


class CatalogTests(unittest.TestCase):
    def test_actual_reagent_defaults_override_catalogue_assumptions_but_not_explicit_recipes(self):
        import tempfile
        from sacfactory.storage import Store
        from sacfactory.ui_materials import planning_precursors
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder, 'REAL')
            library = planning_draft(['Cu', 'In', 'Au'])
            for entry in library.values():
                entry['source'] = 'User declared reagent identity'
            store.put_many('meta', [{'id': 'reagent_library', 'origin': 'REAL', 'precursors': library}], 'id')
            old = planning_draft(['Cu'])
            old['Cu'].update(compound='CuCl', oxidation_state=1)
            context = {'precursors': old}
            updated = planning_precursors(store, context, ['Cu', 'In', 'Au', 'Og'])
            self.assertEqual(updated['Cu']['compound'], 'CuCl2')
            self.assertEqual(updated['In']['oxidation_state'], 3)
            self.assertEqual(updated['Au']['oxidation_state'], 3)
            self.assertNotIn('Og', updated)
            self.assertEqual(context['precursors']['Cu']['compound'], 'CuCl')
            old['Cu']['source'] = 'Explicit alternate planned reagent'
            self.assertEqual(planning_precursors(store, context, ['Cu'])['Cu']['compound'], 'CuCl')
            demo_default = planning_precursors(Store(folder, 'DEMO'), {}, ['Cu'])['Cu']
            self.assertEqual(demo_default['compound'], 'CuCl2')
            self.assertTrue(demo_default['source'].startswith('PLANNED_CATALOG_ASSUMPTION:'))

    def test_registration_is_idempotent_namespace_local_and_never_backfills_samples(self):
        import tempfile
        from sacfactory.storage import Store
        with tempfile.TemporaryDirectory() as folder:
            real, demo = Store(folder, 'REAL'), Store(folder, 'DEMO')
            record = source(0)
            real.put_many('sample', [record], 'sample_id')
            before = real.all('sample')
            custom = {'Co': {**planning_draft(['Co'])['Co'],
                'compound': 'Co(NO3)2', 'type': 'NITRATE', 'source': 'User-edited actual reagent'}}
            real.put_many('meta', [{'id': 'reagent_library', 'origin': 'REAL', 'precursors': custom}], 'id')
            registered = register_defaults(real)
            self.assertEqual(len(registered), 50)
            self.assertEqual(registered['Co']['compound'], 'Co(NO3)2')
            self.assertEqual(registered['Fe']['compound'], 'FeCl2')
            self.assertEqual(real.all('sample'), before)
            self.assertIsNone(demo.get('meta', 'reagent_library'))
            audit = real.export()['audit']
            self.assertEqual(register_defaults(real), registered)
            self.assertEqual(real.export()['audit'], audit)
            other = register_defaults(demo)
            self.assertEqual(other['Co']['type'], 'CHLORIDE')
            self.assertEqual(len(other), 50)

    def test_all_reviewed_defaults_are_sourced_chlorides_with_no_nitrate_fallback(self):
        from unittest.mock import patch
        defaults = reagent_defaults()
        self.assertEqual(set(defaults), set(TARGETS))
        self.assertTrue(all(r['type'] == 'CHLORIDE' and r['compound'] and
            'https://www.sigmaaldrich.com/' in r['source'] for r in defaults.values()))
        with patch('sacfactory.precursor_catalog.catalog_rows', return_value=[{
            'element': 'Ag', 'compound': 'AgNO3', 'type': 'NITRATE',
            'formal_oxidation_state': 1, 'source_url': 'https://www.sigmaaldrich.com/'}]):
            self.assertEqual(planning_draft(['Ag']), {})

    def test_original_fifty_have_linked_candidates_without_actual_use_or_solution_states(self):
        data = catalog()
        self.assertEqual({r['element'] for r in data['entries']}, set(TARGETS))
        self.assertEqual(len(data['snapshot_sha256']), 64)
        for row in data['entries']:
            self.assertEqual(urlsplit(row['source_url']).hostname, 'www.sigmaaldrich.com')
            self.assertEqual(row['checked_date'], '2026-10-13')
            self.assertTrue(row['catalog_number'])
            self.assertTrue(row['cas'])
            self.assertEqual(row['status'], 'CATALOG_CANDIDATE_NOT_ACTUAL_USAGE')
            self.assertFalse(row['solution_state_verified'])
            self.assertIsNone(row['coordination'])
            self.assertIsNone(row['spin'])
        indexed = {r['element']: r for r in data['entries']}
        self.assertEqual(indexed['Cu']['formal_oxidation_state'], 2)
        for element, formula, identifier in [('Cu', 'CuCl2', '222011'), ('In', 'InCl3', '334065'), ('Au', 'AuCl3', '334049')]:
            self.assertEqual(indexed[element]['compound'], formula)
            self.assertEqual(indexed[element]['catalog_number'], identifier)
            self.assertIn('declared reagent preference', indexed[element]['notes'])
        self.assertEqual(indexed['Fe']['formal_oxidation_state'], 2)
        self.assertEqual(indexed['Eu']['formal_oxidation_state'], 2)
        self.assertIn('adduct', indexed['Nb']['notes'])
        self.assertIn('partner', indexed['Mo']['notes'])

    def test_preference_is_salt_family_then_verified_lower_oxidation_state(self):
        rows = [{'type': 'NITRATE', 'formal_oxidation_state': 1},
                {'type': 'CHLORIDE', 'formal_oxidation_state': 3},
                {'type': 'CHLORIDE', 'formal_oxidation_state': 2},
                {'type': 'OTHER', 'formal_oxidation_state': 1}]
        ordered = sorted(rows, key=preference_key)
        self.assertEqual([(r['type'], r['formal_oxidation_state']) for r in ordered],
                         [('CHLORIDE', 2), ('CHLORIDE', 3), ('NITRATE', 1), ('OTHER', 1)])

    def test_non_target_elements_remain_unresolved_and_no_ionic_state_is_guessed(self):
        self.assertIsNone(catalog_rows(['Og'])[0]['formal_oxidation_state'])
        self.assertEqual(planning_draft(['Og']), {})
        draft = planning_draft(['Fe'])
        self.assertEqual(draft['Fe']['compound'], 'FeCl2')
        self.assertEqual(draft['Fe']['oxidation_state'], 2)
        self.assertEqual(draft['Fe']['spin'], 'UNKNOWN')
        self.assertEqual(draft['Fe']['coordination'], 'UNKNOWN')

    def test_catalogue_assumptions_predict_hypotheses_but_stay_masked_in_historical_training(self):
        row = source(0)
        row['precursors'] = planning_draft(['Fe'])
        original = copy.deepcopy(row)
        future = precursor_features(row, include_types=False)
        training = precursor_features(row, include_types=False, for_training=True)
        self.assertEqual(future[9:12], [.25, 0., 1.])
        self.assertEqual(training[9:12], [0., 0., 0.])
        self.assertIsNone(precursor_coverage(row)[0]['ionic_radius_pm'])
        confirmed = confirm_catalog_precursors(row)
        np.testing.assert_allclose(precursor_features(confirmed, include_types=False, for_training=True), future)
        self.assertEqual(row, original)
        self.assertTrue(confirmed['precursors']['Fe']['source'].startswith('Actual reagent identity confirmed'))

    def test_ion_profile_ignores_counterion_without_erasing_the_reagent_record(self):
        row = source(0)
        row['precursors'] = confirm_catalog_precursors(dict(row, precursors=planning_draft(['Fe'])))['precursors']
        nitrate = copy.deepcopy(row)
        nitrate['precursors']['Fe'].update(compound='Fe(NO3)2', type='NITRATE')
        np.testing.assert_allclose(transfer_feature(row, 'chemistry_ion'), transfer_feature(nitrate, 'chemistry_ion'))
        self.assertFalse(np.allclose(transfer_feature(row, 'chemistry_precursor'), transfer_feature(nitrate, 'chemistry_precursor')))
        self.assertEqual(row['precursors']['Fe']['compound'], 'FeCl2')


if __name__ == '__main__':
    unittest.main()
