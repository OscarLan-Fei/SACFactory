"""Reproduce the five-property offline snapshot without importing mendeleev.

Usage: python scripts/extract_element_descriptors.py --wheel PATH --date YYYY-MM-DD
Use the official mendeleev 1.3.0 wheel. Output defaults to resources/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'resources')
    args = parser.parse_args()
    if not args.wheel.name.startswith('mendeleev-1.3.0-'):
        raise ValueError('Use the pinned mendeleev 1.3.0 wheel.')
    rows = []
    with ZipFile(args.wheel) as archive, tempfile.TemporaryDirectory() as folder:
        data = archive.read('mendeleev/elements.db')
        path = Path(folder)/'elements.db'; path.write_bytes(data)
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        try:
            for row in connection.execute('SELECT atomic_number,symbol,atomic_weight,en_pauling,covalent_radius_cordero,electron_affinity FROM elements ORDER BY atomic_number'):
                item = dict(row)
                energy = connection.execute('SELECT ionization_energy,uncertainty,"references",is_semi_empirical,is_theoretical FROM ionizationenergies WHERE atomic_number=? AND ion_charge=0', (item['atomic_number'],)).fetchone()
                item['first_ionization_energy'] = energy['ionization_energy'] if energy else None
                item['ionization_metadata'] = dict(energy) if energy else None
                rows.append(item)
        finally:
            connection.close()
        license_text = archive.read('mendeleev-1.3.0.dist-info/licenses/LICENSE')
    if len(rows) != 118 or len({r['symbol'] for r in rows}) != 118:
        raise ValueError('The reference must contain exactly 118 unique elements.')
    properties = {
        'atomic_weight': {'unit': 'Da', 'reference': 'mendeleev data: atomic_weight; source bibliography keys 39,73'},
        'en_pauling': {'unit': 'dimensionless', 'reference': 'mendeleev data: electronegativity_pauling; source bibliography key 25'},
        'covalent_radius_cordero': {'unit': 'pm', 'reference': 'mendeleev data: covalent_radius_cordero; source bibliography key 16'},
        'electron_affinity': {'unit': 'eV', 'reference': 'mendeleev data: electron_affinity; source bibliography keys 6,25'},
        'first_ionization_energy': {'unit': 'eV', 'reference': 'Neutral-atom ionizationenergies, ion_charge=0; per-element reference/estimated flags retained'}}
    payload = {'version': 'mendeleev-1.3.0-five-properties-v1', 'package_version': '1.3.0',
        'source_url': 'https://pypi.org/project/mendeleev/1.3.0/', 'documentation': 'https://mendeleev.readthedocs.io/en/stable/data.html',
        'bibliography': 'https://mendeleev.readthedocs.io/en/stable/bibliography.html',
        'wheel_sha256': hashlib.sha256(args.wheel.read_bytes()).hexdigest(), 'database_sha256': hashlib.sha256(data).hexdigest(),
        'license': 'MIT', 'retrieved_date': args.date, 'properties': properties, 'elements': rows,
        'missing_rule': 'JSON null retained. Model encoding uses a separate missing mask; no chemical measurement is fabricated.',
        'scope': 'Elemental reference descriptors, not precursor decomposition, ionic radius, oxidation state, support binding or SAC experimental data.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'element_properties.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    (args.output/'LICENSE.mendeleev').write_bytes(license_text)
    print('Extracted 118-element reference; restart the app to reload its cached snapshot.')


if __name__ == '__main__':
    main()
