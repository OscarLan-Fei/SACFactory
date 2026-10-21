"""Extract sourced thermochemical escape proxies without inventing sublimation points."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = '76f415767b654634e51dead755513ef431abf200e892eab8468aa659ff218d1c'
    if hashlib.sha256(args.wheel.read_bytes()).hexdigest() != expected:
        raise ValueError('Use the verified mendeleev 1.3.0 wheel.')
    with ZipFile(args.wheel) as archive, tempfile.TemporaryDirectory() as folder:
        raw = archive.read('mendeleev/elements.db')
        database = Path(folder) / 'elements.db'
        database.write_bytes(raw)
        with closing(sqlite3.connect(database)) as connection:
            connection.row_factory = sqlite3.Row
            atoms = [dict(r) for r in connection.execute('SELECT symbol,heat_of_formation FROM elements ORDER BY atomic_number')]
            phases = [dict(r) for r in connection.execute('SELECT elements.symbol,phasetransitions.* FROM phasetransitions JOIN elements USING(atomic_number)')]
            metadata = [dict(r) for r in connection.execute("SELECT * FROM propertymetadata WHERE attribute_name IN ('heat_of_formation','melting_point','boiling_point')")]
    entries = []
    for atom in atoms:
        rows = [r for r in phases if r['symbol'] == atom['symbol'] and not r['is_transition']]
        def unique(field, selected):
            values = {r[field] for r in selected if r.get(field) is not None}
            return next(iter(values)) if len(values) == 1 else None
        sublimating = [r for r in rows if r['is_sublimation_point'] is True or r['is_sublimation_point'] == 1]
        vaporizing = [r for r in rows if r not in sublimating]
        entries.append({
            'element': atom['symbol'], 'melting_point_K': unique('melting_point', rows),
            'boiling_point_K': unique('boiling_point', vaporizing),
            'reported_sublimation_temperature_K': unique('boiling_point', sublimating),
            'sublimation_pressure_Pa': None,
            'sublimation_source': 'https://mendeleev.readthedocs.io/en/stable/data.html#phase-transitions' if sublimating else '',
            'sublimation_status': 'Reported flagged gas-transition point; pressure unresolved' if sublimating else 'No independent sublimation temperature in source',
            'heat_of_formation_kJ_mol': atom['heat_of_formation'],
            'energy_proxy_definition': 'Stored elemental heat_of_formation; gas-atom formation/atomization energy proxy for the fifty solid-metal targets, not a sublimation temperature or supported-atom barrier',
        })
    payload = {
        'version': 'mendeleev-1.3.0-loss-reference-v1', 'retrieved_date': '2026-10-13',
        'source_url': 'https://pypi.org/project/mendeleev/1.3.0/',
        'documentation': 'https://mendeleev.readthedocs.io/en/stable/data.html',
        'bibliography': 'https://mendeleev.readthedocs.io/en/stable/bibliography.html',
        'wheel_sha256': expected, 'database_sha256': hashlib.sha256(raw).hexdigest(),
        'license': 'MIT', 'property_metadata': metadata, 'entries': entries,
        'notes': [
            'The heat_of_formation citation is haynes2014crc. NIST gas-atom entries independently confirm the interpretation for Zn and W.',
            'Sublimation pressure depends on temperature. A reported sublimation temperature is not comparable without a pressure convention.',
            'No missing temperature, pressure or enthalpy is imputed. Unknown-allotrope ambiguity remains null.',
            'Database coverage does not validate millisecond supported-atom kinetics.',
        ],
        'interpretation_checks': [
            'https://webbook.nist.gov/cgi/cbook.cgi?ID=C7440666&Mask=2205&Units=SI',
            'https://webbook.nist.gov/cgi/cbook.cgi?ID=C7440337&Mask=D&Units=SI',
            'https://wtt-pro.nist.gov/wtt-pro/help/properties/CG_pressure.html',
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
