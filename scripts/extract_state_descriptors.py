"""Extract attributed Rahm atomic and state-specific ionic radii from a pinned wheel."""
import argparse
from contextlib import closing
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
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'resources')
    args = parser.parse_args()
    expected = '76f415767b654634e51dead755513ef431abf200e892eab8468aa659ff218d1c'
    if hashlib.sha256(args.wheel.read_bytes()).hexdigest() != expected:
        raise ValueError('Use the verified official mendeleev 1.3.0 wheel.')
    with ZipFile(args.wheel) as archive, tempfile.TemporaryDirectory() as folder:
        raw = archive.read('mendeleev/elements.db')
        path = Path(folder) / 'elements.db'
        path.write_bytes(raw)
        with closing(sqlite3.connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            atoms = [dict(row) for row in connection.execute('SELECT atomic_number,symbol,atomic_radius_rahm FROM elements ORDER BY atomic_number')]
            ions = [dict(row) for row in connection.execute('SELECT elements.symbol,ionicradii.* FROM ionicradii JOIN elements USING(atomic_number) ORDER BY ionicradii.id')]
    payload = {
        'version': 'mendeleev-1.3.0-state-radii-v1', 'retrieved_date': args.date,
        'source_url': 'https://pypi.org/project/mendeleev/1.3.0/',
        'documentation': 'https://mendeleev.readthedocs.io/en/stable/data.html#ionic-radii',
        'bibliography': 'https://mendeleev.readthedocs.io/en/stable/bibliography.html',
        'wheel_sha256': expected, 'database_sha256': hashlib.sha256(raw).hexdigest(), 'license': 'MIT',
        'units': {'atomic_radius_rahm': 'pm', 'ionic_radius': 'pm', 'crystal_radius': 'pm'},
        'references': {'atomic_radius_rahm': ['https://doi.org/10.1002/chem.201602949', 'https://doi.org/10.1002/chem.201700610'],
                       'ionic_radius': ['https://doi.org/10.1107/S0567739476001551', 'mendeleev bibliography key 33 for actinoid(III), CN IX additions']},
        'scope': 'Reference radii, not a measured precursor solution structure or catalyst oxidation state. Null values and origin/reliability flags retained.',
        'elements': atoms, 'ionic_radii': ions,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'state_properties.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Extracted {len(atoms)} elemental rows and {len(ions)} explicitly state-indexed ionic rows.')


if __name__ == '__main__':
    main()
