"""Extract phase-transition references from the verified mendeleev wheel."""
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
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = '76f415767b654634e51dead755513ef431abf200e892eab8468aa659ff218d1c'
    if hashlib.sha256(args.wheel.read_bytes()).hexdigest() != expected:
        raise ValueError('Use the verified mendeleev 1.3.0 wheel.')
    with ZipFile(args.wheel) as archive, tempfile.TemporaryDirectory() as folder:
        raw = archive.read('mendeleev/elements.db')
        path = Path(folder) / 'elements.db'
        path.write_bytes(raw)
        with closing(sqlite3.connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = [dict(r) for r in connection.execute('SELECT elements.symbol,phasetransitions.* FROM phasetransitions JOIN elements USING(atomic_number) ORDER BY phasetransitions.id')]
    content = {'version': 'mendeleev-1.3.0-phase-transitions-v1', 'retrieved_date': '2026-10-13',
        'source_url': 'https://pypi.org/project/mendeleev/1.3.0/',
        'documentation': 'https://mendeleev.readthedocs.io/en/stable/data.html#phase-transitions',
        'bibliography': 'https://mendeleev.readthedocs.io/en/stable/bibliography.html',
        'wheel_sha256': expected, 'database_sha256': hashlib.sha256(raw).hexdigest(), 'license': 'MIT',
        'units': {'melting_point': 'K', 'boiling_point': 'K', 'triple_point_temperature': 'K'},
        'note': 'Bulk phase references, not precursor decomposition or supported-atom volatility. Allotropes, transition/sublimation flags and nulls are preserved. No unique bulk reference is assigned when rows are ambiguous.',
        'phase_transitions': rows}
    args.output.write_text(json.dumps(content, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
