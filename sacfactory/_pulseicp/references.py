"""Offline attributed element references with explicit missingness."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
REVIEWED_METALS = 'Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Sr Y Zr Nb Mo Ru Rh Pd Ag In Sn Sb Ba Hf Ta W Ir Pt Au Pb Bi La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu'.split()
# Candidate metal targets extend beyond the original experiment palette. Reference
# sufficiency is checked separately; membership never fabricates a thermal value.
EXTENDED_METAL_CANDIDATES = 'Li Be Na Rb Tc Cd Cs Pm Re Os Hg Tl Po Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv'.split()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=8)
def snapshot(name):
    raw = (ROOT / 'resources' / name).read_bytes()
    data = json.loads(raw)
    data['snapshot_sha256'] = hashlib.sha256(raw).hexdigest()
    return data


@lru_cache(maxsize=1)
def elements():
    return {r['symbol']: r for r in snapshot('element_properties.json')['elements']}


@lru_cache(maxsize=1)
def thermal():
    return {r['element']: r for r in snapshot('element_thermal_references.json')['entries']}


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def supported_elements():
    candidates = ['Fe']+[m for m in REVIEWED_METALS if m != 'Fe']+EXTENDED_METAL_CANDIDATES
    return [e for e in candidates if positive(elements()[e].get('atomic_weight')) and
            any(positive(thermal().get(e, {}).get(k)) for k in
                ('melting_point_K', 'boiling_point_K', 'heat_of_formation_kJ_mol'))]


def support_columns(element):
    fields = ('melting_point_K', 'boiling_point_K', 'heat_of_formation_kJ_mol')
    return {'element_reference_scope': 'ORIGINAL_REVIEWED_50' if element in REVIEWED_METALS else 'EXTENDED_REFERENCE_TRANSFER',
        'missing_thermal_fields': json.dumps([k for k in fields if not positive(thermal().get(element, {}).get(k))])}


def provenance():
    return {name: {k: v for k, v in snapshot(name).items() if k not in {'elements', 'entries'}}
            for name in ('element_properties.json', 'element_thermal_references.json')}
