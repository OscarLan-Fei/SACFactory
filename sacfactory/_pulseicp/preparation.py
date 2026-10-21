"""Auditable salt/volume conversion with compatibility for older mass-based records."""
from functools import lru_cache
import math
import re
from .references import elements, snapshot

PREPARATION_COLUMNS = ['precursor_formula', 'precursor_type', 'metal_atoms_per_formula',
    'salt_amount_mmol', 'metal_amount_mmol', 'dmf_volume_mL',
    'volume_basis', 'precursor_source_url', 'quantity_input_mode']
LEGACY_PREPARATION_COLUMNS = ['dmf_mass_g', 'dmf_density_g_mL', 'dmf_density_source_url']


def present(value):
    return value is not None and str(value).strip().lower() not in {'', 'nan', '<na>', 'none'}


def positive_number(value, label):
    if isinstance(value, bool):
        raise ValueError(label+' must be a positive number, not a boolean.')
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ValueError(label+' must be explicitly recorded as a number.') from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(label+' must be finite and positive.')
    return value


def dmf_volume(mass_g, density_g_mL):
    return positive_number(mass_g, 'DMF mass')/positive_number(density_g_mL, 'DMF density')


def salt_to_mM(salt_mmol, volume_mL, metal_atoms=1):
    count = positive_number(metal_atoms, 'Metal atoms per formula')
    if not count.is_integer():
        raise ValueError('Metal atoms per formula must be a positive integer.')
    return 1000*positive_number(salt_mmol, 'Salt amount')*count/positive_number(volume_mL, 'Solution volume')


def mM_to_salt(concentration_mM, volume_mL, metal_atoms=1):
    """Convert target-element concentration to precursor formula-unit mmol."""
    return positive_number(concentration_mM, 'Metal concentration')/salt_to_mM(1., volume_mL, metal_atoms)


def formula_atoms(formula):
    """Parse explicit integer stoichiometry, parentheses and dot hydrates.

    No charge, variable hydrate, empirical mixture or oxidation state is guessed.
    """
    text = str(formula).strip().translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
    text = text.replace(' ', '')
    # A decimal atom count must not silently become an adduct coefficient.
    # ASCII hydrate notation remains accepted; use the explicit middle dot for
    # other numbered adducts so the recorded molecular amount is unambiguous.
    for match in re.finditer(r'(?<=\d)\.(?=\d)', text):
        suffix = text[match.end():].split('.')[0].split('·')[0]
        if not re.fullmatch(r'[1-9][0-9]*H2O', suffix):
            raise ValueError('Ambiguous decimal stoichiometry; use an explicit middle dot for an adduct.')
    text = text.replace('·', '.')
    if not text:
        raise ValueError('Record the actual precursor formula.')
    total = {}
    for part in text.split('.'):
        coefficient = re.match(r'^([1-9][0-9]*)(?=[A-Z(\[])', part)
        factor = int(coefficient[1]) if coefficient else 1
        body = part[coefficient.end():] if coefficient else part
        tokens = re.findall(r'[A-Z][a-z]?|[0-9]+|[()\[\]]', body)
        if not body or ''.join(tokens) != body:
            raise ValueError('Use an explicit formula with integer stoichiometry; variable hydrates/charges need an explicit metal-atom override.')
        stack, brackets = [{}], []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in ('(', '['):
                stack.append({}); brackets.append(token); i += 1
                continue
            if token in (')', ']'):
                if len(stack) == 1 or brackets.pop() != ( '(' if token == ')' else '[' ):
                    raise ValueError('Unbalanced precursor formula.')
                group = stack.pop()
                if not group:
                    raise ValueError('Empty group in precursor formula.')
                i += 1
                multiplier = int(tokens[i]) if i < len(tokens) and tokens[i].isdigit() else 1
                if i < len(tokens) and tokens[i].isdigit():
                    i += 1
                if multiplier <= 0:
                    raise ValueError('Formula coefficients must be positive integers.')
                for e,n in group.items():
                    stack[-1][e] = stack[-1].get(e, 0)+n*multiplier
                continue
            if token not in elements():
                raise ValueError('Unknown element or coefficient placement in precursor formula.')
            i += 1
            multiplier = int(tokens[i]) if i < len(tokens) and tokens[i].isdigit() else 1
            if i < len(tokens) and tokens[i].isdigit():
                i += 1
            if multiplier <= 0:
                raise ValueError('Formula coefficients must be positive integers.')
            stack[-1][token] = stack[-1].get(token, 0)+multiplier
        if len(stack) != 1:
            raise ValueError('Unbalanced precursor formula.')
        for e,n in stack[0].items():
            total[e] = total.get(e, 0)+n*factor
    return total


def metal_count(formula, target, override=None):
    """Use explicit formula count; an override is only for unparsed formulas."""
    try:
        parsed = formula_atoms(formula)
    except ValueError:
        # A variable water hydrate does not change the count of a non-H/O target.
        # No hydrate coefficient or molar mass is inferred from this exception.
        parts = str(formula).replace(' ', '').replace('·', '.').split('.')
        if target not in {'H', 'O'} and len(parts)>1 and all(re.fullmatch(r'[xn]H2O|[1-9][0-9]*H2O|H2O', p) for p in parts[1:]):
            parsed = formula_atoms(parts[0])
            if target not in parsed:
                raise ValueError(f'The precursor formula does not contain {target}.')
            if override is not None and positive_number(override, 'Metal-atom override') != parsed[target]:
                raise ValueError('The metal-atom override conflicts with the explicit formula.')
            return parsed[target]
        if override is None:
            raise
        if not present(formula) or target not in re.findall(r'[A-Z][a-z]?', str(formula)):
            raise ValueError('The recorded formula must identify the selected target element.')
        count = positive_number(override, 'Metal-atom override')
        if not count.is_integer():
            raise ValueError('The metal-atom override must be a positive integer.')
        return int(count)
    if target not in parsed:
        raise ValueError(f'The precursor formula does not contain {target}.')
    if override is not None and positive_number(override, 'Metal-atom override') != parsed[target]:
        raise ValueError('The metal-atom override conflicts with the explicit formula.')
    return parsed[target]


def precursor_type(formula):
    text = str(formula)
    if re.search(r'N(?:O3|O₃)', text):
        return 'NITRATE'
    if 'Cl' in text:
        return 'CHLORIDE'
    return 'OTHER'


@lru_cache(maxsize=1)
def catalog():
    return {row['element']:row for row in snapshot('sigma_precursors.json')['entries']}


def solvent_reference():
    return snapshot('solvent_reference.json')


def normalize_preparation(row):
    """Derive recorded inputs without filling unknown experimental chemistry.

    Existing canonical values must agree with supplied amount/direct-volume fields.
    Legacy mass/density conversion applies only when direct DMF volume is absent.
    Without those optional fields, legacy concentration/volume input is unchanged.
    """
    result = dict(row)
    volume = result.get('volume_mL')
    if present(result.get('dmf_volume_mL')):
        entered = positive_number(result['dmf_volume_mL'], 'DMF volume')
        basis = result.get('volume_basis')
        if not present(basis):
            basis = 'DMF_VOLUME_NOMINAL'
        if basis == 'MEASURED_FINAL_SOLUTION_VOLUME':
            volume = positive_number(volume, 'Measured final solution volume')
        elif basis == 'DMF_VOLUME_NOMINAL':
            if present(volume) and not math.isclose(positive_number(volume,'Solution volume'),entered,rel_tol=1e-6,abs_tol=1e-8):
                raise ValueError('Recorded volume conflicts with entered DMF volume; declare measured final volume when applicable.')
            volume = entered
        else:
            raise ValueError('Unknown volume basis for direct DMF volume input.')
        result.update(dmf_volume_mL=entered, volume_mL=volume, volume_basis=basis)
    elif present(result.get('dmf_mass_g')):
        # Legacy CSV compatibility only; the current UI uses direct DMF mL.
        if not present(result.get('dmf_density_g_mL')):
            raise ValueError('A recorded DMF density is required for mass-to-volume conversion.')
        estimated = dmf_volume(result['dmf_mass_g'], result['dmf_density_g_mL'])
        basis = result.get('volume_basis', 'DMF_MASS_DENSITY_ESTIMATE')
        if not present(basis):
            basis = 'DMF_MASS_DENSITY_ESTIMATE'
        if basis == 'MEASURED_FINAL_SOLUTION_VOLUME':
            volume = positive_number(volume, 'Measured final solution volume')
        elif basis == 'DMF_MASS_DENSITY_ESTIMATE':
            if present(volume) and not math.isclose(positive_number(volume, 'Solution volume'),estimated,rel_tol=1e-6,abs_tol=1e-8):
                raise ValueError('Recorded volume conflicts with DMF mass/density; declare measured final volume when applicable.')
            volume = estimated
        else:
            raise ValueError('Unknown volume basis.')
        result.update(volume_mL=volume, volume_basis=basis)
    if present(result.get('salt_amount_mmol')):
        if not present(result.get('precursor_formula')):
            raise ValueError('Salt-amount input requires the actual precursor formula.')
        count = metal_count(result['precursor_formula'], str(result.get('element','')).strip(),
            result.get('metal_atoms_per_formula') if present(result.get('metal_atoms_per_formula')) else None)
        concentration = salt_to_mM(result['salt_amount_mmol'], volume, count)
        if present(result.get('concentration_mM')) and not math.isclose(positive_number(result['concentration_mM'],'Concentration'),concentration,rel_tol=1e-6,abs_tol=1e-8):
            raise ValueError('Recorded mM conflicts with salt mmol, stoichiometry and volume.')
        result.update(concentration_mM=concentration, metal_atoms_per_formula=count)
    return result
