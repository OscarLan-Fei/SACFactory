"""Canonical ICP mass percentages and explicit legacy unit conversion."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation

ICP_UNIT = "wt%"
W_HEADERS = ("w_wt_percent", "W", "W / wt%", "W (wt%)")


def canonical_unit(unit):
    text = str(unit or "").strip().lower().replace(" ", "")
    if text in {"wt%", "wt.%", "w%", "%", "mass%", "weight%"}:
        return ICP_UNIT
    if text in {"ug/g", "µg/g", "μg/g"}:
        return "ug/g"
    raise ValueError("ICP content must use W / wt% (mass percent). Solution concentrations require sample mass, digestion volume and dilution metadata before import.")


def to_wt_percent(value, unit):
    """Keep percent points and ND/BLQ; scale only explicitly declared ug/g."""
    unit = canonical_unit(unit)
    if unit == ICP_UNIT or value is None:
        return value
    text = str(value).strip()
    if text.upper() in {"", "ND", "N.D.", "NA", "N/A", "NAN", "NOT_PROVIDED", "BLQ", "<LOQ", "BLOQ"}:
        return value
    operator = text[0] if text.startswith(("<", "≤")) else ""
    try:
        number = Decimal(text[1:] if operator else text)
        if not number.is_finite():
            raise InvalidOperation
        converted = number / Decimal(10000)
    except InvalidOperation:
        raise ValueError("ICP value must be numeric, ND, BLQ or a numeric censoring bound.") from None
    if isinstance(value, (int, float)) and not operator:
        return float(converted)
    return operator + format(converted, "f")


def normalize_assay(assay):
    result = deepcopy(assay)
    headers = [h for h in W_HEADERS if h in result]
    if len(headers) > 1 or headers and "value" in result:
        raise ValueError("Use one ICP content column: w_wt_percent (W / wt%).")
    if headers:
        result["value"] = result.pop(headers[0])
        unit = result.get("unit") or ICP_UNIT
        if canonical_unit(unit) != ICP_UNIT:
            raise ValueError("A W / wt% column cannot declare a different unit.")
    else:
        unit = result.get("unit") or ICP_UNIT
    result["value"] = to_wt_percent(result.get("value", ""), unit)
    result["unit"] = ICP_UNIT
    return result


def normalize_method(method, legacy=False):
    result = deepcopy(method)
    unit = result.get("unit") or ("ug/g" if legacy else ICP_UNIT)
    for limits in result.get("limits", {}).values():
        for key in ("lod", "loq"):
            if limits.get(key) is not None:
                limits[key] = to_wt_percent(limits[key], unit)
    result["unit"] = ICP_UNIT
    return result


def normalize_payload(payload):
    """Normalize active assay arrays, including stored planning contexts."""
    result = deepcopy(payload)
    if isinstance(result, dict):
        for key, value in result.items():
            if key == "icp" and isinstance(value, list):
                result[key] = [normalize_assay(a) for a in value]
            elif isinstance(value, (dict, list)):
                result[key] = normalize_payload(value)
    elif isinstance(result, list):
        result = [normalize_payload(v) for v in result]
    return result
