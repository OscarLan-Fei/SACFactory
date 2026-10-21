"""Composition transfer from single elements through mixtures, with explicit priors.

Permutation-invariant membership features share source information. Pair and
exact-composition residuals retain unobserved mixture effects; they are learned
only from recorded mixture outcomes. No product-of-success rule or synthetic
mixture labels are used.
"""
from collections import Counter
from itertools import combinations
import math

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Kernel
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import GroupKFold

from .batching import PERIODS
from .domain import ELEMENTS, cohort_key, finite_number, recipe_key, screen, RULE_VERSION

POSITIONS = {symbol: (period, group, "s" if group <= 2 or symbol == "He" else "p" if group >= 13 else "d")
             for period, row in enumerate(PERIODS[:7], 1) for group, symbol in row.items() if symbol in ELEMENTS}
POSITIONS.update({symbol: (6, None, "lanthanoid") for symbol in PERIODS[7].values()})
POSITIONS.update({symbol: (7, None, "actinoid") for symbol in PERIODS[8].values()})
REGIONS = ("s", "p", "d", "lanthanoid", "actinoid")
MODEL_VERSION = "composition-transfer-fixed-kernel-v2"


def descriptors(element):
    period, group, region = POSITIONS[element]
    return [(ELEMENTS.index(element) + 1) / 20., period / 2., (group or 0) / 4., float(group is not None),
            *[float(region == value) for value in REGIONS]]


_positions = np.array([descriptors(element) for element in ELEMENTS])
ATOM_KERNEL = .5 * np.exp(-.5 * cdist(_positions, _positions, "sqeuclidean")) + .5 * np.eye(len(ELEMENTS))


def composition_key(sample):
    return tuple(sorted(sample["elements"], key=ELEMENTS.index))


def composition_name(sample):
    return "+".join(composition_key(sample))


def transfer_feature(sample, descriptor_profile='periodic', for_training=False):
    """Fixed schema across 1..118 elements; missing actual concentrations reject.

    Absent members have membership zero and masked log features zero. This
    encodes absence, never a substituted measured concentration or assay value.
    """
    from .precursors import base_profile
    requested_profile = descriptor_profile
    descriptor_profile = base_profile(descriptor_profile)
    elements = sample["elements"]
    if not elements or len(set(elements)) != len(elements) or set(elements) - set(ELEMENTS):
        raise ValueError("Transfer composition must contain unique target elements.")
    concentrations = {e: finite_number(sample["concentrations_mM"].get(e), f"{e} concentration", True) for e in elements}
    presence = [float(e in concentrations) for e in ELEMENTS]
    logs = [math.log(concentrations[e] / .05) / math.sqrt(len(elements)) if e in concentrations else 0. for e in ELEMENTS]
    process = [math.log(sum(concentrations.values()) / .05), math.log(len(elements)),
               (finite_number(sample.get("pulse_count"), "Pulse count", True) - 1) / 2.]
    if sample.get("pulse_limit_C") is not None:
        process.append((float(sample["pulse_limit_C"]) - 1000.) / 200.)
    values = presence + logs + process
    if descriptor_profile in {'chemistry_ion', 'chemistry_precursor', 'chemistry_state'}:
        if sample.get('pulse_limit_C') is None:
            raise ValueError('Precursor-aware prediction requires a controlled cutoff temperature.')
        from .precursors import precursor_features
        values += precursor_features(sample, include_state=descriptor_profile == 'chemistry_state',
                                     include_types=descriptor_profile != 'chemistry_ion', for_training=for_training)
    if requested_profile in {'chemistry_thermal', 'chemistry_loss', 'chemistry_mixing', 'chemistry_pulseicp'}:
        from .physics_features import thermal_recipe_features
        from .preparation import model_features as preparation_features
        values += thermal_recipe_features(sample)
        values += preparation_features(sample)
    if requested_profile == 'chemistry_loss':
        from .physics_features import loss_features
        values += loss_features(sample)
    if requested_profile in {'chemistry_mixing', 'chemistry_pulseicp'}:
        from .mixing import feature_vector
        values += feature_vector(sample)
    if requested_profile == 'chemistry_pulseicp':
        from .pulseicp_bridge import recipe_features
        values += recipe_features(sample)
    return values


def model_feature(sample, model, for_training=False):
    """Use the fitted model's schema for every prediction and acquisition call."""
    return transfer_feature(sample, getattr(model, 'descriptor_profile', 'periodic'), for_training)


class CompositionTransferKernel(Kernel):
    """PSD conditions kernel times shared members + pairs + composition identity.

    Shared membership is a normalized inner product under a PSD atom kernel.
    Pair covariance is a normalized dot product of conceptual pair indicators,
    computed from set intersection counts without enumerating all pair features.
    Single-element records have zero pair residual and cannot determine it.
    """
    def __init__(self, concentration_scale=.65, process_scales=(.9, 1., 1., 1.), weights=(.45, .20, .35)):
        if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights) or weights[0] + weights[2] <= 0:
            raise ValueError("Kernel weights must be nonnegative with positive member/composition variance.")
        if len(process_scales) != 4 or any(not math.isfinite(s) or s <= 0 for s in (*process_scales, concentration_scale)):
            raise ValueError("Kernel requires four positive process scales and a positive concentration scale.")
        self.concentration_scale = concentration_scale
        self.process_scales = process_scales
        self.weights = weights

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        dimension = len(ELEMENTS)
        px, py = X[:, :dimension], Y[:, :dimension]
        nx, ny = px.sum(axis=1), py.sum(axis=1)
        overlap = px @ py.T
        norm_x = np.sqrt(np.sum((px @ ATOM_KERNEL) * px, axis=1))
        norm_y = np.sqrt(np.sum((py @ ATOM_KERNEL) * py, axis=1))
        shared = (px @ ATOM_KERNEL @ py.T) / (norm_x[:, None] * norm_y[None, :])
        pair_x, pair_y = nx * (nx - 1) / 2, ny * (ny - 1) / 2
        pair_denominator = np.sqrt(pair_x[:, None] * pair_y[None, :])
        pairs = np.divide(overlap * (overlap - 1) / 2, pair_denominator, out=np.zeros_like(overlap), where=pair_denominator > 0)
        exact = ((overlap == nx[:, None]) & (overlap == ny[None, :])).astype(float)
        scales = np.array([self.concentration_scale] * dimension + list(self.process_scales[:X.shape[1] - 2 * dimension]))
        conditions = np.exp(-.5 * cdist(X[:, dimension:] / scales, Y[:, dimension:] / scales, "sqeuclidean"))
        covariance = conditions * (self.weights[0] * shared + self.weights[1] * pairs + self.weights[2] * exact)
        if eval_gradient:
            return covariance, np.empty((*covariance.shape, 0))
        return covariance

    def diag(self, X):
        count = np.asarray(X)[:, :len(ELEMENTS)].sum(axis=1)
        return self.weights[0] + self.weights[2] + self.weights[1] * (count > 1)

    def is_stationary(self):
        return False


def transfer_classifier():
    return GaussianProcessClassifier(CompositionTransferKernel(), optimizer=None, random_state=42, max_iter_predict=100)


def process_key(sample):
    return cohort_key(sample)[1:]


def validation_metrics(y, predictions, groups, kind):
    valid = np.isfinite(predictions)
    if not valid.any() or len(set(y[valid])) != 2:
        return None
    return {"type": kind, "n": int(valid.sum()), "total_n": len(y), "full_coverage": bool(valid.all()),
            "n_groups": len(set(groups[valid])), "balanced_accuracy": float(balanced_accuracy_score(y[valid], predictions[valid] >= .5)),
            "brier": float(brier_score_loss(y[valid], predictions[valid])), "false_pass_count": int(np.sum((predictions[valid] >= .5) & (y[valid] == 0)))}


def fit_transfer_model(samples, methods, context, origin):
    from .planning import ModelState, independent_groups, fingerprint
    chosen, exclusions = [], Counter()
    for sample in samples:
        reason = None
        if sample.get("origin") != origin:
            reason = "Different data namespace"
        elif process_key(sample) != process_key(context):
            reason = "Different precursor, support, protocol or temperature-control basis"
        elif any(str(sample.get(key) or "UNKNOWN").strip().upper() in {"UNKNOWN", "UNASSIGNED", ""} or str(sample.get(key)).strip().upper().startswith(("REPLACE_", "FILL_IN_")) for key in ("precursor_family", "support_batch", "protocol_id", "batch_id")):
            reason = "Unresolved process metadata"
        elif sample.get("pulse_count") is None or any(sample["concentrations_mM"].get(e) is None for e in sample["elements"]) or (sample.get("pulse_limit_C") is None and sample.get("temperature_C") is None):
            reason = "Incomplete synthesis conditions"
        else:
            result = screen(sample, methods)
            if result["review_conflicts"]:
                reason = "Conflicting reported and numeric judgments"
            elif result["label"] is None:
                reason = "Incomplete ICP/XRD screening"
        if reason:
            exclusions[reason] += 1
        else:
            chosen.append(sample)
    labels = np.array([screen(s, methods)["label"] for s in chosen], dtype=int)
    count = Counter(labels)
    target = composition_key(context)
    occurrences = Counter(e for s in chosen for e in s["elements"])
    compositions = Counter(composition_name(s) for s in chosen)
    stages = Counter(len(s["elements"]) for s in chosen)
    seen_pairs = {tuple(sorted(pair, key=ELEMENTS.index)) for s in chosen for pair in combinations(s["elements"], 2)}
    target_pairs = set(combinations(target, 2))
    target_n = sum(composition_key(s) == target for s in chosen)
    batches = len({s["batch_id"] for s in chosen})
    min_batches = 8 if origin == "REAL" else 3
    info = {"mode": "TRANSFER_BRIDGE_DESIGN", "origin": origin, "feature_kind": "COMPOSITION_TRANSFER", "model_version": MODEL_VERSION,
            "n_labeled": len(chosen), "n_batches": batches, "n_recipes": len({recipe_key(s) for s in chosen}),
            "n_pass": count[1], "n_fail": count[0], "minimum_batches": min_batches, "validation": None,
            "transfer_validation": None, "expansion_validation": None, "source_elements": dict(occurrences),
            "source_compositions": dict(compositions), "source_stages": dict(stages), "target_elements": list(target),
            "target_element": "+".join(target), "target_labeled_n": target_n, "unseen_target": target_n == 0,
            "unseen_elements": [e for e in target if e not in occurrences], "unobserved_pair_count": len(target_pairs - seen_pairs),
            "target_pair_count": len(target_pairs), "highest_source_stage": max(stages, default=0), "exclusions": dict(exclusions),
            "successful_source_ids": [s["sample_id"] for s, label in zip(chosen, labels) if label == 1],
            "features": ["element_membership", "masked_log_concentrations_mM", "total_concentration_mM", "element_count", "pulse_count"] + (["controlled_pulse_limit_C"] if context.get("pulse_limit_C") is not None else []),
            "kernel_components": {"shared_members": .45, "pair_residual": .20, "exact_composition_residual": .35},
            "screening_rule_version": RULE_VERSION, "stem_role": "Optional confirmation only; not a feature or training label",
            "reported_label_count": sum(screen(s, methods)["icp_basis"] == "USER_REPORTED" or screen(s, methods)["xrd_basis"] == "USER_REPORTED" for s in chosen),
            "probability_note": "Uncalibrated composition-transfer hypotheses. Unmeasured interactions retain a prior; singles cannot establish mixture success.",
            "transfer_validation_note": "No held-out-composition check is available with fewer than two eligible compositions.",
            "expansion_validation_note": "Expansion validation needs measured higher-element compositions and valid lower-stage training folds."}
    class_recipes = {label: len({recipe_key(s) for s, y in zip(chosen, labels) if y == label}) for label in (0, 1)}
    if len(chosen) < 12 or batches < min_batches or min(class_recipes.values()) < 3:
        info["reason"] = f"Source-guided bridge design: {len(chosen)} resolved records across {len(compositions)} compositions. A classifier requires 12 records, {min_batches} synthesis batches and 3 distinct recipes per class. No pass probability is assigned before this gate."
        return ModelState(None, info, chosen)
    x = np.array([transfer_feature(s) for s in chosen])
    groups = independent_groups(chosen)
    if len(set(groups)) >= 3:
        predictions = np.full(len(chosen), np.nan)
        for train, test in GroupKFold(n_splits=min(3, len(set(groups)))).split(x, labels, groups):
            if len(set(labels[train])) == 2:
                predictions[test] = transfer_classifier().fit(x[train], labels[train]).predict_proba(x[test])[:, 1]
        info["validation"] = validation_metrics(labels, predictions, groups, "Connected batch/recipe CV within the source pool; not expansion validation")
    else:
        info["validation_note"] = "Fewer than three connected validation groups; no grouped metrics available."
    names = np.array([composition_name(s) for s in chosen])
    if len(compositions) >= 2:
        predictions = np.full(len(chosen), np.nan)
        for name in sorted(compositions)[:6]:
            test = np.flatnonzero(names == name)
            train = np.flatnonzero((names != name) & ~np.isin(groups, groups[test]))
            if len(train) >= 6 and len(set(labels[train])) == 2:
                predictions[test] = transfer_classifier().fit(x[train], labels[train]).predict_proba(x[test])[:, 1]
        info["transfer_validation"] = validation_metrics(labels, predictions, groups, "Held-out composition; training purges connected batches and repeated recipes")
        info["transfer_validation_note"] = "Up to six deterministic composition folds; skipped/partial coverage is reported. This does not establish future-composition reliability."
    sizes = np.array([len(s["elements"]) for s in chosen])
    if max(stages) > min(stages):
        predictions = np.full(len(chosen), np.nan)
        for stage in sorted(stages)[1:7]:
            test = np.flatnonzero(sizes == stage)
            train = np.flatnonzero((sizes < stage) & ~np.isin(groups, groups[test]))
            if len(train) >= 6 and len(set(labels[train])) == 2:
                predictions[test] = transfer_classifier().fit(x[train], labels[train]).predict_proba(x[test])[:, 1]
        info["expansion_validation"] = validation_metrics(labels, predictions, groups, "Lower-stage to higher-stage check; no equal/higher-stage or connected-batch training")
        info["expansion_validation_note"] = "Up to six higher stages evaluated from lower stages only; coverage is retrospective and not a chronological deployment test."
    model = transfer_classifier().fit(x, labels)
    info["mode"] = "COMPOSITION_TRANSFER_GPC"
    info["reason"] = f"Composition-transfer GPC: {len(chosen)} source records across stages {sorted(stages)}; {target_n} exact-target records. New combinations and expansion stages remain transfer hypotheses."
    info["model_id"] = fingerprint({"samples": chosen, "labels": labels.tolist(), "methods": methods, "version": MODEL_VERSION, "rule": RULE_VERSION})
    return ModelState(model, info, chosen)


def bridge_recipe(state, context, bounds):
    """Combine source-informed concentrations as an unmeasured experiment proposal."""
    successful = [s for s in state.training if s["sample_id"] in state.info.get("successful_source_ids", [])]
    sources, concentrations = {}, {}
    for element in context["elements"]:
        candidates = [(s, e) for s in successful for e in s["elements"]]
        if candidates:
            sample, symbol = min(candidates, key=lambda item: (item[1] != element,
                float(np.sum((np.array(descriptors(item[1])) - descriptors(element)) ** 2)),
                abs(len(item[0]["elements"]) - len(context["elements"])), item[0]["sample_id"]))
            concentration = sample["concentrations_mM"][symbol]
            sources[element] = {"sample_id": sample["sample_id"], "source_element": symbol, "source_composition": composition_name(sample), "source_precursor_mM": concentration,
                                "adapted_to_bounds": not bounds[element]["min_mM"] <= concentration <= bounds[element]["max_mM"]}
        else:
            concentration = context["concentrations_mM"][element]
        concentrations[element] = min(bounds[element]["max_mM"], max(bounds[element]["min_mM"], concentration))
    selected = [s for s in successful if s["sample_id"] in {r["sample_id"] for r in sources.values()}]
    pulse = float(np.median([s["pulse_count"] for s in selected])) if selected else context["pulse_count"]
    cutoff = float(np.median([s["pulse_limit_C"] for s in selected])) if selected and context.get("pulse_limit_C") is not None else context.get("pulse_limit_C")
    return {**context, "concentrations_mM": concentrations, "pulse_count": pulse, "pulse_limit_C": cutoff}, sources


def source_evidence(state, candidate, limit=3):
    if not state.training:
        return {"source_sample_ids": [], "source_elements": [], "source_compositions": [], "source_affinity": None, "support_level": "NO_SOURCE_DATA"}
    x = np.array([transfer_feature(s) for s in state.training])
    query = np.array([transfer_feature(candidate)])
    kernel = CompositionTransferKernel()
    affinity = kernel(query, x)[0] / np.sqrt(kernel.diag(query)[0] * kernel.diag(x))
    order = np.argsort(-affinity, kind="stable")[:limit]
    maximum = float(np.clip(affinity[order[0]], 0, 1))
    return {"source_sample_ids": [state.training[i]["sample_id"] for i in order],
            "source_elements": list(dict.fromkeys(e for i in order for e in state.training[i]["elements"])),
            "source_compositions": list(dict.fromkeys(composition_name(state.training[i]) for i in order)),
            "source_affinity": maximum, "support_level": "LOW_KERNEL_SUPPORT" if maximum < .1 else "SOURCE_SUPPORTED_PRIOR"}


def input_extrapolation(training, candidate):
    """Check recorded member concentrations and global settings without absence imputation."""
    if not training:
        return False
    for element in candidate["elements"]:
        observed = [s["concentrations_mM"][element] for s in training if element in s["elements"]]
        if observed and not min(observed) <= candidate["concentrations_mM"][element] <= max(observed):
            return True
    for getter in (lambda s: sum(s["concentrations_mM"].values()), lambda s: s["pulse_count"]):
        values = [getter(s) for s in training]
        if not min(values) <= getter(candidate) <= max(values):
            return True
    if candidate.get("pulse_limit_C") is not None:
        values = [s["pulse_limit_C"] for s in training]
        if not min(values) <= candidate["pulse_limit_C"] <= max(values):
            return True
    return False
