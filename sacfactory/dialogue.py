"""Task-focused local dialogue; an external LLM never owns scientific labels."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Protocol
from .domain import STATUS_NAMES, screen, summary
from .planning import recommend, recommendation_table


class LanguageAdapter(Protocol):
    """Opt-in API: validated read-only intents, not arbitrary code.

    Allowed tools: summarize, diagnose, recommend, explain. Scientific engines
    remain authoritative. No SQL, file writes or changes to method limits.
    Credentials belong in the environment, never in experimental records.
    """
    def interpret(self, text: str, history: list, context: dict) -> dict: ...


@dataclass
class Answer:
    text: str
    recommendations: list
    table: list


def diagnose(sample, methods):
    result = screen(sample, methods)
    if result["status"] in {"BURNED_OUT", "RETEST"}:
        from .domain import SAMPLE_CONDITIONS
        meaning = "Observed complete burnout is a synthesis failure (label 0). Unperformed assays stay pending; no ICP/XRD measurement is invented." if result["status"] == "BURNED_OUT" else "This handling, device or sample-availability exception requires retesting and has no training label."
        return f"**{sample['sample_id']}: {STATUS_NAMES[result['status']]}**. Sample condition: **{SAMPLE_CONDITIONS[result['sample_condition']]}**.\n\n{meaning}\n\nRecorded ICP={result['icp']}, XRD={result['xrd']}. Optional STEM={result['stem']}."
    reasons = []
    fail = [r["element"] for r in result["elements"] if r["state"] == "FAIL"]
    missing = [r["element"] for r in result["elements"] if r["state"] == "PENDING"]
    if fail: reasons.append("Elements failing ICP: " + ", ".join(fail))
    if missing and result["icp_basis"] == "NUMERIC_CRITERIA": reasons.append(f"{len(missing)} targets lack complete results or confirmed method limits; choose a direct ICP judgment or add the numeric evidence later")
    if result["xrd"] == "PENDING": reasons.append("XRD judgment is pending; choose PASS/FAIL directly or add a reviewed criterion")
    elif result["xrd"] == "FAIL": reasons.append("XRD review reports a detectable relevant crystalline phase")
    weak = result["weakest"]
    weakest = f" Weakest measured element: {weak[0]}, criterion margin {weak[1]:.3g}." if weak else ""
    text = f"**{sample['sample_id']}: {STATUS_NAMES[result['status']]}**. ICP={result['icp']}, XRD={result['xrd']}.{weakest}"
    text += f"\n\nJudgment basis: ICP **{result['icp_basis']}**, XRD **{result['xrd_basis']}**."
    if result["icp_basis"] == "USER_REPORTED": text += f" Direct ICP judgment is saved independently; numeric evidence currently gives **{result['numeric_icp']}**."
    if result["review_conflicts"]: text += "\n\n" + "; ".join(result["review_conflicts"]) + ". Resolve the conflict before model training."
    text += f"\n\nOptional STEM confirmation: **{result['stem']}**. This is recorded separately from the ICP/XRD screening label."
    if result["stem"] == "NOT_CONFIRMED": text += " The reported STEM interpretation does not confirm atomic dispersion, even if candidate screening passes."
    if reasons: text += "\n\n" + "; ".join(reasons) + "."
    if fail or result["xrd"] == "FAIL":
        text += "\n\nThese observations identify the failing branch. ICP/XRD alone cannot distinguish volatilization, digestion loss or aggregation; controlled experiments are needed."
    text += "\n\nThis is candidate screening; it does not independently establish atomic dispersion."
    return text


def reply(text, samples, methods, state, context, space, origin, reservations=(), default_count=6):
    lower = text.lower().strip()
    if any(term in lower for term in ("api", "llm", "who are you", "language model")):
        return Answer("I am a **local task and calculation assistant**. I can summarize data, diagnose evidence and recommend experiments. Configure an optional OpenAI-compatible provider on **Language model**. Local mode sends no API requests; enabled API mode shares the displayed minimal context and your current request. Scientific calculations always run locally.", [], [])
    selected = next((s for s in sorted(samples, key=lambda x: len(x["sample_id"]), reverse=True) if s["sample_id"].lower() in lower), None)
    if selected: return Answer(diagnose(selected, methods), [], [])
    if any(term in lower for term in ("next batch", "recommend", "plan experiments", "select experiments")):
        match = re.search(r"(?:recommend|next batch|plan|select)\s*(\d+)", lower)
        number = int(match.group(1)) if match else default_count
        if not 1 <= number <= 12:
            return Answer("Supported batch size is 1 to 12. Please choose a batch size in that range.", [], [])
        chosen_space = dict(space)
        pulse = re.search(r"(?:only|just)\s*(?:use|do)?\s*(\d+)\s*(?:pulse|pulses)", lower)
        if pulse:
            n = int(pulse.group(1))
            if n not in space["pulses"]: return Answer(f"{n} pulses are outside the registered range. Update the validated range on the Next batch page first.", [], [])
            chosen_space["pulses"] = [n]
        if re.search(r"\d+\s*(?:mm\b|°c|celsius|elements?\b)", lower) or any(t in lower for t in ("change concentration", "change temperature", "add element", "change elements")):
            return Answer("Set concentration, temperature or composition changes on the Next batch page first. The local parser supports batch size and 'only N pulses'; it does not guess complex recipe edits.", [], [])
        try: rows, note = recommend(state, samples, context, chosen_space, origin, number, reservations)
        except ValueError as exc: return Answer(f"Cannot generate a batch yet: {exc}\n\nInspect evidence or import data first.", [], [])
        name = "Model-assisted recommendation" if state.model is not None else "Initial experimental design"
        temperature_note = f"controlled cutoff {context['pulse_limit_C']:g} °C or registered range" if context.get("pulse_limit_C") is not None else f"recorded temperature {context['temperature_C']:g} °C"
        answer = f"**{name}: {len(rows)} proposals.** Target: {len(context['elements'])} elements, {temperature_note}."
        answer += f"\n\nThis training pool has {state.info['n_labeled']} trainable records from {state.info['n_batches']} synthesis batches. {note}"
        answer += "\n\nPreview or export these recipes. 'Add batch to pending' creates unmeasured SAC records and linked tasks; 'Reserve batch' creates tasks only."
        return Answer(answer, rows, recommendation_table(rows))
    if any(term in lower for term in ("why", "fail", "diagnos", "check", "missing", "screen")):
        s = next((s for s in samples if s["sample_id"] == context.get("sample_id")), samples[-1] if samples else None)
        return Answer(diagnose(s, methods) if s else "No records yet. Register a sample or import the template, then ask about its ID.", [], [])
    if any(term in lower for term in ("bayes", "optimiz", "algorithm", "train", "predict", "learn", "model")):
        info = state.info
        engine_name = ('Recipe reliability ensemble with protected target/source weights and adaptive repeat budgets' if info.get('engine') == 'reliability' else
                       'Adaptive hierarchical Bernoulli GP with learned kernel parameters and joint information acquisition' if info.get('engine') == 'adaptive' else 'Gaussian-process classification and active learning')
        answer = f"The current engine uses **{engine_name}** for joint ICP/XRD screening. Inputs: {', '.join(info.get('features', ['element_concentrations_mM', 'pulse_count']))}. Composition, precursor, support and protocol define the cohort. Controlled cutoff is used when recorded; measured peak temperature, voltage, current and STEM are not predictor inputs.\n\nMode: {info['mode']}. {info['reason']}"
        if info.get("feature_kind") == "COMPOSITION_TRANSFER":
            answer += "\n\nComposition transfer pools single-element and mixture records under the same process and temperature-control basis. Shared-member, pair and exact-composition kernel components distinguish individual concentrations, total concentration and element count. Source stages: " + str(info["source_stages"]) + ". New mixtures retain unmeasured residuals and require bridge experiments. " + info["transfer_validation_note"] + " " + info["expansion_validation_note"]
        if info.get("validation"):
            v = info["validation"]
            answer += f"\n\nGrouped validation: balanced accuracy {v['balanced_accuracy']:.3f}, Brier score {v['brier']:.3f}. Retrospective metrics do not establish prospective reliability."
        if info.get('transfer_weights'):
            answer += '\n\nDevelopment weights: '+str(info['transfer_weights']['weights'])+'. The ensemble remains uncalibrated; these weights are not independent performance scores. Repeat counts use batch-balanced working precision and are reconsidered after returned results.'
        return Answer(answer, [], [])
    if any(term in lower for term in ("expand", "expansion", "50-element", "route", "1-element", "2-element")):
        return Answer("Expansion can share single-element data into explicit 2-, 3- or 50-element compositions under the same precursor/support/protocol and temperature-control basis. Individual concentrations, total concentration, element count and pair/composition residuals remain distinct. Select the full composition and enable transfer on Next batch. Sparse data produce source-guided bridge experiments without numeric probabilities; adequate eligible sources enable GPC transfer estimates. Mixture outcomes update the model. Successful singles do not become synthetic mixture labels. Inspect source stages, unseen members/pairs, kernel support and separate lower-to-higher-stage validation before interpreting predictions.", [], [])

    if any(term in lower for term in ("overview", "summary", "summar", "how many", "data", "result")):
        total = summary(samples, methods)
        counts = "; ".join(f"{STATUS_NAMES[k]}: {v}" for k, v in total["counts"].items()) or "No data"
        return Answer(f"The **{origin}** space contains {total['total']} samples: {counts}.\n\nThe selected {len(context['elements'])}-element cohort has {state.info['n_labeled']} trainable records. Complete method limits, XRD evidence and independent batches before fitting a model. STEM is optional confirmation. Try 'Check missing evidence' or 'Recommend 6 experiments'.", [], [])
    return Answer("Try: **Summarize results; Check missing evidence; Diagnose SAMPLE_ID; Recommend 6 experiments; Next batch only 1 pulse; Explain the model; Explain the expansion route**. Enable an optional provider on Language model for free-form request interpretation. Set recipe limits on the Next batch page.", [], [])


def execute_intent(intent, samples, methods, state, context, space, origin, reservations=(), default_count=6):
    """Execute validated language intents using the existing scientific engine."""
    from .language import shared_context, validate_intent
    intent = validate_intent(intent, shared_context(samples, methods, state, context, space, origin))
    scoped = [s for s in samples if s.get("origin") == origin]
    action = intent["intent"]
    if action == "clarify":
        return Answer("Please specify whether you want a results summary, diagnosis of a sample, a next batch, or a model/expansion explanation. Include a sample ID or batch size when relevant.", [], [])
    if action == "unsupported":
        return Answer("This request needs a supported tool or a settings change. Use Samples & data for records, Analytical methods for criteria, and Next batch for composition or operating limits. The assistant cannot change these settings or operate equipment through chat.", [], [])
    if action == "diagnose":
        sid = intent["sample_id"] or context.get("sample_id")
        selected = next((s for s in scoped if s["sample_id"] == sid), None)
        return Answer(diagnose(selected, methods) if selected else "Include a sample ID from the shared selection, or inspect its evidence on Samples & data.", [], [])
    if action == "recommend":
        count = intent["batch_size"] or default_count
        chosen_space = dict(space)
        if intent["pulse_count_filter"]:
            chosen_space["pulses"] = intent["pulse_count_filter"]
        return reply(f"Recommend {count} experiments", scoped, methods, state, context, chosen_space, origin, reservations, count)
    if action == "summarize":
        return reply("Summarize results", scoped, methods, state, context, space, origin, reservations)
    topic = intent["topic"] or "model"
    if topic == "screening":
        return Answer("Candidate screen pass requires ICP=PASS and XRD=PASS. You can enter these judgments directly without complete files or numeric results; their basis is USER_REPORTED. AUTO ICP uses registered limits and entered values. Pending judgments do not train the model; partial records can still be saved. STEM is optional and separate. Screening does not independently establish atomic dispersion.", [], [])
    return reply({"model": "Explain the model", "expansion": "Explain the expansion route", "api": "Explain API"}[topic], scoped, methods, state, context, space, origin, reservations)
