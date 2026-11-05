# Conversational workflows and Live board

SACFactory 0.21 connects workflow selection, parameter edits, instrument contracts,
versioned definitions and recorded execution to Experiment chat. Natural-language
control uses Research conversation with the existing provider and explicit sharing
settings. Local workflow commands use the same controller without an API request.

## Natural-language operation

For a complete material campaign, choose **Confirmed material synthesis and learning loop**.
It requests explicit plan approval before dispatch, records preparation and performance
returns, and joins eligible historical and new records for a versioned model update.
See [Confirmed synthesis workflows](CONFIRMED_SYNTHESIS_WORKFLOWS.md).

Examples:

- "Show the workflow templates and preview an operator-assisted synthesis loop."
- "Use Fe and Co, propose six recipes, keep 2 g PAN and 20 mL DMF, and use four or eight pulses."
- "Change Fe to 50 mM in 40 mL DMF and leave Co unchanged."
- "List my registered ICP instruments and bind an ICP-MS planning contract to the ICP role."
- "Save this preview, start its run, then execute ready steps in the background."
- "What is running now? Pause the remaining work."
- "I completed weighing and recorded the actual masses. Acknowledge that checkpoint."

The language model calls workflow_template, workflow_devices and workflow_control.
Parameters and bindings persist across turns within the same discussion and
permitted source scope. Tool outputs supply definition and run IDs. The Current
workflow card shows the preview or saved definition, recorded progress and operator
waiting instructions. Open Live board selects the matching preview, definition or
run and carries over the conversation source selection.

Creation does not start a run. Starting creates a READY run. Explicit execution
requests launch a bounded background worker. Manual checkpoints require a documented
operator acknowledgement. Acknowledgement does not create assays or outcome labels.

## Parameters and immutable changes

Options include element-specific bounds_mmol, temperature, pulses, models, epochs,
folds, bindings by equipment role and scheduling by existing task ID. For example,
`{"scheduling":{"weigh":{"priority":80,"estimated_seconds":90}}}` changes an
estimate and queue priority, not measured execution time. Preparation settings include pan_mass_g,
solvent_volume_mL, voltage_V, current_A and carbonization_loss_fraction. They are
stored with the template and copied to the run context. Later UI context changes
do not silently change the saved preparation basis.

The conversation-only bounds_mM alias uses mmol = mM × mL / 1000. Thus 50 mM in
20 mL is 1 mmol; 50 mM in 40 mL is 2 mmol. Partial element edits preserve other
bounds. One edit cannot specify both bounds_mM and bounds_mmol.

Preview edits remain unsaved. Saving or revising creates a new immutable definition.
Existing runs retain their controls, receipts and observations. New versions do not
start automatically. Older copies with a template audit event can recover parameter
mappings without rewriting their definitions. Untracked imported workflows can be
inspected and copied through the definition editor.

## Devices and recorded state

workflow_devices lists registered instruments or searches the 103-type catalog.
It can register a requested inventory or simulator contract. Registration sends
no instrument command. Binding checks capability, modality and connection mode;
an ICP instrument cannot replace XRD. Occupied instruments cannot be edited.
Real MCP actuation remains unimplemented.

Simulator acquisition is DEMO-only. REAL physical workflows use operator checkpoints
with planning or disconnected contracts. Equipment stays reserved while a manual
checkpoint waits; acknowledgement or cancellation releases it. Progress and waiting
reasons come from persisted records rather than language-model text.

## Offline commands

```text
/workflow templates
/workflow preview precursor_preparation
/workflow edit {"pan_mass_g":3,"solvent_volume_mL":40}
/workflow catalog ICP
/workflow register {"profile_id":"liquid_handler","id":"backup_dispenser","mode":"INVENTORY_ONLY"}
/workflow edit {"bindings":{"dispenser":"backup_dispenser"}}
/workflow create
/workflow start
/workflow run
/workflow status
/workflow ack weigh Actual masses recorded by the operator.
/workflow step
/workflow pause
/workflow resume
/workflow cancel
```

Omitted IDs refer to the current conversation workflow. Use explicit IDs for older
runs. `/workflow select <workflow_id>` selects a definition; `/workflow revise
{options}` saves an edited copy. `/workflow template <template_id>` still creates
a copy. Inspect actions and local errors preserve completed work.

## Scope and provider failures

State is separate for REAL and DEMO. A new discussion or narrowed sharing scope
resets the active conversation selection; saved definitions and runs remain.
Expanded selections can retain active work without changing its frozen specimen
scope. Workflow requests do not pretrain the ordinary planner. Its fit is deferred
until an actual planner preview requests it.

A provider failure after a tool action does not undo that local action. The card
retains its IDs and recorded status for subsequent inspection. No automatic retry
is sent. Integration tests use mocked transports. A live provider and endpoint
permissions are configured independently on Language model.

## Specialist coordination and shared results

Live board projects six bounded roles from the actual workflow: Knowledge, Design,
Execution, Characterization, Learning and Feedback. Each operation has one owner.
Successful operation receipts retain the specialist identity, and audit events
record the result fields. These are tool-based responsibilities orchestrated by
the workflow engine, not six independently running LLM sessions. One background
worker advances a data space; physical concurrency is not implied by the graph.

The coordination panel shows the specialist team, dependency handoffs and recorded
result summaries. Referenced input fields become available only after their source
task completes. A manual acknowledgement remains an operator record, not an assay.
Completed runs expose proposal tables, the candidate model identifier and exports.
The simulated complete loop verifies that the next design uses the candidate
registered by the update task.

Use Inspect evidence, Inspect validation, Inspect models or Inspect feedback to
open the corresponding Scientific tools view. These views share the selected
records, campaign ledger and model registry. Continue in conversation returns
the selected definition or run to chat without expanding API sharing permission.
If the run uses unshared records, chat explains the scope mismatch instead.

Changes made through Live board preserve template provenance, parameter values,
device-role bindings and scheduling. They save a new version and open its preview.
The old run keeps its original version. Subsequent conversation edits retain those
board settings rather than reconstructing a disconnected workflow.

Characterization review uses direct artifact and specimen lookup for unchanged
display IDs. Stable physical identity is still checked; compacted display IDs fall
back to an identity search. Evidence summaries select only that specimen's events
and recheck file hashes. Simulation imports aligned measurement updates as a batch
before registering individual raw exports. These optimizations preserve review and
mass-balance checks.
