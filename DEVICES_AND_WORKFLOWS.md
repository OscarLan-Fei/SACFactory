# Devices and conversational workflows

SACFactory 0.20 connects the seven Discovery lab functions through one persistent
workflow service. Experiment chat, LLM tool calls and Discovery & Workflows use
that same service. Numerical forecasts come from scientific functions.

## Run the simulator

1. Select **Simulated demo · DEMO**.
2. Open **Discovery & Workflows > Workflow library**.
3. Choose **Complete simulated SAC-Gen loop** in the template library and click
   **Create workflow from template**. The saved definition contains 16 steps; running it creates 32 specimens
   and uses five simulator contracts: a pulsed heater plus independent ICP, XRD, STEM
   and SEM instruments. Review its 16 steps and dependencies.
4. Select the saved workflow and click **Start saved workflow**.
5. Open **Live board** or **Runs**, select the run, and execute one step or all remaining ready steps.
6. Inspect states and recorded outputs. Export the definition or complete run record.

The demonstration executes:

`Knowledge → Design → Frozen Pending → Simulated synthesis → Independent ICP + XRD + STEM + SEM
→ Feature extraction → Fixture review → Forecast evaluation → Protected validation
→ Set-model update → Model comparison → Hypothesis registration → Next design`.

The last design uses the new candidate. The first uses the fixed prior. Protected
validation fits separate fold-local models. The later observed-recipe comparison
is a diagnostic and can be in-sample.

The simulator is an integration fixture. ICP is generated from a perturbed
physical prior; structural outcomes are prescribed fixture truth. TIFF metadata,
diffraction headers, assays, sample metadata, device receipts and reviews identify
them as synthetic. Fixture review exercises the feedback route; it is not learned
recognition and cannot process REAL evidence. This run does not establish real
recognition accuracy or prospective discovery improvement.

## Conversation

With the existing provider and research-sharing settings enabled, Research
conversation exposes `workflow_capabilities` and `workflow_control`. The first
describes operations and device contracts. The second saves definitions, starts
and executes runs, reads status, pauses, resumes, retries or cancels work.

For example: “Build a workflow that searches the research memory, generates six
Fe–Co recipes, freezes them into Pending, and waits for me to perform synthesis.”
The assistant can save that definition. A follow-up request can execute it.
Review real measurements under Characterization before using a workflow to
commit their assessment.

Local commands need no API:

```text
/workflow capabilities
/workflow demo
/workflow list
/workflow start <workflow_id>
/workflow run <run_id>
/workflow status <run_id>
/workflow step <run_id>
/workflow pause <run_id>
/workflow resume <run_id>
/workflow retry <run_id>
/workflow cancel <run_id>
```

API tools retain the conversation's selected-record scope. Specimens created by
the run are available downstream. Notes and raw files are not automatically
shared with the provider. The workspace uses its explicitly selected source
records. API integration is tested with mocked transports; simulator execution
needs no live provider request.

## Definitions and references

The collapsed definition editor saves a new immutable definition. A run freezes
its own copy, source selection and fingerprint. A minimal definition is:

```json
{
  "name": "Research memory and manual review",
  "version": 1,
  "steps": [
    {"id": "memory", "operation": "knowledge.search",
     "parameters": {"query": "single atom retention and aggregation"}},
    {"id": "review", "operation": "manual.checkpoint", "depends_on": ["memory"],
     "parameters": {"instruction": "Review the sources before continuing."}}
  ]
}
```

Reference an earlier dependency with `{"$ref":"design.design"}` or
`{"$ref":"pending.sample_ids.0"}`. References resolve as data, without expression
or code evaluation. Dependencies must precede consumers. Execution follows a
linear or branched dependency graph. Ready tasks are ordered by resource
availability, priority and remaining critical path; definition order breaks ties.
Conditional routers, unbounded loops and autonomous workflow rewrites are not
implemented. Repeat a workflow as a new run for another experiment cycle.

| Discovery function | Operations |
| --- | --- |
| Campaign feedback | `campaign.plan`, `campaign.evaluate` |
| Research memory | `knowledge.search` |
| Disagreement and calibrated validation | `model.compare`, `validation.run` |
| Set and other candidate updates | `model.train` |
| Inverse search | `inverse.design` |
| Controlled hypotheses | `hypothesis.register` |
| Characterization | `characterization.analyze`, `characterization.commit` |

Additional operations are `device.synthesize`, `device.characterize`,
`manual.checkpoint` and DEMO-only `characterization.fixture_review`.
The capabilities tool lists required and optional parameters. Feed bounds are
mmol per precursor, ICP targets are numerical wt%, cutoff uses degrees C and
solvent uses mL. Default preparation remains 2 g PAN, 20 mL DMF, 32 V and 80 A.

Characterization analysis accepts registered artifact IDs or a verified simulator
receipt. It cannot open a path supplied by LLM text. REAL assessments require
reviewed observations. Campaign planning freezes only forecasts produced by an
inverse-design step in the same run. Candidate training does not automatically
replace the primary Next batch deployment.

## Devices

A device contract records ID, name, capabilities, limits, notes and connection
mode. Simulators accept specimen IDs and recipes, returning synthesis receipts
or measurement exports. Custom simulator IDs can be used in custom workflows.
Locks prevent concurrent device ownership; job IDs and completed receipts
prevent duplicate completed effects.

The equipment library contains 103 extensible profiles in four categories:
Synthesis equipment, Characterization equipment, Performance testing and General
equipment. It includes furnaces and deposition systems, spectroscopy and microscopy,
electrochemical and catalytic test stations, product analyzers, sample preparation,
transfer, utilities and computing. Search and category filters apply without a
count cap. General equipment is one expandable collection on the graph. Any
category can be collapsed, and each registered instrument has its own contract.

The catalog is a library of types, not an inventory of instruments owned by a
laboratory. Add displayed types as INVENTORY_ONLY planning entries, or register
one instrument with a custom ID, category, capabilities, modalities and notes.
Inventory-only devices cannot execute jobs. SIMULATOR mode is restricted to the
implemented pulsed-heater and ICP/XRD/STEM/SEM fixtures. Performance equipment and
other measurement types can be planned and registered but need their own adapters
before acquisition. No performance values are generated for unimplemented equipment.

New simulated loops use four independent measurement steps. Each instrument
produces only its declared modality and retains a separate job ID and device lock.
The analysis step accepts an acquisitions list, verifies each completed receipt,
joins measurements of the same specimens and synthesis, and rejects duplicate or
conflicting results before committing feedback. XRD tasks cannot be rebound to an
ICP instrument. Historical combined-characterizer runs retain their frozen contracts;
the combined fixture is hidden from new device-network views unless a run uses it.

Future MCP endpoints can be recorded as `MCP_NOT_CONNECTED`. Their actions remain
unavailable. Network transport, manufacturer drivers, real actuation and automatic
MCP-server installation are reserved for the next integration stage.

## Live relationship board and scheduling

The Live board displays registered devices and workflow steps with separate
dependency and device-assignment links. Switch between workflow lanes and a
mind-map layout. Pan, zoom, drag nodes and click a node to inspect its current
state. Browser-local layout positions persist across status refreshes. The board
and scheduling list refresh every two seconds without refitting models.

Equipment catalog graph scope also exposes all library profiles under the four
collections. Click a collection to expand it. Click a catalog type to open its
registration form. Catalog profiles are not draggable execution resources; only
registered compatible devices can be bound to a task.

Drop a registered device onto a compatible task to stage a binding. A keyboard-
accessible binding selector provides the same action. Draft bindings use a
separate dashed link. Saving creates a revised definition; the current frozen
run retains its original bindings. The register-device button opens the device
registry, and registered devices appear in the network automatically.

The scheduling queue checks completed dependencies, device registration,
capability, connection mode and occupancy. Ready work is ordered by priority,
then estimated remaining critical path. Priorities and estimated durations can
be saved in revised workflow definitions. An all-active-runs view can dispatch
the next ready task. A busy or unavailable device produces QUEUED status rather
than claiming the task or marking a scientific failure. Once it becomes
available, continue the queued run from the interface or conversation.

The timeline is a conservative, single-worker estimate. Durations are editable
planning inputs, not measured instrument acquisition times. Recorded/elapsed
execution time is displayed separately. This is a resource-aware heuristic, not
a claim of globally optimal scheduling or concurrent physical execution.

Source selection defaults to all matching records without rendering every
sample as a tag. Expand Source data selection to narrow elements or choose
individual records; no sample-count cap is introduced.

## State and recovery

Runs use READY, RUNNING, QUEUED, WAITING, PAUSED, FAILED, COMPLETED and CANCELLED states.
Manual checkpoints wait for a documented completion note. Pause/cancel stop
subsequent steps; a running function finishes first. Interface run controls start
a bounded background worker so the status graph stays responsive. One worker
advances a data space at a time. It ends at completion, failure, a manual
checkpoint or a blocked resource. The conversation can also advance a run;
there is no indefinitely polling unattended scheduler.

Claims are transactional, preventing a second worker from executing the same
active step. Retry is explicit and completed operations return receipts. A
completed candidate checkpoint can be reloaded if a worker stopped before its
receipt was published. Partial checkpoints require review and are not overwritten
automatically. Training-source changes invalidate interrupted training recovery.

After a 30-minute worker lease expires, reconcile the interrupted worker and
then retry. Expired simulator locks can be reconciled in that explicit path.
Future physical devices will need instrument-specific status reconciliation.

Definitions, runs, events, steps, jobs, locks and receipts use
`data/workflows_real.sqlite3` or `data/workflows_demo.sqlite3`. Existing campaign
and characterization stores retain their evidence. Specimen identities and
recipe hashes are checked at acquisition/feedback boundaries. Changed source
IDs fail explicitly rather than selecting another specimen.

Keep `data/` when updating an installation. Source archives exclude experiment
databases, simulation workspaces, credentials, private files and locally trained
weights. The existing public pretrained release is a separate packaged asset.
