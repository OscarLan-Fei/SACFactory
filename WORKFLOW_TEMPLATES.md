# Discovery and workflow templates

SACFactory 0.20 combines Discovery lab and Devices & Workflows under one sidebar
entry: **Discovery & Workflows**. The workspace has five views:

| View | Purpose |
| --- | --- |
| Live board | Inspect dependencies, assigned instruments, manual reservations and scheduling |
| Workflow library | Configure starter patterns, save editable copies and import existing definitions |
| Scientific tools | Run individual prediction, knowledge, inverse-design, characterization, training and validation tasks |
| Runs | Inspect immutable run specifications, outputs, recovery and operator acknowledgements |
| Devices | Search 103 equipment profiles and manage planning or configured instruments |

The shared source selector defaults to observed elements and all matching records.
It can be narrowed to specific elements and records without a count cap. Design
targets are selected separately in template parameters. Model checkpoints,
campaign rounds, pending specimens, reviewed evidence and the discovery ledger
are shared. Opening a view or template preview does not fit a model or start a job.

## Built-in patterns

| ID | Pattern | Execution |
| --- | --- | --- |
| simulation_loop | Complete simulated SAC-Gen loop | DEMO fixture; 16 steps across all seven science functions |
| simulated_screening | Simulated synthesis and characterization | DEMO fixture; independent ICP, XRD, STEM and SEM without training |
| inverse_screening | Recipe design to Pending | Scientific design and immutable forecast registration; operator review |
| multielement_bridge | Multielement bridge experiments | At least two target elements; explicit transfer review |
| data_benchmark | Selected-data validation and comparison | Protected recipe-group validation, then a separate prediction diagnostic |
| candidate_update | Validate and train a candidate | Validation, operator review, versioned candidate and prediction comparison |
| precursor_preparation | Precursor preparation and electrospinning | Operator weighing, dispensing, mixing and fiber preparation |
| manual_synthesis | Operator-led pulsed synthesis | Design, Pending, manual preparation and synthesis |
| reviewed_characterization | Review registered characterization evidence | Existing artifacts, operator evidence review and supported assessment |
| characterization_panel | Extended characterization panel | Operator ICP, XRD, STEM, SEM, XPS, XAS, Raman and BET |
| electrochemical_performance | Electrochemical catalyst evaluation | Method review, electrode preparation, electrochemistry and product analysis |
| gas_catalysis | Gas-phase catalyst evaluation | Method review, reactor run, product analysis and material-balance review |
| real_feedback_loop | Operator-assisted synthesis feedback loop | Frozen design, operator synthesis and acquisition, reviewed evidence and forecast evaluation |

Eleven patterns are available in REAL; the two synthetic acquisition patterns
are DEMO-only. These patterns reuse the existing SACFactory loop and generic
operator task sequences. They do not prescribe validated instrument operating
conditions. Physical steps refer to the laboratory's approved method or SOP.

## Configure and use a copy

1. Choose a template under Workflow library. Filter by category or search text.
2. Open Template parameters. Configure target elements, feed bounds in mmol,
   temperature, pulse counts, prediction model and relevant training settings.
3. Use the shared source selection for data-based tasks. Evidence review also
   requires registered artifacts belonging to those selected specimens.
4. Bind equipment roles to registered instruments. Alternatively, enable resource
   registration to create missing planning inventory or the supported DEMO fixtures.
5. Inspect the step table and optional relationship preview. Save an editable copy
   with Create workflow from template. The copy opens on Live board; no run starts.
6. Start the saved workflow when ready, then execute a step or remaining ready tasks.

Copies remain independent. Changing parameters or saving another template does
not rewrite a previous definition or an active run. Export a configured workflow
as JSON, or import an existing exported definition through the library. Imported
steps and references pass the same allowlisted schema checks as locally authored
workflows; importing never executes code or starts a run.

## Operator tasks and equipment reservations

An operator checkpoint records an instruction, specimen scope and optional device
assignment. It can use planning inventory without a connected driver. Starting
the step reserves the assigned instrument and stops at WAITING. Another run using
that instrument remains queued. A documented acknowledgement or explicit run
cancellation releases the reservation. Pause retains it. Interrupted dispatches
require explicit recovery; waiting physical work is not silently released by a timer.

Recorded elapsed time for an operator checkpoint includes its waiting period and
ends at acknowledgement. Planning durations remain editable estimates. Operator
notes never generate ICP assays, structural labels or performance measurements.
Enter measurements through Samples & data or Characterization. Assessment commits
still require reviewed evidence. Other instrument types need their own adapters
before automatic acquisition; the current fixtures cover pulsed synthesis and
ICP/XRD/STEM/SEM in DEMO.

## Shared science and history

Scientific tools retains the original individual tools. Open single-step tool
from a matching template, or Use this task in a workflow from Scientific tools to
move between individual operations and an orchestrated sequence. Both call the
same scientific implementations. Workflow training and validation also write
idempotent records into the shared discovery history, with their originating run
IDs. Candidates do not automatically replace the primary Next batch deployment.

Model comparison on observed recipes can be in-sample. Protected validation uses
separate recipe-group masks and retains its own interpretation. Neither a template
nor a larger model establishes superior experimental performance by itself.

## Conversation and local commands

The capabilities tool lists available templates, roles and options. The existing
workflow_control tool supports templates and create_template. The latter accepts
spec_json containing template_id and an options object, then saves a definition.
Starting and executing remain separate actions. Numerical results come from the
scientific service; the language model does not invent measurement receipts.

```text
/workflow templates
/workflow template precursor_preparation
/workflow template simulation_loop
/workflow start <workflow_id>
/workflow step <run_id>
/workflow run <run_id>
/workflow status <run_id>
```

Existing local commands and saved workflow/run records remain compatible.
Historical sidebar selections route to the combined workspace.

Conversation editing, preparation settings and device selection use the same definitions.
See [Conversational workflows](CONVERSATION_WORKFLOWS.md). Live board revisions
retain template provenance and scheduling for further conversation edits.
