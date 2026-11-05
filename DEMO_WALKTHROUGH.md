# Connected discovery demonstration

Use Simulated demo / DEMO. All acquisitions and fixture outcomes below are synthetic.
An external API is unnecessary for the local commands. Natural language requests
use Research conversation and the configured provider/sharing permissions.

## Preparation and a human checkpoint

In Experiment chat, enter these commands one at a time:

```text
/workflow preview precursor_preparation {"pan_mass_g":2,"solvent_volume_mL":20,"voltage_V":32,"current_A":80}
/workflow edit {"pan_mass_g":3,"solvent_volume_mL":40}
/workflow create
/workflow start
```

Open Live board. Run remaining steps. The weighing checkpoint reserves the
assigned balance and waits. For a software demonstration only, enter a clearly
marked DEMO completion note and acknowledge it. Execute next step to inspect
the dispenser reservation. Cancel remaining releases the instrument. Real
laboratory actions require actual completion before acknowledgement.

Use Workflow library to browse 14 starter templates and Devices to search the
103 equipment profiles. An inventory entry is a planning resource, not a driver.

## Complete simulated learning loop

Return to Experiment chat:

```text
/workflow preview simulation_loop {"elements":["Fe"],"batch_size":32,"epochs":10,"model":"Fixed prior","register_resources":true,"pan_mass_g":2,"solvent_volume_mL":20,"voltage_V":32,"current_A":80}
/workflow create
/workflow start
```

Open Live board and run remaining steps. The 16-step graph freezes proposals,
uses separate ICP/XRD/STEM/SEM simulators, registers raw exports, reviews fixture
truth, evaluates forecasts, validates and trains a candidate, then designs again.
Watch the recorded progress. Training time depends on the machine and source size.

After completion, open Specialist coordination & results. Inspect the six roles,
Data handoffs and Recorded results. Select next_design to see the new proposals.
The candidate identifier is the model used by that final design. Inspect validation
opens the saved workflow report in Scientific tools. Inspect originating workflow
run links back. Continue in conversation returns the same run to chat.

The film supplied with this release records these application interactions in an
isolated synthetic workspace. Editing shortens some transitions and execution
segments. Offline synthetic narration and original generative background audio
are added in post-production. The film is software integration evidence; it does
not establish synthesis yield, instrument connectivity or scientific superiority.
