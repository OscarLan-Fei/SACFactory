# Confirmed material synthesis and learning

The `confirmed_synthesis_loop` template connects a material goal to a reviewed,
durable laboratory plan. It is available through Research conversation and the
Workflow library. A configured language-model provider interprets unrestricted
natural-language requests; all numerical and workflow actions use local tools.

## Operator sequence

1. Describe the target elements, preparation basis and search bounds. Request an
   equimolar feed when appropriate. Feed equality is not evidence of an
   equilibrated high-entropy product or atomically dispersed structure.
2. Inspect the calculated recipes, proposed equipment assignments and schedule.
   The `confirm_plan` checkpoint blocks Pending creation and equipment dispatch.
   Explicit approval and a performance protocol are required. Editing historical
   source records during review invalidates approval of that plan.
3. Release the plan and record preparation, weighing, dispensing, mixing and
   electrospinning returns. Required fields are checked before each dependency
   completes. Operator acknowledgements do not automatically import ICP results
   or establish structural PASS/FAIL.
4. Execute heating and independent characterization. REAL equipment remains
   operator-led; MCP registration is a contract, not an active hardware driver.
   DEMO acquisitions use independently verified instrument fixtures.
5. Review evidence, record the approved performance test and attach its results
   and raw-data references. Performance remains a separate endpoint and is not
   treated as an ICP or structural label.
6. Assemble eligible selected historical records and current-group returns.
   Inspect exclusions, run recipe-group validation, train a versioned candidate,
   and generate the next group with that candidate's model ID. The next group
   remains a proposal and requires a new review before execution.

## Work cards and conversation tools

Conversation and Live board share work cards with inputs, dependencies, device
assignment, numerical model, local compute, nested calculation stages, execution
timestamps and returned data. Follow active task tracks the current checkpoint;
turn it off to inspect an earlier step. Large device branches receive separate
vertical space in the graph.

`workflow_control(action="step_result", spec_json={"step_id":"..."})` reads a
saved result within the conversation's selected data scope. For a reported
operator action, `acknowledge` accepts `step_id`, `note` and `returned_data`.
Required return fields are declared by the waiting checkpoint. Confirmation
requires `approved: true` and a nonempty `performance_protocol`.

The roles are bounded tool responsibilities, not independently launched LLM
agents. Execution metadata reports the local four-thread CPU worker used by this
workflow service; it does not imply GPU acceleration or instrument concurrency.
The primary deployed model and REAL records are unchanged by isolated rehearsal
runs or discovery-candidate training.
