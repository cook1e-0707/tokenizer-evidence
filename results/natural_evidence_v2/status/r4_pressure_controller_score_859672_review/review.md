# R4 Pressure-Controller Score 859672 Review

Status: `FAIL_R4_PRESSURE_CONTROLLER_SCORE_859672_WRONG_CONTROLS_PASS_NO_GENERATION`

Job `859672` completed all 72 H200/pomplun controller-grid tasks with exit code `0:0`; the wrapper repair worked and produced 72 summary artifacts.

Summary-level gate results:

- Protected basic teacher-forced gate pass count: `72/72`
- Overall selective gate pass count: `0/72`
- Wrong-key controlled basic gate pass count: `72/72`
- Wrong-payload controlled basic gate pass count: `72/72`

Best protected lift grid: `grid_65` with bonus `1.5`, penalty `0.0`, max target mass `0.45`, max KL `0.1`. Its controlled-protected lift vs base is `0.397503`, lift vs task-only is `0.400662`, rank1 is `1.000000`, median margin is `0.439067`.

However, wrong-key and wrong-payload controlled arms also satisfy the same mass/rank criteria across the grid. Therefore this is not keyed-selective evidence. It remains a failed diagnostic for selectivity, even though it demonstrates that controller pressure can raise committed target mass.

Next allowed action: artifact-only wrong-control mapping/scorer semantics diagnosis. No generation, no new scoring submission, no training, no Llama, no FAR/sanitizer, and no paper-facing claim is unlocked.
