# Codex Execution Checklist

## 1. Purpose

This is the operational checklist for Codex in VS Code.

The repo is an Overleaf LaTeX repo. Codex must revise the manuscript using the guidance files in `docs/revision_guides/`.

Do not fabricate results. Do not write unsupported claims. Do not assume file names.

## 2. Initial Repository Inspection

Run:

```bash
pwd
find . -maxdepth 3 -type f \( -name "*.tex" -o -name "*.bib" -o -name "*.cls" -o -name "*.sty" \) | sort
```

Identify likely main file:

```bash
rg -n "\\documentclass|\\begin\\{document\\}|\\end\\{document\\}" .
```

Find sections:

```bash
rg -n "\\section\\{|\\section\\*\\{|\\subsection\\{|\\paragraph\\{|\\label\\{" .
```

Find abstract:

```bash
rg -n "\\begin\\{abstract\\}|\\end\\{abstract\\}" .
```

Find bibliography:

```bash
rg -n "\\bibliography\\{|\\addbibresource|\\printbibliography|\\begin\\{thebibliography\\}" .
```

Find packages:

```bash
rg -n "\\usepackage" .
```

## 3. Baseline Compile Before Editing

Try one of:

```bash
latexmk -pdf main.tex
```

If `main.tex` does not exist, identify main file first.

Alternative:

```bash
pdflatex <MAIN>.tex
bibtex <MAIN>
pdflatex <MAIN>.tex
pdflatex <MAIN>.tex
```

Record:
- compile success/failure;
- undefined refs;
- missing citations;
- overfull hboxes;
- underfull hboxes;
- page count.

Do not start major edits before knowing baseline compile status.

## 4. Create Revision Guides Directory If Needed

If these guide files are not already present:

```bash
mkdir -p docs/revision_guides
```

Save each markdown guide under `docs/revision_guides/`.

## 5. Phase-by-Phase Editing

### Phase 1: Claim-control search

Run:

```bash
rg -n "we show|we demonstrate|outperform|state-of-the-art|robust|secure|generalize|generalization|guarantee|superior|optimal|fully|universal|solve" .
rg -n "current artifacts|paper-facing|standing|landing|partial calibration|artifact-backed" .
```

For each risky claim:
1. Check support.
2. If unsupported, rewrite conservatively.
3. If result-dependent, mark `NEEDS_RESULTS`, `PLACEHOLDER`, or `TODO_AFTER_RESULTS`.

### Phase 2: Abstract and Introduction

Follow:
- `01_abstract_introduction_revision.md`

Tasks:
1. Replace abstract with conservative skeleton.
2. Add running example.
3. Add scope paragraph.
4. Rewrite contribution bullets.
5. Remove project-report language.

Compile after this phase.

### Phase 3: Problem Formalization

Follow:
- `02_problem_formalization_revision.md`

Tasks:
1. Add notation table.
2. Add assumptions.
3. Rewrite threat model to distinguish covered / motivating / outside scope.
4. Ensure FAR is calibration target, not conclusion.
5. Ensure theorem implications are bounded.

Compile after this phase.

### Phase 4: Method

Follow:
- `03_method_revision.md`

Tasks:
1. Add method overview.
2. Add module input/output descriptions.
3. Add Algorithm 1 and Algorithm 2 or table-box equivalents.
4. Ensure RS is described as standard coding layer.
5. Remove broad method guarantees.

Compile after this phase.

### Phase 5: Related Work

Follow:
- `04_related_work_revision.md`

Tasks:
1. Reorganize related work by evidence object / trainability / verifier / robustness.
2. Add or revise closest-work paragraph on Scalable/Perinucleus.
3. Mark unknown citation keys with `TODO_ADD_CITATION`.
4. Do not invent citation keys.

Compile after this phase.

### Phase 6: Evaluation

Follow:
- `05_evaluation_plan_revision.md`

Tasks:
1. Rewrite opening around RQs.
2. Add claim-evidence map.
3. Replace incomplete result claims with placeholders.
4. Ensure Perinucleus clean comparison is parity, not superiority.
5. Ensure FAR/utility claims are not final unless results complete.
6. Move detailed artifact ledgers to appendix if needed.

Compile after this phase.

### Phase 7: Limitations and Discussion

Follow:
- `06_limitations_discussion_revision.md`

Tasks:
1. Rewrite limitations as scope boundaries.
2. Add coverage table if space allows.
3. Ensure output deletion / delimiter destruction / probe-free distillation are explicit.
4. Align conclusion with limitations.

Compile after this phase.

### Phase 8: Figures, Tables, Algorithms

Follow:
- `08_figures_tables_algorithms_revision.md`

Tasks:
1. Revise Figure 1 caption.
2. Add notation, scope, claim-evidence, and related work tables as appropriate.
3. Add algorithm boxes.
4. Ensure all captions state scope.
5. Do not populate result tables without verified numbers.

Compile after this phase.

## 6. Citation and Bibliography Checks

Run:

```bash
rg -n "TODO_ADD_CITATION|\\cite\\{TODO|undefined citation|Citation.*undefined" .
```

Check `.bib` keys:

```bash
find . -name "*.bib" -print -exec sed -n '1,80p' {} \;
```

Do not create fake references.

If citation missing:
- leave `TODO_ADD_CITATION`;
- add note in final summary.

## 7. Label and Cross-Reference Checks

Run:

```bash
rg -n "\\label\\{|\\ref\\{|\\autoref\\{|\\cref\\{" .
rg -n "undefined references|Reference.*undefined|Label.*multiply defined" *.log */*.log 2>/dev/null || true
```

Do not rename labels unless necessary.
If adding labels:
- use clear names:
  - `tab:notation`
  - `tab:claim-evidence`
  - `alg:compile-train`
  - `alg:verify`
  - `sec:evaluation`
  - `sec:limitations`

## 8. Overfull / Underfull Check

After compile:

```bash
rg -n "Overfull|Underfull|undefined|Citation|Reference|multiply defined" *.log */*.log 2>/dev/null || true
```

Fix severe overfull boxes, especially in tables.

If tables are too wide:
- use `p{}` columns;
- reduce font to `\small`;
- move detailed table to appendix;
- do not delete important scope notes.

## 9. TODO Marker Check

Run:

```bash
rg -n "NEEDS_RESULTS|PLACEHOLDER|TODO_AFTER_RESULTS|TODO_ADD_CITATION" .
```

Expected:
- `NEEDS_RESULTS` should remain only where results are genuinely incomplete.
- No accidental unsupported claim should replace placeholders.

## 10. Final Modification Summary

After all edits, output a summary:

```markdown
## Revision Summary

### Files Modified
- ...

### Main Changes
- Abstract rewritten around next-token measurability.
- Introduction now includes running example and scope boundary.
- Problem setup now includes notation and assumptions.
- Method now includes algorithmic protocol.
- Related work repositioned around evidence object and verifier.
- Evaluation reorganized by research questions.
- Limitations rewritten as scope boundaries.
- Claim-control risky wording removed or marked.

### Remaining TODOs
- NEEDS_RESULTS: Full FAR.
- NEEDS_RESULTS: matched utility.
- NEEDS_RESULTS: bucket objective Pareto if not complete.
- NEEDS_RESULTS: E/S robustness phase diagram if not complete.
- TODO_ADD_CITATION: any unresolved citations.

### Compile Status
- Command:
- Success/failure:
- Undefined citations:
- Undefined references:
- Overfull boxes:
- Page count:
```

## 11. Codex Prompt

Use this short prompt if needed:

```markdown
你现在在 VS Code 中打开的是 Overleaf LaTeX repo。请按照 docs/revision_guides/ 中的 markdown 文件逐步修改论文。不要编造实验结果。不要写死尚未验证的 claim。优先修改 abstract、introduction、problem formulation、method overview、related work positioning、evaluation plan、limitations 和 claim control。修改后检查 LaTeX 编译、citation、label、cross-reference、overfull/underfull warnings，并输出修改总结。
```

## 12. Detailed Codex Execution Prompt

Copy-paste this into Codex:

```markdown
You are editing an Overleaf LaTeX repository in VS Code. Follow the revision guides under docs/revision_guides/ exactly.

Global constraints:
1. Do not fabricate experimental results.
2. Do not write unsupported claims.
3. Mark result-dependent content as NEEDS_RESULTS, PLACEHOLDER, or TODO_AFTER_RESULTS.
4. Do not assume file names; search for LaTeX sections, labels, captions, and keywords.
5. Do not invent citation keys; use TODO_ADD_CITATION if a key is missing.
6. Compile after each major phase.

Execution order:
1. Inspect repo and identify main .tex file.
2. Run baseline compile and record warnings.
3. Apply claim-control search for risky wording.
4. Rewrite abstract and introduction using docs/revision_guides/01_abstract_introduction_revision.md.
5. Revise problem setup, threat model, assumptions, and notation using docs/revision_guides/02_problem_formalization_revision.md.
6. Revise method overview and add algorithm boxes using docs/revision_guides/03_method_revision.md.
7. Reorganize related work using docs/revision_guides/04_related_work_revision.md.
8. Rewrite evaluation around research questions and claim-evidence map using docs/revision_guides/05_evaluation_plan_revision.md.
9. Rewrite limitations and discussion using docs/revision_guides/06_limitations_discussion_revision.md.
10. Apply global claim-control and style rules from docs/revision_guides/07_claim_control_and_writing_style.md.
11. Add/revise figures, tables, captions, and algorithms using docs/revision_guides/08_figures_tables_algorithms_revision.md.
12. Compile and check undefined references, undefined citations, overfull/underfull warnings, and page count.
13. Output a structured revision summary with modified files, remaining TODOs, and compile status.

Critical writing constraints:
- Do not claim clean superiority over the Qwen-adapted Scalable/Perinucleus baseline unless final matched evidence supports it.
- Do not claim full FAR calibration unless Full FAR is complete.
- Do not claim utility preservation unless matched utility results exist.
- Do not claim broad robustness; describe conditional decoding and explicit failure boundaries.
- Do not claim exact cross-family replication from diagnostic RS-aware results.
- Keep limitations explicit.
```

## 13. Final Safety Check

Before finalizing, ensure:

- [ ] Paper still compiles.
- [ ] Abstract does not overclaim.
- [ ] Introduction explains the core thesis in first 1–2 pages.
- [ ] Contributions are verifiable.
- [ ] Threat model is clear.
- [ ] Method is principled.
- [ ] Related work positions against closest baselines.
- [ ] Evaluation has RQs and placeholders for incomplete results.
- [ ] Limitations are explicit.
- [ ] No fabricated numbers.
- [ ] No fake citations.
