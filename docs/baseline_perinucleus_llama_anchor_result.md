# Perinucleus Llama Anchor Result

Generated at: `2026-04-29T15:44:18Z`
Decision: `LLAMA_ANCHOR_PASS: official-code Llama anchor passed; Qwen capacity sweep may be considered after review.`
Model: `meta-llama/Meta-Llama-3.1-8B`
Official commit: `fdceaba14bd3e89340916a6a40e27c945d48460e`
Training mode: `lora_adaptation`
Scratch run root: `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_llama_anchor/runs/perinucleus_llama_anchor/perinucleus_llama_anchor__baseline_perinucleus_official__llama3.1-8b-base__s17__58cf945__20260429T052904355584Z`

This is an official-code Llama anchor reproduction. It is not a Qwen matched-budget final matrix.

## Stage Metrics

| stage | fingerprints | base acc | trained acc | base target prob | trained target prob | pass | utility |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| anchor16 | 16 | 0.0 | 87.5 | 0.0002203547100371575 | 0.04604601843249763 | True | completed |
| anchor64 | 64 | 0.0 | 56.25 | 0.015751281820217046 | 0.019469288715908507 | True | completed |
| anchor128 | 128 | 0.0 | 42.96875 | 0.008171192282535197 | 0.010275651519957709 | True | completed |

## Decision

LLAMA_ANCHOR_PASS: official-code Llama anchor passed; Qwen capacity sweep may be considered after review.
