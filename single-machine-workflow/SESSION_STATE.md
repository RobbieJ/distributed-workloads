# Session State - Single Machine ML Demo Workflow

**Last Updated:** 2026-02-03

## Repository Setup

- **Upstream:** `https://github.com/opendatahub-io/distributed-workloads` (remote: `origin`)
- **Fork:** `https://github.com/RobbieJ/distributed-workloads` (remote: `fork`)
- **Branch:** `single-machine-workflow-demos`
- **Last Commit:** `b3c9d4f` - "Add single-machine ML demo workflow with 9 interactive demos"

## What Was Completed

### 1. Implemented 12-Step UX Improvement Plan

All improvements from the plan in `~/.claude/plans/generic-churning-giraffe.md` were implemented:

| Step | Description | Status |
|------|-------------|--------|
| 1 | Add `print_intro()` to all 9 demo scripts | ✅ Done |
| 2 | Make validation test-mode-aware (`check_or_warn`) | ✅ Done |
| 3a | Suite roadmap in `run_all_demos.sh` | ✅ Done |
| 3b | Narrative synthesis in `report_all_demos.py` | ✅ Done |
| 4a | Inference demo in script 05 | ✅ Done |
| 4b | Image generation in script 07 | ✅ Done |
| 5 | `suppress_known_warnings()` in all scripts | ✅ Done |
| 6 | Narrative connections between demos | ✅ Done |
| 7 | Fix demo execution order (01→08) | ✅ Done |
| 8 | HPO uses sine-wave data (meaningful) | ✅ Done |
| 9 | Baseline comparison in 03a | ✅ Done |
| 10a | Fix `print(info())` bug in 01 | ✅ Done |
| 10b | Fix HPO metric semantics in 08 | ✅ Done |

### 2. Demo Suite Verified

Ran `bash scripts/run_all_demos.sh` — **9/9 demos passed** (0 failures).

### 3. Documentation Created

- **`DEMO_GUIDE.md`** (667 lines) — Comprehensive user guide covering setup, all 9 demos, how they connect, output interpretation, customization, and troubleshooting.

## Files Modified/Created

### Core Utilities
- `scripts/common/demo_utils.py` — Added `print_intro()`, `suppress_known_warnings()`, `check_or_warn()`

### Demo Scripts (all 9)
- `scripts/01_feast_feature_store.py` — Intro, fixed info() bug, narrative connection
- `scripts/02_sft_llm.py` — Intro, test-mode validation, narrative connection
- `scripts/03_rag_huggingface.py` — Intro, baseline comparison, narrative connection
- `scripts/03_rag_sentence_transformers.py` — Intro, narrative connection
- `scripts/04_feast_rag_ingest.py` — Intro, narrative connection
- `scripts/05_sft_feast_rag.py` — Intro, inference demo after training
- `scripts/06_deepspeed_finetune.py` — Intro, test-mode validation
- `scripts/07_dreambooth.py` — Intro, image generation, test-mode validation
- `scripts/08_hpo_optuna.py` — Intro, sine-wave data, fixed metrics, narrative connection

### Orchestration
- `scripts/run_all_demos.sh` — Suite roadmap, fixed execution order
- `scripts/report_all_demos.py` — Narrative synthesis (KEY TAKEAWAYS section)

### Documentation
- `DEMO_GUIDE.md` — New comprehensive user guide

## To Resume Work

```bash
cd /home/robbie/dev/redhat/feast-test/distributed-workloads
git checkout single-machine-workflow-demos

# Run demos
cd single-machine-workflow
bash scripts/run_all_demos.sh

# Or run individual demo
python scripts/01_feast_feature_store.py --test-mode
```

## Potential Next Steps

- [ ] Create PR to upstream (if desired)
- [ ] Add more demos (e.g., multi-GPU, FSDP)
- [ ] Integrate with Kubernetes/OpenShift deployment
- [ ] Add CI workflow for automated testing
- [ ] Video walkthrough or slides for presentations
