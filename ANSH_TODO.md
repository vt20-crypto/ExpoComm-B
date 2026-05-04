# What Ansh Needs to Do — ExpoComm-B Final Submission

**From:** Aneesh  
**Date:** May 4, 2026  

---

## Step 1 — Pull the latest repo

```bash
cd ~/ExpoComm   # or wherever you cloned it
git pull origin master
```

---

## Step 2 — Verify the MAgent Battle numbers (CRITICAL)

The final report (`final_report.tex`) uses these numbers from `EXPERIMENT_RESULTS.md` which you originally entered. **Please confirm they are correct:**

| Method | Steps | Enemy Survivors | Return |
|--------|-------|----------------|--------|
| QMIX (no comm) | 4.95M | 72.4 / 81 | -0.998 |
| ExpoComm (sparse) | 4.35M | 60.6 / 81 | -0.894 |
| ExpoComm-B σ₀=0.005 | 3.85M | 70.4 / 81 | -0.980 |
| ExpoComm-B σ₀=0.01 | 3.35M | 50.6 / 81 | -0.853 |
| ExpoComm-B σ₀=0.02 | 4.05M | 64.5 / 81 | -0.883 |
| ExpoComm-B σ₀=0.05 | 4.46M | 3.6 / 81 | -0.407 |

If any numbers are wrong, tell Aneesh the correct values and the source log file on your NOTS account.

> **Optional but preferred:** If you still have the raw log files on your NOTS account, push them:
> ```bash
> # On your NOTS account (ad258):
> scp <path_to_magent_logs>/*.log vt20@nots.rice.edu:~/ExpoComm/magent_logs/
> # OR just copy them to the repo and push
> ```

---

## Step 3 — Compile the PDF (Overleaf)

Go to [overleaf.com](https://overleaf.com) → New Project → Blank → upload these files:

| File | Purpose |
|------|---------|
| `final_report.tex` | Main report source |
| `neurips_2020.sty` | NeurIPS style file (required) |
| `magent_survivors.png` | Figure 1 |
| `mpe_learning_curves.png` | Figure 2 |
| `mpe_kl_ablation.png` | Figure 3 (left) |
| `mpe_cr_ablation.png` | Figure 4 (right) |

Set compiler to **pdfLaTeX**, click Compile. Download the PDF.

---

## Step 4 — Review the report text

Read `final_report.tex` and flag anything that is factually wrong or needs rewording. Specifically check:
- Section 4 (Method) — does the architecture description match what you implemented?
- Section 5.2 (MAgent baseline comparison) — are the numbers right?
- Section 6 (Discussion) — does the analysis of σ₀=0.05 make sense to you?

---

## What Aneesh already handled (don't redo)

- ✅ All MPE experiments (QMIX, ExpoComm, BVME-only, ExpoComm-B baselines)
- ✅ All MPE ablation experiments (λ sweep, compression ratio sweep)
- ✅ All 4 figures generated from real log data
- ✅ Full report written and pushed to GitHub
- ✅ All code fixes for PettingZoo 1.14 compatibility on NOTS
