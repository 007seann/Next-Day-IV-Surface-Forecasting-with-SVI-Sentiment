# Next-Day IV Surface Forecasting with SVI & Sentiment

## Executive Summary

**Objective.** Fit the implied volatility surface via Stotastic Volatility Inspired parameter model (SVI) for NVDA and forecast the **next-day smile**; test whether simple activity/sentiment features improve out-of-sample forecasts.

**Data.** Daily NVDA option SVI parameters across expiries (2020–2022). Exogenous features: `sec_vol`, `calls_vol`, `report_vol` (SEC filings, earnings call transcripts, and buy-side analyst reports). Each day, a fixed **0.25y** maturity is synthesised via **maturity interpolation**.

**Method.**

- **SVI** (raw form): $w(k)=a+b\{\rho(k-m)+\sqrt{(k-m)^2+s^2}\}$.
  I model **variance** (σ²) → IV $=\sqrt{w(k)}$ (no division by $T$).
- **Transforms for stability:** log on `b,s0`, atanh on `rho`; identity on `a,m`.
- **No-arbitrage guard:** enforce `b>0`, `s0>0`, `|ρ|<1`, and $w_{\min}=a+b\,s\sqrt{1-\rho^2}\ge0$ via a soft repair.
- **Baseline forecast:** AR(1) on transformed params.
- **Sentiment model:** ARX(1) = AR(1) + standardized exogenous features, fit with **Ridge**.
- **Scoring:** **ATM-weighted** surface MSE over a $k$-grid; **Diebold–Mariano** test on loss differences.

**Findings (NVDA 2020–2022).**

- Baseline AR(1) is strong and stable.
- Adding sentiment yields **conditional gains** (especially on high-news days) but **no consistent average lift**; DM tests are not significant at 5%.
- Event study shows textbook behaviour: **ATM vol rises into earnings and compresses after**, **skew turns more negative** pre-event; curvature changes are small on average.

---

## Key Figures

**Figure 1 — SVI Surface (single day).**
Left-skewed equity smile; term structure flattens with maturity. The very short-dated edge touches a floor due to the **no-arb guard** preventing negative total variance (safety, not a modeling target).
<img width="496" height="505" alt="SVI " src="https://github.com/user-attachments/assets/81624764-3d3a-4db9-b8d8-b9acd495d0ff" />


**Figure 2 — Interpolated Smiles (fixed maturities).**
At $T=\{0.10,0.25,0.50,1.00\}$ years the smile is persistently left-skewed; longer maturities are smoother/flatter. The 0.10y curve shows the same protective floor at far wings.
<img width="613" height="470" alt="SVI Smiles (interpolated)" src="https://github.com/user-attachments/assets/b58bc960-270f-46c0-ae73-d9c97c5c690d" />


**Figure 3 — Event Study (Δ ATM, Δ RR25, Δ BF25 around earnings).**
<img width="844" height="547" alt="earning_event_study" src="https://github.com/user-attachments/assets/eba69644-3e65-4998-a564-dbe767aaa36a" />


- **ΔATM** > 0 into $t=0$ (risk build-up), < 0 after $t=0$ (vol crush).
- **ΔRR25** dips < 0 a few days pre-event → **puts richen** vs calls (more negative skew).
- **ΔBF25** ≈ 0 on average → modest curvature changes.
  _(If you scale by ×100, label y-axis “vol points”.)_

---

## Forecasting Results (Baseline vs Sentiment)

**Fixed daily slice via maturity interpolation** → long, clean test windows.

| T (yrs) | Median MSE (Base) | Median MSE (Sent) | Median % Improvement | DM stat |  DM p |
| ------: | ----------------: | ----------------: | -------------------: | ------: | ----: |
|    0.10 |            0.0085 |            0.0117 |              −25.79% |  −1.259 | 0.209 |
|    0.25 |            0.0291 |            0.0244 |           **+4.85%** |  −1.681 | 0.094 |
|    0.50 |            0.0483 |            0.0402 |           **+0.63%** |  −1.522 | 0.130 |
|    1.00 |            0.0065 |            0.0065 |                0.00% |       — |     — |

**Interpretation.**
Sentiment/flow features provide **situational** gains (median slightly positive at 0.25y/0.50y) but not a statistically significant average improvement. Short tenors are hardest; long tenors have limited signal density.

---

## What I Learned

- **Transforms + guards** are essential: they stabilise forecasting and keep smiles valid.
- **Maturity interpolation** avoids sparse-tenor artefacts and enables daily OOS scoring.
- **Evaluation choices matter:** ATM-weighted loss tames wing noise; DM tests prevent over-claiming small differences.
- **Sentiment is conditional:** helps in **buy-side analy's opinions-heavy** regimes, baseline dominates in quiet markets.

---

## Limitations & Next Steps

- Wings can cause outlier losses → stronger Ridge and ATM-centric weights help.
- Consider forecasting **ATM/RR/BF** directly, then reconstructing the smile.
- Add market controls (spot return, realised vol) to the ARX block.

---
