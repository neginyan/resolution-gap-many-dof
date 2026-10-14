# Critical rate of entropy decrease for nonlinear flows — observation study (stages A–N)

Exact (quadrature) computation, no neural networks. Setting: Y = X + N(0, σ²I), dS/dt = r σ² tr F + Σ,
R = −Σ/(σ² tr F). Flow: pendulum f = (p, −sin q), κ = 1.

## Main observations
- Single Gaussian, divergence-free f (Stein's lemma): **R = tr(F Sym J̄)/tr F**, F = (S+σ²I)⁻¹, J̄ = E_ρ[Df].
  Matches quadrature to 4.3e-12 over 1675 cases; hence R ≤ λmax(Sym J̄) ≤ κ for Gaussian states.
  The key step is Stein's lemma, the same step as statistical linearisation in Gaussian-approximation filters: for Gaussian
  states the entropy production equals that of the flow linearised with the averaged Jacobian J̄. We do not claim it is new.
- Theorem 9 limit (c) is recovered at small σ (error O(σ²)); local stretch (b) and window-averaged stretch (d) do not track R.
- Two-component mixtures: component-wise law exact when components do not overlap; fails with overlap.
  An adversarial search finds R > max_k λmax(Sym E_k Df) (σ = 0.1, ratio 1.035). No case with R > κ was found (max R = 0.949).

## Stage B (mixtures)
- Exact three-term split for any state (second-order Tweedie, Cov(X|y) = σ²I − σ⁴H(y), H = −∇² log p_Y):
  R = tr(F Sym E_ρ[Df])/tr F (≤ λ_ρ) + E[tr((H − F) Sym E[Df|Y])]/tr F + D, with D the posterior Stein defect.
- R > max_k λmax(Sym E_k Df) comes entirely from the covariance term; it needs overlapping components of different widths
  (no separation required).
- Single Gaussian: R − e = −E_ρ[cos q] K_qq K_qp / tr F exactly, so |R − e| ≤ σ² |E_ρ a''|; overlapping mixtures break this bound.

## Stage C
- Adversarial search (K = 2, 3; 24 runs) aimed at valleys: largest R/κ = 0.946; a large excess (covariance + D) appears only
  when the law term is small. Valleys lower R at small σ and raise it at σ = 0.3–1.
- Maps of local stretch, local focus and the covariance-term density; D = D_within + D_between (exact) and a scale S_D.
- Quartic oscillator and double well: law 1 and the three-term split hold; R − e = −E_ρ[g'''] K_qq K_qp / tr F in general.

## Stage C+
- Pendulum: a + a'' = 1/2 pointwise, so E_ρ[a''] = 1/2 − λ_ρ. The sign of the felt curvature does not decide the sign of the
  covariance term (59% agreement). Excess ≤ κ − law in all 1536 pendulum states (max fraction of the room used 0.714).
- Balance bound: f ≤ B = E[|a| tr|H|]/E[tr H] (exact); B ≤ κ without valleys, but B > κ in 21 valley states.

## Stage D
- Grid aligned with the flow directions V, W (obs_eval2.vw_grid); agrees with the (q, p) grid to 6e-11 on all 2352 states.
- Pendulum: R ≤ 1 ⟺ M = E[(1−ā)H_vv] + E[(1+ā)H_ww] − D tr F ≥ 0 (M/tr F = 1 − R); both parts positive in all 1536 states.
- B′ = E[|ā|(tr|H| − 2(H_ww)_+)]/E[tr H] is an exact bound for the pendulum (= f without valleys); B′ + D > 1 only for the
  largest-R state. For flows with ā < 0 the W-version fails; a V/W-switching version holds.
- Near the top: largest excess found scales with distance with slope 1.43 (1.17 for distance ≤ 0.05).

## Stage E
- Verification: tolerant, non-destructive reproducibility checks; failed checks listed by name; tolerance usage reported (verification/vtools.py).
- Local search at distance 0.1–0.3: fraction excess/distance up to 0.9515 (R = 0.9854), still creeping, pinned at distance 0.3.
  Exact: fraction = [ρ·CV(u)·Ω·(1 − λ_ρ) + D]/distance, u = 1 − ā.
- Near the top (0.0003–0.003): fraction ≤ 0.090; envelope slope 0.98 there (1.26 overall).
- |cov| ≤ √Var(ā)·Ω (Cauchy–Schwarz, rms mismatch) holds everywhere; √Var(ā) = CV(u)(1 − λ_ρ) exactly.
- P_v, P_w stay positive under direct minimisation (min 0.00062 and 0.0176).

## Stage F
- Long local searches (distance 0.1–0.6, 6 × 2400 steps): fraction excess/distance up to 0.9838 (R = 0.9928); none above 1.
- The best states are a needle at the hyperbolic point carrying ~99.9 % of the Fisher information plus broad mass far from the top.
  Shrinking the needle (t → 0) drives the fraction to 1 from below (1 − fraction ∝ t) with R(mixture) < R(needle) < λ(needle) < 1.
- Nested partition-of-unity grid (f5_needle.composite) for needles 100× narrower than the rest; the uniform V/W grid hits its
  2401-point cap there (error ~1e-7).
- R of a mixture can exceed the best single component's R (382 of 1671 states), but only by ≤ 7.8e-6 when that R ≥ 0.95.

## Stage G
- Two needles at the top (pendulum): same-shape needles always lower R (R ≤ max_k R_k); the loss is first order in
  (offset from the top)² and in Δā (equivalent at a flat top). Concentric needles of different shape raise R slightly:
  excess, gap 1 − max_k R_k and Δā all scale as σ¹, with excess/gap ≤ 0.027; shifting a needle off the top lowers this as δ².
- Every stored state (1591) satisfies R ≤ R*(σ), the best single Gaussian at the same window (largest (R − R*)/(1 − R*) = −0.057);
  1 − R*(σ) = σ(1 + O(σ)), from a needle at the top with s_w² ≈ 4σ.
- Nested-grid recheck of all 49 states where the uniform V/W grid is capped: largest change 1.9e-7, no conclusion changes.
- Needle limit: 1 − ρ·CV·Ω = c₀ + c·t with c = 2(S_vv + σ₀²)/S_ww + (−cos q_k) S_ww/(8(1 − λ_ρ₀)) from law 1
  (0.011832 vs measured 0.011830).

## Stage H
- True best single Gaussian: by law 1, R = λφ with φ independent of the V/W off-diagonal, so the optimum is a zero-width segment
  through the top, tilted from W towards −V by ≈ σ/4, half-length ≈ 2√σ. One-dimensional problem; the inner optimum is
  p* = 2/(1 + √(1 + 4σ/u)). Exact series: 1 − R* = σ − 7/8 σ² + 65/96 σ³ − 61/128 σ⁴ + … (rational coefficients to σ¹¹).
  Above σ_c = 0.93909 the supremum is the long-needle limit 1/2. The stage-G optimisation had already found R* (to 5e-14).
- R ≤ R*(σ) is FALSE: mixtures of thin segments crossing at the top with slightly different tilts (a fan) exceed R* by
  1–5 % of 1 − R* (g* up to 0.053 for K = 8; a symmetric fan levels off at 0.044), scale invariant, confirmed by two independent
  evaluators (Gauss–Hermite posterior; direct kernel summation). R < 1 throughout and (1 − R)/σ ≥ 0.95.

## Stage I
- Continuous fan (17-parameter functional family, Chebyshev series in the angle): g* = 0.0478, unchanged from 20 to 80 segments,
  scale invariant; the free K = 8 mixture of stage H (0.053) stays slightly above. Pendulum: 1 − R ≥ ~0.95σ observed.
- Lower-bound scaffold: exact identity J_tot(1 − f) = E[g H_vv] − E[g H_ww] + 2J_ww (g = 1 − ā) plus one inequality from
  H ≤ I/σ² (a bathtub bound on where the V-focus can sit): a rigorous state-wise bound, tight for single Gaussians, ≥ 0.767σ on
  all 290 states (actual ≥ 0.946σ). With full weight a line-wise Cramér–Rao inequality along W closes it with c ≈ 1.
- Other flows: κ − R* = 2√(2βκ) σ + O(σ²), β = decay of the stretch along W (pendulum 1/8, double well 3/4, quartic 3/4,
  twist 3/8); polynomial tops give a universal closed form (v* = √(4 + 9ε²) − 3ε, ε = σ√(b/κ)). In the twist flow
  H = ½ ln(1 + q² + p²) the stretching direction turns; a curved needle (curvature = turning rate 1/√2) removes the turning part
  of β and the gap drops from (√3/2)σ to 0.509σ (prediction σ/2); still ∝ σ, R < κ.
- verification/vtools.reproduce(..., loose=...) compares outputs of iterative fits / argmax locations with their own tolerances.

## Stage J
- State-wise theorem (pendulum): 1 − R ≥ Θ′σ/(1 + Θ′σ/2) − ε̃ − D (ε̃ = ε j/(σ²J_tot); corrected in stage K), from the exact identity, sin²(x/2) ≥ x²/4 − x⁴/48,
  a weighted Cramér–Rao inequality along W (Cauchy–Schwarz with the V-focus weight w = σ²(H_vv)₊ ≤ 1) and H ≤ I/σ².
  Θ′ = (|E[w u s_w]|/E[w]) √(J_ww/E[w s_w²]), u = √2 (m_q − π).  Over all 290 states: bound ≥ 0.898σ (closed form) /
  0.909σ (integral form) against actual ≥ 0.946σ (stage I: 0.767σ); Θ′ ≥ 0.9125 on all states with σ ≤ 10⁻².
- Not proved: a state-independent lower bound on Θ′ (the focus-slope term E[u ∂_w w], i.e. third derivatives of log p_Y,
  is not controlled by H ≤ I/σ²).  For priors on a single line along W (j = 1), Θ′ = 1 − σ²J_ww and c → 1.

## Stage K
- Corrected step 6 of the stage-J chain: the positive part of the numerator is divided by J_tot ≤ j/σ² + J_ww, the ε part by J_tot
  exactly — no sign condition (numbers unchanged: 0.8978 → 0.8977).
- Smoothed focus weight χ ≤ w with √χ (λ√J_ww/2)-Lipschitz along W removes the focus-slope term: 1 − R ≥ T/(1 + T/κ₂) − ε₂ − D,
  T = Θ_χ σ √(κ₂/2), Θ_χ = α G_χ/(1 + λ√P_χ). Conditional uniform bound in terms of α, G_χ, κ₂, ε₂; on the 236 near-top states
  c = 0.344 (λ = 1), state-wise ≥ 0.389σ. No unconditional c for K ≥ 2 (K = 1: 1 − R ≥ 1 − R*(σ) exactly).

## Stage L (first step towards paper C)
- Adversarial minimisation of (1 − R)/σ (pendulum): free mixtures polished by L-BFGS reach **0.9163σ** (σ = 10⁻², K = 16) and
  **0.9179σ** (σ = 10⁻³, K = 24), below the 0.946σ of stages H–K; fans (also extended / shifted) stay at 0.943–0.951. Scale invariant
  ((1 − R)/(1 − R*) within 0.0016 between σ = 10⁻³ and 10⁻²); three independent evaluators agree. Search not converged; inf ≈ 0.91 (estimate).
- Focus-slope ratios: none is uniformly bounded (ρ = 90 for a 2-component state with gap 1105σ); on near-optimal states ρ ≤ 0.275,
  giving a conditional c ≈ 0.68. Θ′ ≥ geo(1 − ρ) (proved). Fan profile (estimate): weight ∝ sech²(α/0.43), length ∝ exp(0.32α²).
- Coupled pendulums (4-d): law 1 checked against 4-d Gauss–Hermite and Monte Carlo; κ = (1 + ε + √(1 + ε²))/2 at (π, 0); best Gaussian
  gap coefficient = 2 tr[(B^½ C B^½)^½], C = diag(2κ, κ) (leading-order formula; matches the optimum to 6 digits for ε = 0.1, 0.3 and,
  out of sample, ε = 1); momentum-only directions are free, so the coefficient grows with the coupling (≈ 1 + √ε), not with the dimension.
- Twist flow: fans of bent needles (12 straight pieces each) reach 0.4894σ. Correction to stage I: a bent needle built from n straight
  pieces gives 0.503 (n = 24), 0.509 (48) … 0.569 (384) — the agreement of 0.509 with the prediction 1/2 is not established for a smooth
  curved needle (confirmed by direct kernel summation).
- L4_拡散モデル計画書.md: design (no implementation) of a phase-space diffusion-model demonstration.

## Stage M (pillars of paper C)
- Best Gaussian state in n degrees of freedom: lim (κ − R*)/σ = 2 tr[(B^½ C B^½)^½], C = diag(2κ, κ, …, κ), B the first-order loss of
  the top stretch under a covariance of q (analytic). M1: assumption-free optimisation (60 random starts per ε, full 4 × 4 factor,
  mean free) never goes below the optimum; the optimum (σ = 1e-4 … 1e-6, extrapolated) equals the formula within 1e-8 for
  ε = −0.3, −0.1, 0.1, 0.3, 1, 2, 3 (for ε < 0 the top moves to (π, π), κ = 1 − ε); Hessian: no negative direction, flat modes =
  momentum mean (exact) and the p⁻ row. M2: predictions registered before optimising (SHA-256 in logs/m2_prediction_hash.txt) for
  pendulum chains n = 3, 4 and a product-coupled pair: all confirmed within 4e-8; momentum-only directions cost nothing.
- Mixtures in the coupled pendulums (3-d reduction with p⁻ infinitely long): the gain over the best Gaussian is about half of the
  1-d gain at the same search effort (K = 2, 4).
- Uniform gap: the stage-J chain before the Cramér–Rao step is nearly exact; the gap is controlled by P = d² y/σ (focus spread ×
  W-information); P ≥ 0.858 on all 317 states (candidate inequality E[w u²] J_ww ≥ P₀ E[w]²).
- Twist flow: the optimal bent needle is a continuous bundle of straight tangent pieces of fixed length ≈ 0.44√σ (not a thin smooth
  curve); with the piece length fixed, n = 64, 128, 256 converge to 0.5035σ (prediction 1/2), scale invariant.

## Stage N (paper C: mixtures in many degrees of freedom, the P bound, the tangent bundle)
- N1: same procedure and effort (hill climb + L-BFGS-B) for the 1-d pendulum and the coupled pendulums (ε = 0.1, 0.3), K = 2 … 16,
  σ = 1e-3, product states (plane mixture × common Gaussian q⁻ law, rotation θ free), exact evaluator n_tools.R_product (checked
  against the 3-d evaluator and 4-d law 1). Ratio (κ − R)/(κ − R*) at K = 16: 0.920 (1-d), 0.958 (ε = 0.1), 0.969 (ε = 0.3).
  The multi-d gain ≈ (1-d gain) × f^a, f = plane share of the Gaussian coefficient, a ≈ 2.6 (fit). Hybrid decomposition: the
  plane mixture carries 90–99 % of the gain and raises the q⁻ cost (d_cross < 0, hypothesis (2)). Sealed prediction for ε = 1 and
  −0.3 (SHA-256 in logs/n1_prediction_hash.txt): gain within 6 % / 3 %.
- N2: P = d² y/σ alone has no positive infimum (search reaches P = 1e-4 at y/σ ≈ 100, gap ≈ 100σ); restricted to y/σ ≤ Y the
  minimum stays ≈ 0.86–0.88 and the gap stays above √P.
- N3: tangent bundle in the twist flow: gap(ℓ) ≈ 0.493 + 0.00285/ℓ + 0.0179 ℓ² (ℓ in √σ), ℓ_opt ≈ 0.43√σ; for f = (p, −q)/(1 + r^{2m})
  the optimum is m²/2 + 0.003 (σ units) with the same ℓ; by scaling ℓ_opt = c √(σ / turning-curvature), c ≈ 0.36.

## Files
| file | role |
|---|---|
| fast_eval.py | reference evaluator (copied from the code of the arrow-of-time paper, doi:10.5281/zenodo.23163727) |
| obs_eval.py | adaptive-grid evaluator, candidates (b)–(e) |
| sweep.py, sweep_tilt.py | single-Gaussian sweeps → results/sweep*.json |
| mixtures.py, mixtures_e.py | random two-component mixtures → results/mixtures*.json |
| adversarial.py | hill-climbing search → results/adversarial_*.json |
| summarize.py | every quoted number → results/summary.json |
| make_figures.py | figures/stageA_single.png, figures/stageA_law.png |
| verification/run_all.py | stage A: 29 independent checks (all pass) |
| obs_eval2.py | stage B evaluator: decompositions, (f), D, local Fisher matrix |
| m1_mechanism.py | measurement 1: slices and maps → results/m1_*.json |
| m2_remainder.py | measurement 2: remainder R − e → results/m2_*.json |
| summarize_B.py, make_figures_B.py | results/summary_B.json, figures/stageB_*.png |
| verification/run_all_B.py | stage B: 34 checks incl. an independent brute-force computation (all pass) |
| c1_adversarial.py, c1_report.py | stage C proposal 1: adversarial search, table, figure |
| c3_spatial.py | stage C proposal 3: spatial maps |
| c2_D.py | stage C proposal 2: D split and scales |
| c4_other_flows.py | stage C proposal 4: quartic oscillator and double well |
| summarize_C.py, make_figures_C.py, make_summary_C.py | results/summary_C.json, figures/stageC_*.png, summary table |
| verification/run_all_C.py | stage C: 25 checks incl. a brute-force check for the double well (all pass) |
| c5_balance.py, c5_analyze.py | stage C+: balance bound B and curvature picture on all states |
| verification/run_all_C5.py | stage C+: 20 checks (all pass) |
| d1_all_states.py | stage D tasks 1–2: all states on the V/W grid |
| d3_peak.py | stage D task 3: searches near the top of the stretch |
| d_analyze.py, make_summary_D.py | results/summary_D.json, figures/stageD_*.png, summary table |
| verification/run_all_D.py | stage D: 23 checks (all pass) |
| e1_local.py, e1b_continue.py | stage E task 1: local search of the fraction at mid distances |
| e2_dense.py | stage E task 2: dense search near the top |
| e4_valley.py | stage E task 4: minimise P_v, P_w |
| e_analyze.py | results/summary_E.json, figures/stageE_*.png |
| verification/vtools.py | shared checking helpers (named failures, tolerance usage, tolerant reproducibility) |
| verification/run_all_E.py | stage E: 23 checks (all pass) |
| f1_chain.py, f4_rcvo.py | stage F: long chains with factor tracking; ρ·CV·Ω at law ≈ λ_ρ |
| f5_needle.py | stage F: needle sequence on a nested partition-of-unity grid |
| f6_components.py | stage F: mixture R vs best single-component R |
| f_analyze.py, make_summary_F.py | results/summary_F.json, figures/stageF_*.png, summary table |
| verification/run_all_F.py | stage F: 19 checks (all pass) |
| g1_two_needles.py, g2_concentric.py | stage G: two needles near the top (families, concentric map/scale/shift, search) |
| g3_best_single.py | stage G: best single Gaussian R*(σ) and the test R ≤ R*(σ) on all states |
| g4_recheck.py, g4b_fallback.py | stage G: nested-grid recheck of all capped states |
| g5_coefficient.py | stage G: coefficient c of 1 − ρ·CV·Ω from law 1 |
| g_analyze.py | results/summary_G.json, figures/stageG_*.png |
| verification/run_all_G.py | stage G: 27 checks (all pass) |
| h1_rstar.py | stage H: true R*(σ) (1-d reduction) and its exact series |
| h2_adversarial.py, h3_xsearch.py, h4_growk.py, h5_fan.py | stage H: searches for R > R*(σ); growing K; fans of segments |
| h_analyze.py | results/summary_H.json, figures/stageH_*.png |
| verification/indep_eval.py | independent evaluators of R (Gauss–Hermite posterior; direct kernel summation) |
| verification/run_all_H.py | stage H: 23 checks (all pass) |
| i_tools.py | stage I: lean mixture evaluator for four flows (closed forms / Gauss–Hermite), fields for the bound |
| i1_fan.py | stage I: functional (continuous) fan |
| i2_bound.py | stage I: lower-bound chain on all states |
| i3_flows.py, i4_twist.py, i5_arc.py | stage I: other flows; twist flow K-growth; curved needle |
| i_analyze.py | results/summary_I.json, figures/stageI_*.png |
| verification/run_all_I.py | stage I: 29 checks (all pass) |
| j1_weighted_cr.py, j_analyze.py | stage J: weighted Cramér–Rao bound on all states; summary_J.json, figures/stageJ_bound.png |
| verification/run_all_J.py | stage J: 14 checks (all pass) |
| j2_smooth.py, k_analyze.py | stage K: smoothed focus weight; summary_K.json, figures/stageK_smooth.png |
| verification/run_all_K.py | stage K: 15 checks (all pass) |
| l_tools.py | stage L: diagnostics (Θ′, ε̃, D, ρ, the focus-slope ratios) |
| l1_free.py, l1_polish.py, l1_fan_ext.py, l1_post.py | stage L, L1(a): adversarial searches and final diagnostics |
| l1_rho_max.py, l1_ratios.py, l1_profile.py | stage L, L1(b)(c): ratio maximisation, ratio survey, fan profile fits |
| l2_bentfan.py, l2_check.py, l2_arc_continuum.py | stage L, L2: bent fans (twist flow), checks, n-dependence of the bent needle |
| l3_coupled.py | stage L, L3: coupled pendulums (4-d), law 1 check, best Gaussian, gap-coefficient formula |
| l_analyze.py | results/summary_L.json, figures/stageL_*.png |
| verification/run_all_L.py | stage L: 55 checks (all pass) |
| 本筋用要約_観測L.md, L4_拡散モデル計画書.md | stage-L summary and the L4 plan (Japanese) |
| m_tools.py | stage M: n-dof Gaussian law-1 evaluator (QR, stable to σ = 1e-7), analytic B, the formula, scaled-frame optimiser |
| m1_free.py, m1_frame.py, m1_hessian.py, m1_range.py | stage M, M1: assumption-free search, precise optimum, Hessian, other ε |
| m2_predict.py, m2_opt.py, m2_frame.py | stage M, M2: registered predictions, optimisation, precise optimum |
| m3_mix.py, m4_dichotomy.py, m5_arc.py | stage M, M3/M4/M5: mixtures (3-d), two-variable form of the bound, bent needle n = 128 |
| m_analyze.py, verification/run_all_M.py | stage M: summary_M.json, figures/stageM_*.png; checks |
| n_tools.py | stage N: product-state evaluator for the coupled pendulums (2-d grid + closed-form q⁻ integrals) |
| n1_mix.py, n1_more.py, n1_gauss.py, n1_decomp.py, n1_nonproduct.py, n1_predict.py, n1_test.py | stage N1: searches, Gaussian reference, decomposition, non-product polish, sealed prediction and its test |
| n2_pmin.py | stage N2: direct minimisation of P (unrestricted and y/σ ≤ Y) |
| n3_tangent.py | stage N3: tangent bundle vs piece length and radial profile m |
| n_analyze.py, verification/run_all_N.py | stage N: summary_N.json, figures/stageN_*.png; checks |
| 本筋用要約_観測M.md | stage-M summary (Japanese) |
| notes/本筋用要約_観測A.md … _K.md | stage summaries as written during the study (Japanese) |
| logs/ | console logs of the runs |

## Reproduce
```
python sweep.py; python sweep_tilt.py; python mixtures.py; python mixtures_e.py
python adversarial.py 0.1 0.3 1.0; python adversarial.py ratio 0.1 0.3 1.0
python summarize.py; python make_figures.py; python verification/run_all.py
python m1_mechanism.py slices; python m1_mechanism.py maps; python m2_remainder.py single; python m2_remainder.py mix
python summarize_B.py; python make_figures_B.py; python verification/run_all_B.py
python c1_adversarial.py; python c1_adversarial.py room; python c1_report.py; python c3_spatial.py; python c2_D.py
python c4_other_flows.py single; python c4_other_flows.py family; python c4_other_flows.py mix
python summarize_C.py; python make_figures_C.py; python make_summary_C.py; python verification/run_all_C.py
python c5_balance.py; python c5_analyze.py; python verification/run_all_C5.py
python d1_all_states.py; python d3_peak.py ratio; python d3_peak.py sweep; python d3_peak.py sweep2
python d_analyze.py; python make_summary_D.py; python verification/run_all_D.py
python d1_all_states.py results/e_all.json; python e1_local.py; python e1b_continue.py; python e2_dense.py; python e4_valley.py
python e_analyze.py; python verification/run_all_E.py
python f1_chain.py 2400 0.1 0.6 results/f1_chains.json; python f4_rcvo.py 600; python f5_needle.py; python f6_components.py
python f_analyze.py; python make_summary_F.py; python verification/run_all_F.py
python g1_two_needles.py; for p in map map2 scale shift search; do python g2_concentric.py $p; done; python g3_best_single.py
python g4_recheck.py 2; python g4b_fallback.py; python g5_coefficient.py; python g_analyze.py; python verification/run_all_G.py
python h1_rstar.py; python h2_adversarial.py 400; python h3_xsearch.py 500 9; python h4_growk.py 600; python h5_fan.py
python h_analyze.py; python verification/run_all_H.py
python i1_fan.py 1500; python i2_bound.py 1; python i3_flows.py single; python i3_flows.py check; python i3_flows.py search 300
python i4_twist.py 400; python i5_arc.py; python i_analyze.py; python verification/run_all_I.py
python j1_weighted_cr.py 2; python j_analyze.py; python verification/run_all_J.py
python j2_smooth.py 2; python k_analyze.py; python verification/run_all_K.py
python l3_coupled.py check; python l3_coupled.py single; python l3_coupled.py eps1
python l1_free.py 2500; python l1_free.py cont 5000; python l1_polish.py 200; python l1_fan_ext.py opt 400; python l1_fan_ext.py scan
python l1_rho_max.py 400 1; python l1_ratios.py 1; python l1_post.py; python l1_profile.py
python l2_bentfan.py; python l2_check.py; python l2_arc_continuum.py
python -c "import l2_arc_continuum as m; m.more((384,)); m.brute()"; python l_analyze.py; python verification/run_all_L.py
python m2_predict.py   # registers the predictions (writes results/m2_prediction.json and its hash) -- run only once
for e in 0.1 0.3 1; do python m1_free.py $e 60 2; done; python m1_range.py 6 2; python m1_frame.py; python m1_hessian.py
python m2_opt.py 4 2; python m2_frame.py; python m3_mix.py 1500 2,4; python m4_dichotomy.py 2; python m5_arc.py
python m_analyze.py; python verification/run_all_M.py
python n1_mix.py 1500 300 2,4,8 2; python n1_mix.py 1500 100 16 2; python n1_more.py 600 4,8   # extra iterations as in logs/n1_more_log.txt
python n1_gauss.py; python n1_decomp.py; python n1_nonproduct.py
python n1_predict.py; python n1_test.py; python n2_pmin.py 3000 300 8; python n2_pmin.py 3000 300 8 0.5   # also Y = 1, 0.3; K = 16 Y = 0.5
python n3_tangent.py ell; python n3_tangent.py m; python n_analyze.py; python verification/run_all_N.py
```
(stage L: l1_polish.py ran to K = 24 for σ = 10⁻³ and K = 16 for σ = 10⁻² before the machine restarted; l1_rho_max.py was stopped
after K = 2. The searches are random and long; reruns give other (similar) states.)
(i1_fan.py: in stage I the 4th start, a random one, was stopped after ~7 h; results/i1_fan.json holds the other three.)
(python h2_brute.py writes results/h2_brute.json, the direct-summation checks of the σ = 0.1, 0.3 violations;
results/h2_probe.json is the first violating state found by a short probe search.)
(`m2_remainder.py mix` needs about 3 GB of memory for the largest grids.)
Environment: Python 3.13.16, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2.

## Use of AI assistants
This study was carried out with the help of AI assistants: Google Gemini (Gemini 3.8 Flash) and Anthropic Claude.
All numbers are checked by the scripts in `verification/` (`python verification/run_all_stages.py` runs every stage).
