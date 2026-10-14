# Independent checks

These scripts were written separately from the study code in `../study/`. They check the analytical statements of the paper with
code that does not use the study's evaluators. Only two things are read from the study: its stored optimisation results, which are
compared against, and one Gaussian-average helper (`l3_coupled.EVpp`) in the check of Proposition 1.

| script | what it checks | usage |
|---|---|---|
| `check_paperC.py` | Lemma 2 on random matrices (attained value and numerical infimum); Theorem 3 with the exact block form of F; Proposition 1; Corollary 4; the strain-decay matrix B from its closed form and from a numerical Hessian of e₊ᵀV''e₊; Γ for all thirteen systems; the weak-coupling series; the stage-L optimum and its shape against the formula; singular Gaussian states as limits of non-degenerate ones; that the largest strain lies on the positive side, as assumed in the definition of κ (Sec. II), in every example. 64 checks. | `python check_paperC.py` |
| `bq.py` | B and Γ for the coupled pendulums from the analytic Hessian, against the study's finite-difference values (ε = 0.1, 0.3, 1) | `python bq.py` |
| `m1_free.py` | a shape-free search for the best Gaussian state of two coupled pendulums, with a parametrisation different from the study's (mean free on the torus, general lower-triangular factor in the original coordinates). The stored runs are `m1_e0.1_s0.001_seed1.json`, `m1_e0.3_s0.001_seed2.json`, with logs `log_e0.1.txt`, `log_e0.3.txt`. | `python m1_free.py eps sigma starts seed` |
| `m2_main_prediction.py` | a second, independent computation of the predicted Γ for the pendulum chains (Sec. V of the paper). Its output `m2_main_prediction.json` (SHA-256 `7f8b5629…c8bb29`, written 2026-10-13T04:08Z) agrees with the study's sealed predictions to seven digits. | `python m2_main_prediction.py` |

Run every script from this folder. `check_paperC.py` exits with code 0 if and only if all checks pass.
