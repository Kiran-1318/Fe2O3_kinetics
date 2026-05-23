# Fe₂O₃ Kinetics — Coupled Inverse PINN and SINDy

Data-driven discovery of Fe₂O₃ reduction kinetics using a coupled
Inverse Physics-Informed Neural Network (PINN) and SINDy (Sparse
Identification of Nonlinear Dynamics) pipeline.

**No prior kinetic model assumed.** Both the Arrhenius parameters
(A, Eₐ) and the reaction model f(X) are discovered simultaneously
from sparse isothermal TGA data.

**Related preprint (Notebook 1 + 2):**
[Physics-Informed Neural Networks and Operator Learning for Fe₂O₃
Reduction Kinetics Identification from Sparse Thermogravimetric Data](https://doi.org/10.26434/chemrxiv.15003636/v1)
— ChemRxiv, May 2026

---

## Notebook

### `3_inv_PINN_SINDY.ipynb` — Coupled Inverse PINN + SINDy

**Research question:** Without assuming any kinetic model, what are
the Arrhenius parameters A and Eₐ, and what functional form f(X)
governs Fe₂O₃ reduction by H₂?

**Governing equation (unknown f(X)):**
dX/dt = A · exp(−Eₐ/RT) · f(X),   X(0) = 0

**Algorithm — coupled iterative loop:**
Initialise: A⁰, Eₐ⁰ (arbitrary), f⁰(X) = (1−X)^(2/3)
Iteration k:
Step A — Inverse PINN:
Fix f^k(X) → optimise network weights + log A + log Eₐ
Physics residual: dX/dt|PINN − A·exp(−Eₐ/RT)·f^k(X) = 0
Output: A^(k+1), Eₐ^(k+1)
Step B — SINDy sparse regression:
Fix A^(k+1), Eₐ^(k+1) → compute f̂(X) = dX/dt|PINN / k(T)
Sparse regression with modified BIC threshold sweep
Physical validity enforced: f(X) ≥ 0, cancellation penalty
Output: f^(k+1)(X) — symbolic, 1 active term
Until ΔA < 1% and ΔEₐ < 1%

**Key design decisions (from literature):**
- PINN autograd derivatives for f(X) — no numerical differentiation
  noise (Chen et al. 2021, Nature Communications)
- Modified BIC with physical cancellation penalty for sparsity
  (Brunton et al. 2016 SINDy; Zhao et al. 2025)
- Alternate optimisation — PINN and SINDy never share a backprop pass
  (Zhao et al. 2025, Stephany & Earls 2022)

**SINDy candidate library (8 terms, no constant):**

| Term | Model |
|---|---|
| `(1−X)^(2/3)` | Shrinking core (SCM) |
| `(1−X)^(1/3)` | Grain model (GM) |
| `(1−X)` | First order |
| `(−ln(1−X))^0.5` | Avrami-Erofeev n=0.5 |
| `(−ln(1−X))^(2/3)` | Avrami-Erofeev n=2/3 |
| `X(1−X)` | Autocatalytic |
| `1/(2X)` | Parabolic diffusion P2 |
| `1−(2X/3)−(1−X)^(2/3)` | Ginstling-Brounshtein D4 |

**Final results:**

| | Value |
|---|---|
| Converged at iteration | 8 |
| ΔA | 0.33% |
| ΔEₐ | 0.08% |
| Recovered A | 7.50×10⁻² s⁻¹ |
| Recovered Eₐ | 27.75 kJ/mol |
| Discovered f(X) | **0.990·(1−X)^(1/3)** — grain model |
| Active SINDy terms | 1 |
| Starting conditions | A=5 s⁻¹, Eₐ=50 kJ/mol (arbitrary) |

**Forward validation:**

| Temperature | Status | R² |
|---|---|---|
| 750°C | Training | 0.9870 |
| 850°C | Training | 0.9819 |
| 900°C | Training | 0.9759 |
| 800°C | Validation (interpolation) | 0.9211 |
| 950°C | Validation (extrapolation) | 0.9530 |

**Comparison with existing methods:**

| Method | f(X) assumed | Eₐ | Model assumption |
|---|---|---|---|
| Wang 2023 (TGA) | JMA nucleation | 10.3–26.7 kJ/mol | Yes |
| Preprint 1 PINN | (1−X)^(2/3) SCM | 24.07 kJ/mol | Yes |
| **This work** | **(1−X)^(1/3) grain** | **27.75 kJ/mol** | **No** |

---

## Methodology schematic

`pinn_sindy_v3.pdf` — publication-quality multi-panel figure showing
the coupled iterative pipeline (panels a–e).

`pinn_sindy_v3.tex` — LaTeX/TikZ source for the schematic.

---

## Data source

Wang H. et al. (2023). "Multistep kinetic study of Fe₂O₃ reduction
by H₂ based on isothermal thermogravimetric analysis data deconvolution."
*Int. J. Hydrogen Energy*, 48, 16601–16613.

Sparse baseline-corrected TGA data at 750, 800, 850, 900, 950°C
(same dataset as the related preprint).

---

## Requirements
torch>=2.0.0
pysindy>=1.7.0
numpy
pandas
matplotlib
scikit-learn
scipy

---

## Related repositories

[1D-Heat-Equation-PINN](https://github.com/Kiran-1318/1D-Heat-Equation-PINN)
— Heat equation PINN. Rel L2: 0.99%.

[DeepXDE-Chemical-Looping-Problems](https://github.com/Kiran-1318/DeepXDE-Chemical-Looping-Problems)
— 10 original PINN problems.

[NeuralOperator-Chemical-Looping-Problems](https://github.com/Kiran-1318/NeuralOperator-Chemical-Looping-Problems)
— 5 neural operator problems including PI-DeepONet.

[Fe2O3_redox_PINN](https://github.com/Kiran-1318/Fe2O3_redox_PINN)
— Preprint 1: inverse PINN + PI-DeepONet for lumped kinetics.
Preprint DOI: https://doi.org/10.26434/chemrxiv.15003636/v1

---

## Author

**Kiran Thammina**
M.Tech Energy Systems Engineering, IIT Bombay (CPI 9.84, Best Thesis Award)
GitHub: [github.com/Kiran-1318](https://github.com/Kiran-1318)
