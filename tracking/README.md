# Charged-track momentum scale

File: `momentumScale_9495.json.gz`. Correction: `track_momentum_scale`.

The stored track momenta of the DELPHI 1994 (94C2) and 1995 (95D2) processings are about
0.3% too high in the barrel. This correction removes the bias. It was calibrated on
K0S → π+π− in data and validated on J/ψ → μμ, D0 → Kπ (inclusive and D*-tagged) and
D+ → Kππ.

**Data only.** Don't apply it to simulation. **One processing per key:** other
processings store different momenta and are not covered.

## Applying the correction

The correction returns a factor `s`. Multiply the momentum vector by it; the direction
doesn't change. Recompute energies with your mass hypothesis.

| Input | Type | Meaning |
|---|---|---|
| `dataset` | string | DELPHI processing: `94C2` or `95D2` |
| `variation` | string | `nominal`, `scale_up`, `scale_down`, `alignment_up`, `alignment_down` |
| `r0` | real | production radius of the track in the xy plane (mm); `0` for tracks from the primary vertex, the V0 vertex radius for V0 daughters |
| `theta` | real | polar angle of the stored momentum (rad) |
| `p` | real | magnitude of the stored momentum (GeV) |
| `q` | real | charge (+1 or −1) |

```python
import numpy as np, correctionlib
corr = correctionlib.CorrectionSet.from_file("momentumScale_9495.json.gz")["track_momentum_scale"]
p = np.linalg.norm(p3, axis=-1); theta = np.arccos(p3[..., 2] / p)

s = corr.evaluate("94C2", "nominal", r0, theta, p, q)      # scalars or numpy / awkward arrays
p3_corrected = s[..., None] * p3

# systematic variations: rerun the analysis with each one in place of "nominal"
for v in ("scale_up", "scale_down", "alignment_up", "alignment_down"):
    p3_varied = corr.evaluate("94C2", v, r0, theta, p, q)[..., None] * p3
```

```cpp
#include "correction.h"
auto corr = correction::CorrectionSet::from_file("momentumScale_9495.json.gz")->at("track_momentum_scale");
double s       = corr->evaluate({std::string("94C2"), std::string("nominal"),  r0, theta, p, q});
double s_up    = corr->evaluate({std::string("94C2"), std::string("scale_up"), r0, theta, p, q});
double s_align = corr->evaluate({std::string("94C2"), std::string("alignment_up"), r0, theta, p, q});
```

## Model

The model has three regions: backward (cos θ < −0.77), barrel, and forward
(cos θ > 0.77). It has 10 parameters per dataset: (A, ε, M) per region, plus ρ.

| Step | Formula | Details |
|---|---|---|
| 1 | p₁ = p − ρ F(r₀) g(βγ) / sin θ | DELPHI's stored momentum assumes the track comes from the beam line, so tracks produced at r₀ > 0 also carry the energy-loss correction of the material inside r₀. F(r₀) is the summed loss of the track-fit material cylinders inside r₀, from DELPHI's track-fit database records `GLOB$XEL 940401` (94C2) and `GLOB$XEL 950401` (95D2), `BLOCKDATA FKDBDAT` in `tkfb.car` (`/cvmfs/delphi.cern.ch/releases/rhel-9-x86_64/latest/simana/v96g/src/car/tkfb.car`, lines 303 and 326; cylinder radii `BMA94`/`BMA95`, lines 288 and 311). g is DELPHI's dE/dx shape for pions, `FKDEDX` in the same file (line 4181). |
| 2 | s = (p₁/p) (1 + A − ε g / p_T1 + q M p_T1) | the form used in the CMS W-mass measurement (arXiv:2412.13872): A is field-like (charge-even), ε energy-loss-like, M alignment-like (charge-odd) |

The correction is species-blind, like DELPHI's track fit, which assumes the pion mass.

## Systematic variations

Each variation is one standard deviation. Within it, every source is added in quadrature
for each track.

| Variation | Charge behaviour | Barrel (\|cos θ\| < 0.77) | End caps |
|---|---|---|---|
| `scale_up/down` | same for both charges | closure systematic σ_A = a ⊕ b·p_T ⊕ calibration statistics of A, ε, ρ | calibration statistics only |
| `alignment_up/down` | opposite for the two charges (follows q) | closure systematic σ_M ⊕ calibration statistics of M | calibration statistics only |

- **Closure systematic:** a maximum-likelihood fit of the scatter of the closure
  residuals (K0S, J/ψ, D0, D*-tagged D0, D+) beyond their statistical errors, in bins of
  the leading and of the subleading daughter p_T, fitted together. The D kaon ID uses
  DELPHI's TPC dE/dx tag only, with no RICH. The end caps have too few closure samples to
  determine it.
- **Calibration statistics:** the full covariance of the fit, including the A–ε
  correlation (r ≈ +0.8).
- **Using them:** treat `scale` and `alignment` as independent sources. Each is fully
  correlated across all tracks and events of a dataset, and uncorrelated between 94C2
  and 95D2.

Typical size, as a median relative shift of p:

| | p_T < 1 GeV | 4–8 GeV | > 15 GeV |
|---|---|---|---|
| `scale`, barrel | 0.04% | 0.10% | 0.30–0.31% |
| `scale`, end caps | 0.04–0.06% | 0.05–0.08% | 0.06–0.09% |
| `alignment`, barrel | 0.002–0.004% | 0.03–0.04% | 0.09–0.12% |
| `alignment`, end caps | 0.02–0.03% | 0.17–0.35% | 0.6–1.2% |

**Not included:** effects specific to protons and kaons, which the pion-mass track fit
doesn't describe.
