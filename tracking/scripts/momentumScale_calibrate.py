"""Calibrate the charged-track momentum scale on K0S, test it on the closure samples, and
write the calibration file that momentumScale_makeCorrection.py bakes into correctionlib.

    python3 momentumScale_calibrate.py fit     --skim skim_94C2.npz --output calibration_94C2.json
    python3 momentumScale_calibrate.py closure --skim skim_94C2.npz --calibration calibration_94C2.json
    python3 momentumScale_calibrate.py check   --skim skim_94C2.npz --calibration calibration_94C2.json \
        --correction momentumScale_9495.json.gz

  fit      K0S calibration: the 10 model parameters and their covariance.
  closure  Closure on all samples in the barrel, and the closure systematic fitted to
           it; adds the uncertainties to the calibration file.
  check    The correctionlib file against the model, on the skimmed tracks.

The input is the output of momentumScale_skim.py for one DELPHI processing.

Model, per track (the stored momenta are corrected; there is no refit):

  1. Energy loss of material the track never crossed. DELPHI's stored momentum assumes
     the track comes from the beam line, so a track produced at radius r0 also carries
     the energy-loss correction of the material cylinders inside r0:
         p1 = p - rho * F(r0) * g(beta gamma) / sin(theta)
     F(r0) = summed loss of the track-fit material cylinders inside r0 (per processing,
     MATERIAL); g = DELPHI's dE/dx shape FKDEDX for the pion mass; rho a free factor.
  2. Residual bias, as in the CMS W-mass measurement (arXiv:2412.13872, App. A.5):
         s = (p1 / p) * (1 + A - eps * g / pt1 + q * M * pt1),   pt1 = p1 sin(theta)
     A (field-like), eps (GeV, energy-loss-like) and M (GeV^-1, alignment-like) per
     region: backward (cos theta < -0.77), barrel, forward (cos theta > 0.77).
  The corrected momentum is s * p, direction unchanged.

Calibration: K0S -> pi+ pi- at all decay radii; the peak position of each bin (decay
radius x region and p_T of each pion) is fitted, and the parameters solve the
linearised mass shifts against the PDG mass (chi2 fit, iterated).

Closure (barrel), after CMS: per p_T bin of the subleading or of the leading daughter,
each sample (K0S, J/psi -> mu mu, D0 -> K pi inclusive and D*-tagged, D+ -> K pi pi) is
split by which charge carries more p_T; the two peak positions give a residual
charge-even A' and charge-odd M'. Their scatter beyond the statistical errors (of the
closure point and of the calibration) gives the closure systematic: sigma_A = a (+) b p_T
and a constant sigma_M, by one maximum-likelihood fit to the points of both binnings.
The end caps have too few closure samples: statistics only there.

Peak fits (K0S bins and cells, meson full samples and cells): binned Poisson likelihood,
minimised with MIGRAD and errors from HESSE (iminuit), with bounded parameters. Line
shape: a Voigtian (Gaussian convolved with a Lorentzian) on a background, in a fixed
mass window and binning per sample. The significance of a peak is the likelihood ratio
against the background-only fit.
"""
import argparse, json, math, sys
from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------- constants
M_PI, M_K, M_MU = 0.13957, 0.49368, 0.1056584      # GeV
PDG_MASS = {'K0S': 0.497611, 'J/psi': 3.096900, 'D0': 1.86484, 'D+': 1.86966}

# Material cylinders of DELPHI's track fit: radius (mm) and energy loss of a
# minimum-ionising particle at normal incidence (GeV), the inner seven of the track-fit
# database records GLOB$BMA / GLOB$XEL (simana v96g tkfb.car, BLOCKDATA FKDBDAT: BMA94
# line 288, XEL94 line 303, BMA95 line 311, XEL95 line 326). The radii are the same in
# both years.
CYLINDER_RADIUS = [52.5, 57.0, 83.0, 103.0, 123.0, 253.0, 331.0]
MATERIAL = {
    '94C2': [0.0008, 0.0004, 0.0004, 0.0004, 0.0008, 0.0018, 0.0025],    # GLOB$XEL 940401
    '95D2': [0.0008, 0.0004, 0.0004, 0.0004, 0.0011, 0.0009, 0.0025],    # GLOB$XEL 950401
}

# FKDEDX (tkfb.car line 4181): DELPHI's dE/dx in MIP units at log10(beta gamma) =
# -0.64, -0.60, ..., 4.92 (140 points)
DEDX_LOG10_BG = [-0.64 + 0.04 * i for i in range(140)]
DEDX_MIP = [
    13.61, 11.84, 10.24, 8.82, 7.590, 6.530, 5.625, 4.856, 4.206, 3.656,
    3.194, 2.805, 2.479, 2.206, 1.978, 1.788, 1.630, 1.500, 1.392, 1.304,
    1.232, 1.174, 1.128, 1.091, 1.063, 1.041, 1.025, 1.013, 1.006, 1.002,
    1.000, 1.001, 1.003, 1.007, 1.012, 1.019, 1.026, 1.034, 1.043, 1.052,
    1.061, 1.071, 1.081, 1.092, 1.102, 1.113, 1.124, 1.135, 1.146, 1.157,
    1.168, 1.179, 1.190, 1.201, 1.212, 1.222, 1.233, 1.243, 1.252, 1.262,
    1.272, 1.281, 1.290, 1.300, 1.309, 1.318, 1.327, 1.336, 1.345, 1.355,
    1.364, 1.373, 1.382, 1.391, 1.400, 1.409, 1.418, 1.426, 1.434, 1.442,
    1.450, 1.457, 1.464, 1.471, 1.477, 1.482, 1.487, 1.492, 1.496, 1.500,
    1.503, 1.505, 1.507, 1.509, 1.510, 1.512, 1.513, 1.514, 1.515, 1.516,
    1.517, 1.518, 1.519, 1.519, 1.520, 1.521, 1.521, 1.521, 1.522, 1.522,
    1.523, 1.523, 1.523, 1.524, 1.524, 1.524, 1.524, 1.525, 1.525, 1.525,
    1.525, 1.525, 1.525, 1.525, 1.526, 1.526, 1.526, 1.526, 1.526, 1.526,
    1.526, 1.526, 1.526, 1.526, 1.526, 1.526, 1.526, 1.526, 1.526, 1.526]

# regions in signed cos(theta); parameters (A, eps, M) per region, then rho
REGIONS = ('backward', 'barrel', 'forward')
REGION_COS_THETA_EDGES = (-1.0, -0.77, 0.77, 1.0)
BARREL = 1
PARAMETERS = ['A_bwd', 'eps_bwd', 'M_bwd', 'A_bar', 'eps_bar', 'M_bar', 'A_fwd', 'eps_fwd', 'M_fwd', 'rho']
N_PARAMETERS = len(PARAMETERS)
# steps of the numerical derivatives: A, eps (GeV), M (GeV^-1) of each region, rho
DERIVATIVE_STEP = [1e-4, 1e-4, 1e-5] * 3 + [1e-2]

# K0S selection. DELPHI's tight K0S tag decides K0S against Lambda by the smaller mass
# pull where a V0 fits both (NIM A 378 section 8), which biases the K0S mass. So all
# standard V0s are taken as K0S, and the Lambda is removed by the Armenteros q_T instead:
# q_T > 0.12 GeV keeps the K0S band above the Lambda ellipse (q_T <= 101 MeV).
K0S_TAGS = (1, 2, 3, 22, 33)
K0S_MASS_WINDOW = (0.35, 0.65)        # GeV, m(pi pi) of the stored momenta
K0S_MIN_QT = 0.120                    # GeV

# K0S calibration bins: decay radius x (region, p_T) of each pion. Candidates with a
# pion outside the barrel (~13%) get coarser bins.
K0S_PT_EDGES = (0.0, 0.35, 0.6, 1.0, 1.8, 50.0)                        # GeV
K0S_RADIUS_EDGES = (0.0, 40.0, 53.0, 83.0, 123.0, 253.0, 331.0, 1200.0)  # mm, between the cylinders
K0S_PT_EDGES_ENDCAP = (0.0, 0.5, 1.2, 50.0)
K0S_RADIUS_EDGES_ENDCAP = (0.0, 40.0, 123.0, 1200.0)
K0S_MIN_CANDIDATES = 300              # per calibration bin
K0S_ITERATIONS = 4
# K0S peak fits: a fixed window around the PDG mass, (half width (GeV), number of bins),
# for candidates with both pions in the barrel and for those with a pion in an end cap;
# it holds the widest peaks (boosted K0S at large radius, ~20 MeV) and their sidebands.
# Starting width (GeV) of the fits for 0, 1 or 2 pions outside the barrel. A calibration
# bin needs a signal of K0S_MIN_SIGNIFICANCE standard deviations and a position error
# below K0S_MAX_POSITION_ERROR (GeV).
K0S_WINDOW_BARREL = (0.060, 60)
K0S_WINDOW_ENDCAP = (0.080, 40)
K0S_START_WIDTH = (0.0035, 0.006, 0.009)
K0S_MIN_SIGNIFICANCE = 5.0
K0S_MAX_POSITION_ERROR = 0.003

# D*-tagged D0: window in Delta M = M(K pi pi_soft) - M(K pi) (GeV)
DSTAR_DM_WINDOW = (0.1435, 0.1480)

# Closure: p_T bins of the subleading or the leading daughter (GeV), barrel
CLOSURE_PT_EDGES = (0.1, 0.3, 0.6, 1.0, 2.0, 4.0, 8.0, 15.0, 45.0)
CLOSURE_BINNINGS = ('sublead', 'lead')
# minimum candidates in the fit range: full sample (mesons), closure cell; minimum signal
# significance of a meson full-sample fit
MIN_FULL, MIN_CELL = 100, 15
MESON_FULL_MIN_SIGNIFICANCE = 3.0

# Line shape of every peak: a Voigtian (Gaussian of width sigma convolved with a
# Lorentzian of half width gamma), for the resolution and its tails, on a background.
# K0S fits leave sigma and gamma free within K0S_SIGMA_RANGE and K0S_GAMMA_RANGE (GeV),
# on a linear background given by its values at the two window edges (both >= 0). A
# peak needs a full width at half maximum of at least K0S_MIN_FWHM, which one-bin spikes
# (~3 MeV) do not reach and real K0S peaks (>= ~10 MeV) exceed; a fit ending on the
# upper sigma or gamma limit has no usable peak. Meson cells keep sigma and gamma from
# their full-sample fit.
K0S_SIGMA_RANGE = (0.001, 0.030)
K0S_GAMMA_RANGE = (0.0, 0.020)
K0S_MIN_FWHM = 0.005
GAMMA_MAX = 0.05                        # GeV, mesons

# Meson peaks: background, its reference point m0 (GeV), mass range (GeV), number of
# bins, the plausible range of the full-sample FWHM (GeV) and the starting values of
# (signal yield, position, sigma, gamma, background parameters). A start given as
# ('n', f) is f times the number of candidates in the range.
#   exp       a exp(-b (m - m0))
#   swap_exp  D0: 170 MeV wide Gaussian of K <-> pi swaps (amplitude n_swap) + exp
#   quad      D+: a + b (m - m0) + c (m - m0)^2
MESON_PEAKS = {
    'J/psi':      dict(background='exp', m0=3.1, range=(2.75, 3.45), bins=35, fwhm=(0.030, 0.200),
                       start=[('n', 0.5), 3.095, 0.03, 0.005, ('n', 1 / 100), 1.0]),
    'D0 incl':    dict(background='swap_exp', m0=1.8, range=(1.72, 2.10), bins=76, fwhm=(0.015, 0.100),
                       start=[('n', 0.05), 1.865, 0.018, 0.005, ('n', 1 / 200), ('n', 1 / 80), 1.0]),
    'D0 from D*': dict(background='exp', m0=1.8, range=(1.70, 2.00), bins=60, fwhm=(0.015, 0.100),
                       start=[('n', 0.2), 1.865, 0.018, 0.005, ('n', 1 / 60), 3.0]),
    'D+':         dict(background='quad', m0=1.87, range=(1.72, 2.04), bins=64, fwhm=(0.015, 0.100),
                       start=[('n', 0.05), 1.869, 0.012, 0.004, ('n', 1 / 70), 0.0, 0.0]),
}

# Requirements on a closure cell: signal significance, position error (GeV); a meson
# position may move by at most max(25 MeV, 0.64 FWHM) from the full-sample fit (0.64 FWHM
# = 1.5 sigma of a Gaussian)
CELL_MIN_SIGNIFICANCE = 2.0
CELL_MAX_POSITION_ERROR = 0.030
CELL_POSITION_LIMIT = (0.025, 0.64)
K0S_CELL_MAX_POSITION_ERROR = 0.005

# typical sin(theta) of a barrel track, to turn a bin's p_T into |p| for g
TYPICAL_SIN_THETA = 0.8


# ---------------------------------------------------------------- the model
class Model:
    """The correction of one processing: s(p3, q, r0) for parameters par."""

    def __init__(self, dataset):
        # the material table is the only per-processing input of the model
        if dataset not in MATERIAL:
            sys.exit(f'no material table for {dataset}; known: {", ".join(MATERIAL)}')
        self.dataset = dataset
        self.radius = np.array(CYLINDER_RADIUS)
        self.eloss = np.array(MATERIAL[dataset])

    @staticmethod
    def dedx_shape(p):
        """g = FKDEDX at beta gamma = p / m_pi (the pion mass the track fit assumes)."""
        # linear interpolation in log10(beta gamma); held constant beyond the table
        return np.interp(np.log10(np.maximum(p, 1e-6) / M_PI), DEDX_LOG10_BG, DEDX_MIP)

    def inner_eloss(self, r0):
        """F(r0): summed energy loss (GeV) of the cylinders inside r0 (mm)."""
        # one column per cylinder: its loss if it lies inside r0, else 0
        return (self.eloss * (self.radius < np.asarray(r0)[..., None])).sum(-1)

    @staticmethod
    def region(p3):
        """Region index per track: 0 backward, 1 barrel, 2 forward."""
        cos_theta = p3[..., 2] / np.linalg.norm(p3, axis=-1)
        return np.clip(np.digitize(cos_theta, REGION_COS_THETA_EDGES) - 1, 0, len(REGIONS) - 1)

    def correct(self, p3, q, r0, par):
        """Corrected momenta. p3 (..., 3); q and r0 broadcastable to p3[..., 0]."""
        p3 = p3.astype(np.float64)
        p = np.linalg.norm(p3, axis=-1)
        sin_theta = np.hypot(p3[..., 0], p3[..., 1]) / p
        g = self.dedx_shape(p)

        # step 1: remove the energy-loss correction of the material inside r0 (the loss
        # grows with the path length through a cylinder, 1 / sin(theta))
        rho = par[-1]
        p1 = p - rho * self.inner_eloss(np.broadcast_to(r0, p.shape)) * g / sin_theta
        pt1 = p1 * sin_theta

        # step 2: the residual bias with the parameters of the track's region
        per_region = np.asarray(par[:-1]).reshape(len(REGIONS), 3)[self.region(p3)]
        A, eps, M = per_region[..., 0], per_region[..., 1], per_region[..., 2]
        s = (p1 / p) * (1 + A - eps * g / pt1 + q * M * pt1)
        return p3 * s[..., None]                         # direction unchanged

    def mass_derivatives(self, sample, par):
        """Mass of each candidate after the correction, and d(mass)/d(parameter)."""
        base = sample.mass(self.correct(sample.p3, sample.q, sample.r0, par))
        derivatives = np.zeros((len(base), N_PARAMETERS))
        # forward differences, one parameter at a time
        for j in range(N_PARAMETERS):
            shifted = np.array(par, float)
            shifted[j] += DERIVATIVE_STEP[j]
            derivatives[:, j] = (sample.mass(self.correct(sample.p3, sample.q, sample.r0, shifted)) - base) / DERIVATIVE_STEP[j]
        return base, derivatives


# ---------------------------------------------------------------- samples
@dataclass
class Sample:
    """Candidates of one decay: daughter momenta (n, daughters, 3), charges (n, daughters),
    production radius (n, daughters; 0 for prompt decays) and the daughter masses."""
    name: str
    p3: np.ndarray
    q: np.ndarray
    r0: np.ndarray
    daughter_masses: tuple

    @property
    def reference_mass(self):
        # both D0 samples ('D0 incl', 'D0 from D*') are measured against the D0 mass
        return PDG_MASS['D0' if self.name.startswith('D0') else self.name]

    def mass(self, p3):
        """Invariant mass of the daughters with momenta p3 and the sample's masses."""
        energy = sum(np.sqrt((p3[:, i]**2).sum(1) + m**2) for i, m in enumerate(self.daughter_masses))
        total = p3.sum(1)
        return np.sqrt(np.maximum(energy**2 - (total**2).sum(1), 0))

    def subset(self, keep):
        return Sample(self.name, self.p3[keep], self.q[keep], self.r0[keep], self.daughter_masses)

    def corrected(self, model, par):
        """The sample with corrected momenta (r0 kept, for further corrections on top)."""
        return Sample(self.name, model.correct(self.p3, self.q, self.r0, par), self.q, self.r0, self.daughter_masses)


def armenteros_qt(p3):
    """Armenteros q_T of a V0 (GeV): daughter momentum transverse to the V0 direction."""
    total = p3.sum(1)
    u = total / np.linalg.norm(total, axis=1)[:, None]          # V0 flight direction
    p_long = (p3[:, 0] * u).sum(1)                              # positive daughter along it
    return np.linalg.norm(p3[:, 0] - p_long[:, None] * u, axis=1)


def load_samples(path):
    """The skim file -> (dataset, name -> Sample) with the final selections applied."""
    skim = np.load(path)
    dataset = str(skim['dataset'])
    get = lambda key: skim[key].astype(np.float64)
    samples = {}

    # K0S: standard V0s in the m(pi pi) window, Lambda removed by q_T; every daughter
    # carries the V0 vertex radius
    p3, q = get('k0s_p3'), get('k0s_q')
    r0 = np.repeat(get('k0s_r0')[:, None], 2, 1)
    k0s = Sample('K0S', p3, q, r0, (M_PI, M_PI))
    m = k0s.mass(p3)
    keep = (np.isin(skim['k0s_tag'], K0S_TAGS) & (m > K0S_MASS_WINDOW[0]) & (m < K0S_MASS_WINDOW[1])
            & (armenteros_qt(p3) > K0S_MIN_QT))
    samples['K0S'] = k0s.subset(keep)

    # prompt decays: the daughters come from the primary vertex, r0 = 0
    prompt = lambda name, key, masses: Sample(name, get(f'{key}_p3'), get(f'{key}_q'),
                                              np.zeros(skim[f'{key}_q'].shape), masses)
    samples['J/psi'] = prompt('J/psi', 'jpsi', (M_MU, M_MU))
    samples['D0 incl'] = prompt('D0 incl', 'd0', (M_K, M_PI))
    samples['D+'] = prompt('D+', 'dplus', (M_K, M_PI, M_PI))

    # D*-tagged D0: the Delta M window; the D0 is (K, pi), without the soft pion
    dm = skim['dstar_dm']
    keep = (dm > DSTAR_DM_WINDOW[0]) & (dm < DSTAR_DM_WINDOW[1])
    samples['D0 from D*'] = Sample('D0 from D*', get('dstar_p3')[keep][:, :2], get('dstar_q')[keep][:, :2],
                                   np.zeros((keep.sum(), 2)), (M_K, M_PI))
    return dataset, samples


# ---------------------------------------------------------------- peak fits
class Failure:
    """A fit or cell without a result, with the reason; nothing is dropped silently."""

    def __init__(self, why):
        self.why = why

    def __repr__(self):
        return f'FAILED ({self.why})'


def voigt(x, sigma, gamma):
    """Voigtian density of unit integral: Gaussian (sigma) convolved with a Lorentzian of
    half width gamma (GeV)."""
    from scipy.special import voigt_profile
    return voigt_profile(x, sigma, gamma)


def poisson_fit(function, x, counts, start, lower, upper, steps):
    """Binned Poisson likelihood fit within bounds, minimised with MIGRAD (iminuit) and
    errors from HESSE. steps: starting step of each parameter, on the scale of its
    uncertainty (the parameters differ by many orders of magnitude).
    -> (parameters, covariance, significance) or (None, None, None). The significance of
    the signal (parameter 0, the yield) is sqrt(2 (nll(yield = 0) - nll(best))), from a
    second fit with the yield fixed to 0."""
    from iminuit import Minuit

    def nll(p):
        # Poisson negative log likelihood, without the constant log(counts!)
        mu = np.maximum(function(x, *p), 1e-9)
        return float((mu - counts * np.log(mu)).sum())

    def minimise(start, fix_yield=False, strategy=1):
        minuit = Minuit(nll, np.clip(start, lower, upper))
        minuit.errordef = Minuit.LIKELIHOOD
        minuit.strategy = strategy
        minuit.errors = steps
        minuit.limits = [(None if np.isinf(a) else a, None if np.isinf(b) else b) for a, b in zip(lower, upper)]
        if fix_yield:
            minuit.values[YIELD] = 0.0
            minuit.fixed[YIELD] = True
        minuit.migrad()
        return minuit

    # MIGRAD's default strategy first; the careful strategy 2 (second derivatives checked)
    # when that does not give a valid minimum with an accurate covariance: it handles
    # flat directions (e.g. sigma << gamma), while strategy 1 handles minima on a limit
    for strategy in (1, 2):
        best = minimise(start, strategy=strategy)
        if best.valid:
            best.hesse()
            if best.valid and best.accurate:
                break
    else:
        return None, None, None
    background_only = minimise(np.array(best.values), fix_yield=True)
    significance = math.sqrt(max(2 * (background_only.fval - best.fval), 0.0))
    return np.array(best.values), np.array(best.covariance), significance


def voigt_fwhm(sigma, gamma):
    """Full width at half maximum of a Voigtian (Olivero and Longbothum, 0.02% accuracy)."""
    f_gauss, f_lorentz = 2 * math.sqrt(2 * math.log(2)) * sigma, 2 * gamma
    return 0.5346 * f_lorentz + math.sqrt(0.2166 * f_lorentz**2 + f_gauss**2)


def at_bound(value, lower, upper, tolerance=1e-3):
    """True if value sits on one of its bounds (within tolerance x the allowed range)."""
    span = (upper - lower) if np.isfinite(upper - lower) else max(abs(value), 1.0)
    return abs(value - lower) < tolerance * span or abs(value - upper) < tolerance * span


# Parameter order of every peak fit: signal yield, position, sigma, gamma, background
YIELD, POSITION, SIGMA, GAMMA = 0, 1, 2, 3


class K0sPeak:
    """K0S peak fit in the fixed window of one category (0, 1 or 2 pions outside the
    barrel): signal yield x Voigtian + linear background, all parameters free within
    their ranges."""

    def __init__(self, n_endcap_pions):
        self.reference_mass = PDG_MASS['K0S']
        self.half, self.bins = K0S_WINDOW_ENDCAP if n_endcap_pions else K0S_WINDOW_BARREL
        self.start_width = K0S_START_WIDTH[n_endcap_pions]

    def histogram(self, m):
        h, e = np.histogram(m, bins=self.bins, range=(self.reference_mass - self.half, self.reference_mass + self.half))
        return 0.5 * (e[1:] + e[:-1]), h, e[1] - e[0]

    def function(self, bin_width):
        """Signal yield x Voigtian + linear background with the values left and right at
        the two window edges."""
        lo, span = self.reference_mass - self.half, 2 * self.half

        def peak(x, n, mu, sigma, gamma, left, right):
            t = (x - lo) / span                              # 0 at the lower, 1 at the upper edge
            return n * bin_width * voigt(x - mu, sigma, gamma) + left * (1 - t) + right * t
        return peak

    def fit(self, m, min_significance, max_error):
        """One calibration bin or closure cell, masses m -> dict(shift = position - PDG,
        error, width = FWHM, sigma, gamma) or a Failure with the reason."""
        x, h, w = self.histogram(m)

        # background level at the two edges of the window (mean of the outer five bins)
        left_level, right_level = float(h[:5].mean()), float(h[-5:].mean())
        edge_level = max(left_level, right_level, 1.0)

        # parameters: signal yield, position, sigma, gamma, background left and right
        start = [0.5 * h.sum(),                         # half of the candidates as signal
                 self.reference_mass,
                 0.7 * self.start_width,
                 0.3 * self.start_width,
                 max(left_level, 0.1),
                 max(right_level, 0.1)]
        lower = [0,
                 self.reference_mass - self.half / 2,   # position within the inner half of the window
                 K0S_SIGMA_RANGE[0],
                 K0S_GAMMA_RANGE[0],
                 0,                                     # background never negative
                 0]
        upper = [np.inf,
                 self.reference_mass + self.half / 2,
                 K0S_SIGMA_RANGE[1],
                 K0S_GAMMA_RANGE[1],
                 np.inf,
                 np.inf]
        # starting steps on the scale of each parameter's uncertainty
        steps = [math.sqrt(start[0]) + 1,               # yield: Poisson fluctuation
                 1e-4, 1e-4, 1e-4,                      # position, sigma, gamma: 0.1 MeV
                 math.sqrt(edge_level) + 1,
                 math.sqrt(edge_level) + 1]

        p, cov, significance = poisson_fit(self.function(w), x, h, start, lower, upper, steps)
        if p is None:
            return Failure('fit did not converge')
        if not cov[POSITION, POSITION] > 0:
            return Failure('likelihood not positive definite at the minimum')
        error = math.sqrt(cov[POSITION, POSITION])
        fwhm = voigt_fwhm(p[SIGMA], p[GAMMA])

        # a usable peak: position off its limits; the width not stuck at an upper limit
        # (no peak in the window) and not a one-bin spike (FWHM); a significant signal;
        # a precise position
        if at_bound(p[POSITION], lower[POSITION], upper[POSITION]):
            return Failure(f'position at its +-{1e3 * self.half / 2:.0f} MeV limit')
        sigma_at_limit = abs(p[SIGMA] - upper[SIGMA]) < 1e-3 * upper[SIGMA]
        gamma_at_limit = abs(p[GAMMA] - upper[GAMMA]) < 1e-3 * upper[GAMMA]
        if sigma_at_limit or gamma_at_limit:
            return Failure(f'sigma {1e3 * p[SIGMA]:.1f} / gamma {1e3 * p[GAMMA]:.1f} MeV at the upper limit')
        if fwhm < K0S_MIN_FWHM:
            return Failure(f'FWHM {1e3 * fwhm:.1f} MeV < {1e3 * K0S_MIN_FWHM:.0f}')
        if significance < min_significance:
            return Failure(f'signal {significance:.1f} sd < {min_significance:g}')
        if error >= max_error:
            return Failure(f'position error {1e3 * error:.1f} MeV > {1e3 * max_error:.0f}')

        return dict(shift=p[POSITION] - self.reference_mass, error=error,
                    width=fwhm, sigma=p[SIGMA], gamma=p[GAMMA])


class MesonPeak:
    """Full-sample and cell fits of one meson (MESON_PEAKS): signal yield x Voigtian +
    background. The full sample fixes sigma and gamma; each cell refits the yield, the
    position and the background with them."""

    def __init__(self, name, reference):
        self.spec = MESON_PEAKS[name]
        self.reference = reference                          # PDG mass
        self.n_background = {'exp': 2, 'swap_exp': 3, 'quad': 3}[self.spec['background']]

    def histogram(self, m):
        h, e = np.histogram(m, bins=self.spec['bins'], range=self.spec['range'])
        return 0.5 * (e[1:] + e[:-1]), h, e[1] - e[0]

    def in_range_mask(self, m):
        lo, hi = self.spec['range']
        return (m >= lo) & (m < hi)

    def in_range(self, m):
        """Number of candidates in the fit range."""
        return int(self.in_range_mask(m).sum())

    def function(self, bin_width):
        """Signal yield x Voigtian + the sample's background (MESON_PEAKS)."""
        m0, kind = self.spec['m0'], self.spec['background']

        def peak(x, n, mu, sigma, gamma, *bkg):
            signal = n * bin_width * voigt(x - mu, sigma, gamma)
            if kind == 'exp':
                a, b = bkg
                return signal + a * np.exp(-b * (x - m0))
            if kind == 'swap_exp':
                n_swap, a, b = bkg
                return signal + n_swap * np.exp(-0.5 * ((x - mu) / 0.170)**2) + a * np.exp(-b * (x - m0))
            a, b, c = bkg
            return signal + a + b * (x - m0) + c * (x - m0)**2
        return peak

    def bounds(self):
        """Signal yield, background normalisations >= 0; position inside the range;
        1 MeV <= sigma <= the largest plausible FWHM; 0 <= gamma <= GAMMA_MAX."""
        lo, hi = self.spec['range']
        lower = [0, lo, 0.001, 0.0] + [0.0] + [-np.inf] * (self.n_background - 1)
        upper = [np.inf, hi, self.spec['fwhm'][1], GAMMA_MAX] + [np.inf] * self.n_background
        if self.spec['background'] == 'swap_exp':
            lower[5] = 0.0                                  # the exponential's normalisation
        return np.array(lower), np.array(upper)

    def steps(self, p):
        """Starting steps: yield and normalisations sqrt(value), position and widths 0.1 MeV,
        background shape parameters 10% (at least 0.1)."""
        p = np.asarray(p, float)
        steps = np.maximum(0.1 * np.abs(p), 0.1)
        steps[YIELD] = math.sqrt(max(p[YIELD], 1)) + 1
        steps[[POSITION, SIGMA, GAMMA]] = 1e-4
        return steps

    def fit_full(self, m):
        """The full sample, all parameters free -> dict(p = parameters, n_in = candidates in
        the histogram) or a Failure with the reason. Its sigma and gamma are then fixed in
        the cells."""
        n = self.in_range(m)
        if n < MIN_FULL:
            return Failure(f'full sample n = {n} < {MIN_FULL}')
        x, h, w = self.histogram(m)

        # starting values from MESON_PEAKS, ('n', f) entries scaled by the candidates
        start = [s[1] * n if isinstance(s, tuple) else s for s in self.spec['start']]
        lower, upper = self.bounds()

        p, cov, significance = poisson_fit(self.function(w), x, h, start, lower, upper, self.steps(start))
        if p is None:
            return Failure('full-sample fit did not converge')
        if not cov[POSITION, POSITION] > 0:
            return Failure('full-sample fit: likelihood not positive definite')

        # a real peak: a plausible width (FWHM within the sample's range, gamma not at its
        # limit) and a significant signal
        fwhm = voigt_fwhm(p[SIGMA], p[GAMMA])
        lo_fwhm, hi_fwhm = self.spec['fwhm']
        gamma_at_limit = abs(p[GAMMA] - GAMMA_MAX) < 1e-3 * GAMMA_MAX
        if not lo_fwhm < fwhm < hi_fwhm or gamma_at_limit:
            return Failure(f'full sample: FWHM {1e3 * fwhm:.0f} MeV outside {1e3 * lo_fwhm:.0f}-{1e3 * hi_fwhm:.0f}')
        if significance < MESON_FULL_MIN_SIGNIFICANCE:
            return Failure(f'full sample: signal {significance:.1f} sd < {MESON_FULL_MIN_SIGNIFICANCE:g}')

        return dict(p=p, n_in=int(h.sum()))

    def fit_cell(self, m, full):
        """sigma and gamma fixed to the full sample -> dict(shift = position - PDG, error,
        width) or a Failure."""
        if self.in_range(m) < MIN_CELL:
            return Failure(f'n = {self.in_range(m)} < {MIN_CELL}')
        x, h, w = self.histogram(m)
        fixed = np.array(full['p'], float)
        free = [i for i in range(len(fixed)) if i not in (SIGMA, GAMMA)]
        shape = self.function(w)

        def with_shape_fixed(x, *p_free):
            """The shape with the free parameters inserted and the full-sample sigma, gamma."""
            p = fixed.copy(); p[free] = p_free
            return shape(x, *p)

        # start from the full-sample values, the yield and the background normalisations
        # scaled by the cell's share of the candidates
        share = h.sum() / max(full['n_in'], 1)
        scaled = {YIELD} | {i for i, s in enumerate(self.spec['start']) if isinstance(s, tuple)}
        start = np.array([fixed[i] * share if i in scaled else fixed[i] for i in free])

        # the position may move by at most max(25 MeV, 0.64 FWHM): a cell with little
        # signal cannot slide onto a background feature, while a real peak stays free
        lower_all, upper_all = self.bounds()
        limit = max(CELL_POSITION_LIMIT[0], CELL_POSITION_LIMIT[1] * voigt_fwhm(fixed[SIGMA], fixed[GAMMA]))
        lower_all[POSITION], upper_all[POSITION] = fixed[POSITION] - limit, fixed[POSITION] + limit
        lower, upper = lower_all[free], upper_all[free]
        k = free.index(POSITION)

        p, cov, significance = poisson_fit(with_shape_fixed, x, h, start, lower, upper, self.steps(fixed)[free])
        if p is None:
            return Failure('fit did not converge')
        if not cov[k, k] > 0:
            return Failure('likelihood not positive definite at the minimum')
        error = math.sqrt(cov[k, k])
        if at_bound(p[k], lower[k], upper[k]):
            return Failure(f'position at the +-{1e3 * limit:.0f} MeV bound')
        if significance < CELL_MIN_SIGNIFICANCE:
            return Failure(f'signal {significance:.1f} sd < {CELL_MIN_SIGNIFICANCE:g}')
        if not error < CELL_MAX_POSITION_ERROR:
            return Failure(f'position error {1e3 * error:.0f} MeV > {1e3 * CELL_MAX_POSITION_ERROR:.0f}')
        return dict(shift=p[k] - self.reference, error=error, width=voigt_fwhm(fixed[SIGMA], fixed[GAMMA]))


# ---------------------------------------------------------------- K0S calibration
class K0sCalibration:
    """The linearised chi2 fit of the K0S peak positions in the calibration bins."""

    def __init__(self, model, k0s):
        self.model, self.k0s = model, k0s
        self.bin = self.bin_index(model, k0s)
        regions = model.region(k0s.p3)
        self.n_endcap_pions = (regions != BARREL).sum(1)        # 0, 1 or 2 per candidate

    @staticmethod
    def bin_index(model, k0s):
        """Bin number of each candidate: decay-radius shell x (region, p_T) of each pion."""
        pt = np.hypot(k0s.p3[..., 0], k0s.p3[..., 1])
        regions = model.region(k0s.p3)
        endcap = (regions != BARREL).any(1)
        index = np.empty(len(pt), np.int64)
        # barrel candidates and those with an end-cap pion get separate binnings; the
        # offset keeps their bin numbers apart
        for mask, pt_edges, radius_edges, offset in ((~endcap, K0S_PT_EDGES, K0S_RADIUS_EDGES, 0),
                                                     (endcap, K0S_PT_EDGES_ENDCAP, K0S_RADIUS_EDGES_ENDCAP, 10**6)):
            n_pt = len(pt_edges) - 1
            pt_bin = [np.clip(np.digitize(pt[mask, i], pt_edges) - 1, 0, n_pt - 1) for i in (0, 1)]
            shell = np.clip(np.digitize(k0s.r0[mask, 0], radius_edges) - 1, 0, len(radius_edges) - 2)
            region_pair = regions[mask, 0] * len(REGIONS) + regions[mask, 1]
            # one number from (shell, region pair, p_T bin of pi+, p_T bin of pi-)
            index[mask] = offset + ((shell * len(REGIONS)**2 + region_pair) * n_pt + pt_bin[0]) * n_pt + pt_bin[1]
        return index

    def run(self):
        bins, counts = np.unique(self.bin, return_counts=True)
        bins = bins[counts >= K0S_MIN_CANDIDATES]
        print(f'{len(bins)} calibration bins; {(counts < K0S_MIN_CANDIDATES).sum()} with fewer than '
              f'{K0S_MIN_CANDIDATES} candidates are not used')

        # start from no correction (rho = 1: the full table loss); each iteration refits
        # the peaks with the current correction and solves the linearised shifts
        par = np.zeros(N_PARAMETERS)
        par[-1] = 1.0
        for iteration in range(K0S_ITERATIONS):
            m, D = self.model.mass_derivatives(self.k0s, par)
            shift, error, slope, failed = [], [], [], []
            for b in bins:
                in_bin = self.bin == b
                n_endcap = self.n_endcap_pions[in_bin][0]
                result = K0sPeak(n_endcap).fit(m[in_bin], K0S_MIN_SIGNIFICANCE, K0S_MAX_POSITION_ERROR)
                if isinstance(result, Failure):
                    failed.append((b, in_bin.sum(), result.why))
                    continue
                # how the peak moves with each parameter: the mean derivative of its core,
                # +- 1 FWHM around the fitted position
                position, err, width = PDG_MASS['K0S'] + result['shift'], result['error'], result['width']
                core = in_bin & (np.abs(m - position) < width)
                shift.append(position - PDG_MASS['K0S'])
                error.append(err)
                slope.append(D[core].mean(0))

            # linearised least squares: the step that moves every peak to the PDG mass,
            # y + X step = 0 weighted by the position errors
            y, e, X = np.array(shift), np.array(error), np.array(slope)
            W = X / e[:, None]
            cov = np.linalg.pinv(W.T @ W)
            step = cov @ (W.T @ (-y / e))
            par = par + step
            chi2 = float((((y + X @ step) / e)**2).sum())
            ndf = len(y) - N_PARAMETERS
            print(f'iteration {iteration}: {len(y)} bins, chi2/ndf {chi2:.1f}/{ndf}, {len(failed)} bins without a usable peak')

        for b, n, why in failed:
            print(f'   no usable peak: bin {b}, n = {n}: {why}')
        return dict(values=par.tolist(), covariance_fit=cov.tolist(), chi2=chi2, ndf=ndf, n_bins=len(y))


# ---------------------------------------------------------------- closure
class Closure:
    """CMS-style closure in the barrel: residual A' and M' per daughter-p_T bin and
    sample, and the closure systematic fitted to their scatter."""

    def __init__(self, model, par, cov):
        self.model, self.par, self.cov = model, np.array(par), np.array(cov)
        self.edges = CLOSURE_PT_EDGES
        # each bin is represented by its geometric centre
        self.pt_centre = np.sqrt(np.array(self.edges[:-1]) * np.array(self.edges[1:]))

    def calibration_band(self):
        """Calibration (statistical) uncertainty of (A', M') per p_T bin, for a barrel track
        at the bin's centre: A' takes the charge-even A - eps g / p_T, M' the charge-odd M."""
        # J: d(A', M' of each bin) / d(calibration parameters); band = sqrt(diag(J C J^T))
        J = np.zeros((2 * len(self.pt_centre), N_PARAMETERS))
        a, eps, M = 3 * BARREL, 3 * BARREL + 1, 3 * BARREL + 2
        for i, pt in enumerate(self.pt_centre):
            J[2 * i, a] = 1.0
            J[2 * i, eps] = -self.model.dedx_shape(pt / TYPICAL_SIN_THETA) / pt
            J[2 * i + 1, M] = 1.0
        sigma = np.sqrt(np.diag(J @ self.cov @ J.T))
        return sigma[0::2], sigma[1::2]

    def residual_derivatives(self, sample):
        """Mass, and d(mass)/d(A') and d(mass)/d(M') on top of the corrected momenta."""
        base = sample.mass(sample.p3)
        out = np.zeros((len(base), 2))
        # a residual A' (or M') is the same small extra A (or M) in every region; rho = 0
        for column, j in enumerate((0, 2)):                     # A, then M
            par = np.zeros(N_PARAMETERS)
            for r in range(len(REGIONS)):
                par[3 * r + j] = DERIVATIVE_STEP[j]
            out[:, column] = (sample.mass(self.model.correct(sample.p3, sample.q, sample.r0, par)) - base) / DERIVATIVE_STEP[j]
        return base, out

    def sample_residuals(self, sample, binning):
        """Per p_T bin: dict(A, dA, M, dM, cov_AM) from the two charge halves, or a Failure."""
        m, D = self.residual_derivatives(sample)

        # mesons: the full-sample fit fixes the line shape (sigma, gamma) for the cells;
        # without it, every bin fails with its reason. K0S cells use the calibration's
        # own peak fit, in the barrel window.
        meson = None
        if sample.name == 'K0S':
            k0s_peak = K0sPeak(0)
        else:
            meson = MesonPeak(sample.name, sample.reference_mass)
            full = meson.fit_full(m)
            if isinstance(full, Failure):
                return [full] * len(self.pt_centre)

        pt = np.hypot(sample.p3[..., 0], sample.p3[..., 1])
        x = pt.min(1) if binning == 'sublead' else pt.max(1)
        positive_leads = (sample.q * pt).sum(1) > 0          # the positive tracks carry more p_T

        rows = []
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            in_bin = (x >= lo) & (x < hi)
            halves = []
            for label, half in (('q+ leading', in_bin & positive_leads), ('q- leading', in_bin & ~positive_leads)):
                if meson is None:
                    cell = (Failure(f'n = {half.sum()} < {MIN_CELL}') if half.sum() < MIN_CELL else
                            k0s_peak.fit(m[half], CELL_MIN_SIGNIFICANCE, K0S_CELL_MAX_POSITION_ERROR))
                else:
                    cell = meson.fit_cell(m[half], full)
                if not isinstance(cell, Failure):
                    # d(peak position)/d(A', M'): the mean over the peak core (+-1 FWHM), or
                    # over the cell's candidates in the fit range when the core has few
                    core = half & (np.abs(m - (sample.reference_mass + cell['shift'])) < cell['width'])
                    fallback = half if meson is None else half & meson.in_range_mask(m)
                    d = D[core if core.sum() > 20 else fallback].mean(0)
                    cell.update(dA=d[0], dM=d[1])
                halves.append((label, cell))
            rows.append(self.solve_halves(halves))
        return rows

    @staticmethod
    def solve_halves(halves):
        """The two cells' peak shifts -> A', M' and their covariance, or a Failure."""
        bad = [f'{label}: {cell.why}' for label, cell in halves if isinstance(cell, Failure)]
        if bad:
            return Failure('; '.join(bad))
        # two equations, shift = dA * A' + dM * M', one per charge half. M' enters the
        # halves with opposite signs (it is charge-odd), A' with the same sign.
        (_, a), (_, b) = halves
        X = np.array([[a['dA'], a['dM']], [b['dA'], b['dM']]])
        try:
            X_inverse = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            return Failure('charge halves not separable')
        solution = X_inverse @ np.array([a['shift'], b['shift']])
        cov = X_inverse @ np.diag([a['error']**2, b['error']**2]) @ X_inverse.T
        return dict(A=float(solution[0]), dA=float(math.sqrt(cov[0, 0])),
                    M=float(solution[1]), dM=float(math.sqrt(cov[1, 1])), cov_AM=float(cov[0, 1]))

    def systematic(self, residuals, key, form):
        """Closure systematic by maximum likelihood over the closure points (key 'A' or 'M')
        of all samples, each with variance stat^2 + band^2 + sigma(p_T)^2 at its bin's p_T:
            form 'a+bpt'  sigma = sqrt(a^2 + (b p_T)^2)
            form 'const'  sigma = a
        One fit to the points of both binnings: they place each candidate at the p_T of its
        subleading or of its leading daughter, and so cover complementary p_T ranges (the
        subleading binning the low p_T, the leading one the high p_T). For information,
        the fit of each binning alone is also returned."""
        from scipy.optimize import minimize
        band = self.calibration_band()[0 if key == 'A' else 1]
        sigma_of = ((lambda p, pt: np.hypot(p[0], p[1] * pt)) if form == 'a+bpt'
                    else (lambda p, pt: abs(p[0]) + 0 * pt))

        def fit(binnings):
            # every closure point with a result: residual y, its known error e (statistics
            # of the point (+) calibration band of its bin) and its bin's p_T
            y, e, pt = [], [], []
            for rows in residuals.values():
                for binning in binnings:
                    for i, r in enumerate(rows[binning]):
                        if not isinstance(r, Failure):
                            y.append(r[key])
                            e.append(math.hypot(r['d' + key], band[i]))
                            pt.append(self.pt_centre[i])
            y, e, pt = map(np.array, (y, e, pt))

            def nll(p):
                # Gaussian -2 log likelihood of residuals with mean 0 and variance
                # e^2 + sigma^2; the log term stops sigma from growing without limit
                variance = e**2 + sigma_of(p, pt)**2
                return float((y**2 / variance + np.log(variance)).sum())

            start = [np.std(y) / 3, np.std(y) / 30] if form == 'a+bpt' else [np.std(y) / 3]
            p = np.abs(minimize(nll, start, method='Nelder-Mead',
                                options=dict(xatol=1e-9, fatol=1e-9, maxiter=20000)).x)
            # chi2 of the points without and with the systematic, as a check of the fit
            return dict(par=p.tolist(), n=len(y),
                        chi2_stat=float((y**2 / e**2).sum()),
                        chi2_syst=float((y**2 / (e**2 + sigma_of(p, pt)**2)).sum()))

        result = dict(form=form, **fit(CLOSURE_BINNINGS))
        result['per_binning'] = {b: fit((b,)) for b in CLOSURE_BINNINGS}
        return result

    def run(self, samples):
        """Residuals of every sample in the barrel and the systematic of A' and M'."""
        residuals = {}
        for name, sample in samples.items():
            # candidates with every daughter in the barrel, with the calibration applied
            barrel = sample.subset((self.model.region(sample.p3) == BARREL).all(1)).corrected(self.model, self.par)
            residuals[name] = {b: self.sample_residuals(barrel, b) for b in CLOSURE_BINNINGS}
        return residuals, self.systematic(residuals, 'A', 'a+bpt'), self.systematic(residuals, 'M', 'const')


# ---------------------------------------------------------------- the calibration file
def tables(model):
    """The track-fit tables the correction needs, written into the calibration file so
    that momentumScale_makeCorrection.py needs no physics input of its own."""
    return dict(cylinder_radius_mm=CYLINDER_RADIUS, cylinder_eloss_GeV=MATERIAL[model.dataset],
                dedx_log10_bg=DEDX_LOG10_BG, dedx_mip=DEDX_MIP, m_pi=M_PI)


def command_fit(args):
    dataset, samples = load_samples(args.skim)
    model = Model(dataset)
    print(f'{dataset}: {len(samples["K0S"].p3)} K0S candidates')
    result = K0sCalibration(model, samples['K0S']).run()

    covariance = result['covariance_fit']
    calibration = dict(
        dataset=dataset,
        model=dict(regions=list(REGIONS), cos_theta_edges=list(REGION_COS_THETA_EDGES), parameters=PARAMETERS,
                   units='A, rho: 1; eps: GeV; M: 1/GeV'),
        fit=dict(values=result['values'], covariance=covariance,
                 chi2=result['chi2'], ndf=result['ndf'], n_bins=result['n_bins']),
        tables=tables(model))
    json.dump(calibration, open(args.output, 'w'), indent=1)

    # summary
    values, errors = np.array(result['values']), np.sqrt(np.diag(covariance))
    print(f'{dataset}: A (1e-3), eps (MeV), M (1e-4 GeV^-1); chi2/ndf {result["chi2"]:.1f}/{result["ndf"]}')
    for r, name in enumerate(REGIONS):
        a, eps, M = values[3 * r:3 * r + 3]
        da, deps, dM = errors[3 * r:3 * r + 3]
        print(f'   {name:8s} A {1e3 * a:+.3f} +- {1e3 * da:.3f}   eps {1e3 * eps:+.2f} +- {1e3 * deps:.2f}'
              f'   M {1e4 * M:+.3f} +- {1e4 * dM:.3f}')
    print(f'   rho {values[-1]:.3f} +- {errors[-1]:.3f}')
    print(f'wrote {args.output}')


def command_closure(args):
    dataset, samples = load_samples(args.skim)
    calibration = json.load(open(args.calibration))
    if calibration['dataset'] != dataset:
        sys.exit(f'the skim is {dataset}, the calibration {calibration["dataset"]}')
    model = Model(dataset)
    fit = calibration['fit']
    cov = np.array(fit['covariance'])
    residuals, syst_A, syst_M = Closure(model, fit['values'], cov).run(samples)

    # print every bin of every sample, failures with their reason
    for binning in CLOSURE_BINNINGS:
        print(f"{dataset} barrel, binned in the {'subleading' if binning == 'sublead' else 'leading'} daughter p_T:"
              " A' (1e-3), M' (1e-4 GeV^-1)")
        for i, (lo, hi) in enumerate(zip(CLOSURE_PT_EDGES[:-1], CLOSURE_PT_EDGES[1:])):
            print(f'   p_T [{lo}, {hi})')
            for name, rows in residuals.items():
                r = rows[binning][i]
                text = (repr(r) if isinstance(r, Failure)
                        else f"A' {1e3 * r['A']:+.2f} +- {1e3 * r['dA']:.2f}   M' {1e4 * r['M']:+.1f} +- {1e4 * r['dM']:.1f}")
                print(f'      {name:11s} {text}')
    # the systematic from both binnings together, and from each alone for comparison
    for key, syst, unit, scale in (('A', syst_A, '1e-3', 1e3), ('M', syst_M, '1e-4 GeV^-1', 1e4)):
        for label, b in [('both binnings', syst)] + [(f'{k} only', v) for k, v in syst['per_binning'].items()]:
            print(f"{dataset} {key}' systematic ({syst['form']}, {label}): {[round(scale * x, 3) for x in b['par']]} ({unit}),"
                  f" chi2 {b['chi2_stat']:.1f} -> {b['chi2_syst']:.1f} / {b['n']}")

    # uncertainties per region: the closure systematic in the barrel only; sigma_M combines
    # the systematic with the statistical error of M
    per_region = []
    for r, name in enumerate(REGIONS):
        barrel = r == BARREL
        a, b = syst_A['par'] if barrel else (0.0, 0.0)
        syst_m = syst_M['par'][0] if barrel else 0.0
        per_region.append(dict(region=name, closure_sigma_A_a=a, closure_sigma_A_b=b,
                               sigma_M=math.hypot(syst_m, math.sqrt(cov[3 * r + 2, 3 * r + 2]))))
    calibration['uncertainties'] = dict(
        per_region=per_region,
        note='closure systematic in the barrel only; sigma_M = statistics (+) closure systematic')
    # the closure points themselves, for the record (failures as {'why': reason})
    calibration['closure'] = {name: {b: [r.__dict__ if isinstance(r, Failure) else r for r in rows[b]]
                                     for b in CLOSURE_BINNINGS} for name, rows in residuals.items()}
    calibration['closure_pt_edges'] = list(CLOSURE_PT_EDGES)
    json.dump(calibration, open(args.calibration, 'w'), indent=1)
    print(f'updated {args.calibration}')


def command_check(args):
    """The correctionlib file against the model: nominal s on the skimmed K0S tracks, with
    their decay radius and as prompt tracks; and the median size of the variations."""
    import correctionlib
    dataset, samples = load_samples(args.skim)
    calibration = json.load(open(args.calibration))
    model = Model(dataset)
    par = np.array(calibration['fit']['values'])
    correction = correctionlib.CorrectionSet.from_file(args.correction)['track_momentum_scale']

    # every K0S daughter as one track, in the inputs of the correction
    k0s = samples['K0S']
    p3, q = k0s.p3.reshape(-1, 3), k0s.q.reshape(-1)
    p = np.linalg.norm(p3, axis=1)
    theta = np.arccos(p3[:, 2] / p)
    pt = p * np.sin(theta)

    # the nominal correction must equal the model, displaced (r0 > 0) and prompt (r0 = 0)
    for label, r0 in (('displaced', k0s.r0.reshape(-1)), ('prompt', np.zeros(len(p)))):
        expected = np.linalg.norm(model.correct(p3, q, r0, par), axis=1)
        s = correction.evaluate(dataset, 'nominal', r0, theta, p, q)
        print(f'{dataset} {label:9s}: max |p_model / (s p) - 1| = {np.abs(expected / (s * p) - 1).max():.1e}')

    # typical size of the up variations relative to nominal, for prompt tracks
    zero = np.zeros(len(p))
    nominal = correction.evaluate(dataset, 'nominal', zero, theta, p, q)
    regions = {'barrel': np.abs(np.cos(theta)) < 0.77, 'end caps': np.abs(np.cos(theta)) >= 0.77}
    for variation in ('scale_up', 'alignment_up'):
        shift = correction.evaluate(dataset, variation, zero, theta, p, q) / nominal - 1
        if variation == 'alignment_up':
            shift = shift * q                                     # charge-odd: the size for both charges
        for name, in_region in regions.items():
            medians = [np.median(shift[in_region & (pt >= lo) & (pt < hi)]) * 100
                       for lo, hi in ((0, 1), (4, 8), (15, np.inf))]
            print(f'   {variation:12s} {name:8s}: p_T < 1 {medians[0]:.3f}%, 4-8 {medians[1]:.3f}%, > 15 {medians[2]:.3f}%')


def main():
    ap = argparse.ArgumentParser(description='Momentum-scale calibration, closure and check.')
    sub = ap.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit', help='K0S calibration -> calibration file')
    fit.add_argument('--skim', required=True, help='output of momentumScale_skim.py')
    fit.add_argument('--output', required=True, help='calibration_<dataset>.json')
    closure = sub.add_parser('closure', help='closure and its systematic -> added to the calibration file')
    closure.add_argument('--skim', required=True)
    closure.add_argument('--calibration', required=True)
    check = sub.add_parser('check', help='correctionlib file against the model')
    check.add_argument('--skim', required=True)
    check.add_argument('--calibration', required=True)
    check.add_argument('--correction', required=True, help='correctionlib file')
    args = ap.parse_args()
    {'fit': command_fit, 'closure': command_closure, 'check': command_check}[args.command](args)


if __name__ == '__main__':
    main()
