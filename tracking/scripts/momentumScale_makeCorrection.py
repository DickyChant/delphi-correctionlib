"""Build the correctionlib file of the charged-track momentum-scale correction.

    python3 momentumScale_makeCorrection.py --output momentumScale_9495.json.gz \
        --calibration calibration_94C2.json calibration_95D2.json

Input: one calibration file per DELPHI processing (94C2, 95D2, ...), written by the fit and
closure scripts. Each holds the 10 fitted parameters and their covariance, the
uncertainties per region (closure systematic, total sigma_M), and the track-fit tables
(material cylinders and DELPHI's dE/dx shape FKDEDX). This script only arranges those
numbers into the correctionlib tree; all physics choices are made in the calibration files.

Output: one correction, 'track_momentum_scale', with inputs (dataset, variation, r0, theta,
p, q). It returns the factor s for the stored momentum: p_corrected = s * p, direction
kept. The model, per track:

    g   = FKDEDX(log10(p / m_pi))           dE/dx shape for the pion mass
    p1  = p - rho * F(r0) * g / sin(theta)  remove the energy loss of the material inside r0
    pt1 = p1 * sin(theta)
    s   = (p1 / p) * (1 + A - eps * g / pt1 + q * M * pt1)

with (A, eps, M) taken from the region of the track (backward, barrel, forward).

Variations, one standard deviation, all sources added in quadrature per track:
    scale_up/down      charge-even. Calibration statistics of A, eps, rho (from the
                       covariance) plus the closure systematic sqrt(a^2 + (b * pt1)^2).
    alignment_up/down  charge-odd: the shift follows the sign of q, pt1 * sigma_M.
"""
import argparse, json, gzip
import numpy as np
import correctionlib.schemav2 as cs

# Version of the correction in the output file. Increase it with every change of the
# correction (new calibration, model or variations), in the same commit.
CORRECTION_VERSION = 1

# variation name -> (sign of the scale shift, sign of the alignment shift)
VARIATIONS = {
    'nominal':        (0, 0),
    'scale_up':       (+1, 0),
    'scale_down':     (-1, 0),
    'alignment_up':   (0, +1),
    'alignment_down': (0, -1),
}

# ---------------------------------------------------------------- the leaf formula
# correctionlib formulas use the TFormula syntax: the inputs are called x, y, z (here p,
# theta, q) and the constants of each leaf [0], [1], ... There are no local definitions,
# so the expression is assembled below from named pieces. LEAF_PARAMETERS names the
# constants, in the order they are stored in each leaf.
LEAF_PARAMETERS = [
    'g0', 'g1',               # dE/dx shape in this p bin: g = g0 + g1 * log10(p / m_pi)
    'rhoF',                   # rho * F(r0) for this radius shell (GeV)
    'A', 'eps', 'M',          # correction parameters of this region
    'scale_sign',             # 0 nominal, +1 / -1 for scale_up / scale_down
    'closure_a', 'closure_b', # closure systematic of the region: sqrt(a^2 + (b * pt1)^2)
    'alignment_sign',         # 0 nominal, +1 / -1 for alignment_up / alignment_down
    'sigma_M',                # total uncertainty of M in the region
    'var_A', 'var_eps', 'var_rho', 'cov_A_eps', 'cov_A_rho', 'cov_eps_rho',  # calibration covariance
    'F',                      # F(r0) alone, needed for the rho term of the statistical uncertainty
]
P = {name: f'[{i}]' for i, name in enumerate(LEAF_PARAMETERS)}


def leaf_formula(m_pi):
    """The expression of s in TFormula syntax, built from named sub-expressions."""
    p, theta, q = 'x', 'y', 'z'
    sin_theta = f'sin({theta})'
    g = f"({P['g0']}+{P['g1']}*log10({p}/{m_pi:.5f}))"
    p1 = f"({p}-{P['rhoF']}*{g}/{sin_theta})"
    pt1 = f"({p1}*{sin_theta})"

    # nominal factor
    bias = f"({P['A']}-{P['eps']}*{g}/{pt1}+{q}*{P['M']}*{pt1})"
    nominal = f"({p1}/{p})*(1+{bias})"

    # statistical variance of the charge-even part, J C J^T, with J the derivatives of
    # ln s with respect to (A, eps, rho): d/dA = 1, d/deps = -g / pt1, d/drho = -F g / (p sin theta)
    d_eps = f"(-{g}/{pt1})"
    d_rho = f"(-{P['F']}*{g}/({sin_theta}*{p}))"
    var_stat = (f"({P['var_A']}+{d_eps}*{d_eps}*{P['var_eps']}+{d_rho}*{d_rho}*{P['var_rho']}"
                f"+2*{d_eps}*{P['cov_A_eps']}+2*{d_rho}*{P['cov_A_rho']}+2*{d_eps}*{d_rho}*{P['cov_eps_rho']})")

    # one-standard-deviation shifts of s / s_nominal
    sigma_scale = f"sqrt({P['closure_a']}*{P['closure_a']}+{P['closure_b']}*{P['closure_b']}*{pt1}*{pt1}+{var_stat})"
    sigma_alignment = f"{pt1}*{P['sigma_M']}"
    return f"{nominal}*(1+{P['scale_sign']}*{sigma_scale}+{P['alignment_sign']}*{q}*{sigma_alignment})"


# ---------------------------------------------------------------- per-dataset tree
def dedx_bins(tables):
    """FKDEDX as a function of |p|: bin edges at the table nodes and (g0, g1) per bin, so
    that g0 + g1 log10(p/m_pi) is the linear interpolation between the nodes. Below the
    first and above the last node the value is held constant (as numpy.interp does)."""
    x = np.array(tables['dedx_log10_bg'])        # log10(beta gamma) of the nodes
    y = np.array(tables['dedx_mip'])             # dE/dx in MIP units
    slope = np.diff(y) / np.diff(x)
    lines = [(y[0], 0.0)]                                             # below the first node
    lines += [(y[i] - slope[i] * x[i], slope[i]) for i in range(len(slope))]
    lines += [(y[-1], 0.0)]                                           # above the last node
    edges = [0.0] + [float(tables['m_pi'] * 10**v) for v in x] + ['inf']
    return edges, [(float(a), float(b)) for a, b in lines]


def dataset_node(cal):
    """variation -> r0 shell -> theta region -> p bin -> leaf, for one dataset."""
    tables, fit, model = cal['tables'], cal['fit'], cal['model']
    values = np.array(fit['values'])            # (A, eps, M) backward, barrel, forward, then rho
    rho = values[-1]
    cov = np.array(fit['covariance'])
    unc = {u['region']: u for u in cal['uncertainties']['per_region']}

    # radius shells between the material cylinders, and F = summed energy loss of the
    # cylinders inside each shell
    radius = np.array(tables['cylinder_radius_mm'])
    eloss = np.array(tables['cylinder_eloss_GeV'])
    r_edges = [0.0] + [float(r) for r in radius] + ['inf']
    F_per_shell = [float(eloss[:i].sum()) for i in range(len(radius) + 1)]

    # regions in theta: the model is defined in cos(theta) (|cos theta| < 0.77 = barrel);
    # theta increases from the forward to the backward end
    theta0 = float(np.arccos(model['cos_theta_edges'][2]))
    theta_edges = [0.0, theta0, float(np.pi - theta0), float(np.pi)]
    BACKWARD, BARREL, FORWARD = 0, 1, 2          # region index in the parameter vector (model['regions'])
    regions_in_theta_order = (FORWARD, BARREL, BACKWARD)

    p_edges, dedx_lines = dedx_bins(tables)

    def leaves(F, region, scale_sign, alignment_sign):
        """One leaf per p bin; only g0, g1 change between them."""
        iA, ieps, iM, irho = 3 * region, 3 * region + 1, 3 * region + 2, len(values) - 1
        u = unc[model['regions'][region]]
        constants = dict(
            rhoF=rho * F,
            A=values[iA], eps=values[ieps], M=values[iM],
            scale_sign=scale_sign,
            closure_a=u['closure_sigma_A_a'], closure_b=u['closure_sigma_A_b'],
            alignment_sign=alignment_sign,
            sigma_M=u['sigma_M'],
            var_A=cov[iA, iA], var_eps=cov[ieps, ieps], var_rho=cov[irho, irho],
            cov_A_eps=cov[iA, ieps], cov_A_rho=cov[iA, irho], cov_eps_rho=cov[ieps, irho],
            F=F,
        )
        out = []
        for g0, g1 in dedx_lines:
            constants.update(g0=g0, g1=g1)
            out.append(cs.FormulaRef(nodetype='formularef', index=0,
                                     parameters=[float(constants[name]) for name in LEAF_PARAMETERS]))
        return out

    def tree(scale_sign, alignment_sign):
        shells = []
        for F in F_per_shell:
            per_region = [cs.Binning(nodetype='binning', input='p', edges=p_edges, flow='clamp',
                                     content=leaves(F, region, scale_sign, alignment_sign))
                          for region in regions_in_theta_order]
            shells.append(cs.Binning(nodetype='binning', input='theta', edges=theta_edges, flow='clamp',
                                     content=per_region))
        return cs.Binning(nodetype='binning', input='r0', edges=r_edges, flow='clamp', content=shells)

    return cs.Category(nodetype='category', input='variation',
                       content=[cs.CategoryItem(key=name, value=tree(*signs)) for name, signs in VARIATIONS.items()])


# ---------------------------------------------------------------- the correction file
def main():
    ap = argparse.ArgumentParser(description='Bake the momentum-scale correction into a correctionlib file.')
    ap.add_argument('--output', required=True, help='output file (.json or .json.gz)')
    ap.add_argument('--calibration', required=True, nargs='+', help='calibration_<dataset>.json, one per processing')
    args = ap.parse_args()
    cals = [json.load(open(f)) for f in args.calibration]
    datasets = [c['dataset'] for c in cals]
    correction = cs.Correction(
        name='track_momentum_scale', version=CORRECTION_VERSION,
        description=('Momentum-scale correction for charged tracks of the DELPHI processings ' + ', '.join(datasets)
                     + ', calibrated on K0S in data: p_corrected = s * p, direction unchanged. Data only; do not apply '
                     'to simulation. Variations are one standard deviation: scale_* is charge-even, alignment_* '
                     'charge-odd (opposite for the two charges). End caps (|cos theta| > 0.77): statistical '
                     'uncertainty only.'),
        inputs=[
            cs.Variable(name='dataset', type='string', description='DELPHI processing: ' + ' or '.join(datasets)),
            cs.Variable(name='variation', type='string', description=', '.join(VARIATIONS)),
            cs.Variable(name='r0', type='real', description='production radius of the track in the xy plane (mm); 0 for tracks from the primary vertex'),
            cs.Variable(name='theta', type='real', description='polar angle of the stored momentum (rad)'),
            cs.Variable(name='p', type='real', description='|p| of the stored momentum (GeV)'),
            cs.Variable(name='q', type='real', description='charge (+1 or -1)'),
        ],
        output=cs.Variable(name='s', type='real', description='factor for the momentum vector'),
        generic_formulas=[cs.Formula(nodetype='formula', parser='TFormula', variables=['p', 'theta', 'q'],
                                     expression=leaf_formula(cals[0]['tables']['m_pi']))],
        data=cs.Category(nodetype='category', input='dataset',
                         content=[cs.CategoryItem(key=c['dataset'], value=dataset_node(c)) for c in cals]),
    )
    cset = cs.CorrectionSet(schema_version=2, description='DELPHI open data: charged-track momentum scale',
                            corrections=[correction])
    text = cset.model_dump_json(exclude_unset=True)
    if args.output.endswith('.gz'):
        with gzip.open(args.output, 'wt') as f:
            f.write(text)
    else:
        open(args.output, 'w').write(text)
    print('wrote', args.output)


if __name__ == '__main__':
    main()
