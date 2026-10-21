"""Skim the calibration and closure samples of the momentum-scale correction from
DELPHI EDM4hep files.

    python3 momentumScale_skim.py --dataset 94C2 --output skim_94C2.npz \
        --jobs 8 /path/to/*.edm4hep.root
    python3 momentumScale_skim.py --dataset 95D2 --output skim_95D2.npz \
        --jobs 8 --filelist files_95D2.txt

Selecting the input files is the user's job: give only files of one processing
(94C2, 95D2, ...) and only runs with the VD and the TPC fully working. The script
applies no run selection.

Events: hadronic Z decays, >= 5 charged particles and charged energy > 12% of sqrt(s)
(all charged particles taken as pions).

Samples. Every candidate stores the momenta of its daughters as written in the file
(<sample>_p3, shape (n, daughters, 3), GeV) and their charges (<sample>_q). Mass
windows are wide; the final cuts are applied by momentumScale_calibrate.py.
  k0s    DELPHI V0s (sDST_V0_V0Candidates) with any tag (k0s_tag, KRV0(5)), daughters
         positive first, momenta turned to the V0 vertex; k0s_r0 = vertex radius in xy (mm).
  jpsi   mu+ mu- pairs: |p| > 1.5 GeV and a muon tag of sDST_MUID_MuonID
         (loose, standard or tight).
  dstar  D*+ -> D0 pi+, D0 -> K- pi+: daughters (K, pi, soft pi); dstar_dm = M(K pi pi) - M(K pi).
  d0     D0 -> K- pi+, daughters (K, pi).
  dplus  D+ -> K- pi+ pi+, daughters (K, pi, pi).
Charge conjugates are included. The D selections are DELPHI's (NIM A 378 (1996) 57 and
the charm papers cited there), with one change: the kaon of the D0 and D+ is identified
by DELPHI's TPC dE/dx kaon tag only (sDST_RPRODE_DedxTags), not by the RICH, so that
the samples need only the VD and the TPC.

Output: one .npz with the samples above, plus the dataset name, the input file names
and the event counts.

Structure: Event finds the candidates of one event, using a TrackTable of its charged
tracks and the VertexFitter for the D vertices; Samples collects the candidates of
all files and writes them.
"""
import argparse, collections, math, multiprocessing, sys
from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------- constants
M_PI, M_K, M_MU = 0.13957, 0.49368, 0.1056584      # GeV

# DetectorMask bits (sDST_MAIN_Particles_DetectorMask)
VD_BIT = 1 << 3            # vertex detector
OD_BIT = 1 << 8            # outer detector

# sDST_MUID_MuonID parameter 0 is a bit mask; these bits are the loose, standard and
# tight muon tags
MUON_TAG_BITS = 0b1110

# TrackState location of the track parameters at the interaction point
AT_IP = 1

# hadronic event selection
MIN_CHARGED = 5
MIN_CHARGED_ENERGY_FRACTION = 0.12

# mass windows of the stored candidates (GeV)
JPSI_WINDOW = (2.0, 4.0)
D0_WINDOW = (1.6, 2.2)
DPLUS_WINDOW = (1.7, 2.05)
DSTAR_KPI_WINDOW = (1.6, 2.1)
DSTAR_MAX_DM = 0.170

# D* helicity cut of DELPHI: cos(theta_K) > limit, for x_E(D*) below each upper edge
DSTAR_HELICITY_CUT = ((0.25, -0.6), (0.50, -0.8), (math.inf, -0.9))

# Vertex fit. The fitter (F. Bedeschi's, see VertexFitter) uses the Delphes helix
# parameters (D, phi0, C, z0, cot theta). They equal EDM4hep's (D0, phi, omega, Z0,
# tanLambda) except for the curvature, C = -omega / 2: the conversion multiplies the
# third parameter, and its row and column of the covariance, by -1/2. The fit is pure
# helix geometry (mm); it needs no magnetic field.
TO_FITTER_CONVENTION = np.array([1.0, 1.0, -0.5, 1.0, 1.0])
VERTEX_FIT_MAX_ITERATIONS = 100
VERTEX_FIT_TOLERANCE = 1e-12        # stop when the vertex moves by less than this, in chi2 units

# per sample: number of daughters, and the per-candidate columns beyond momenta and charges
SAMPLES = {
    'k0s':   {'daughters': 2, 'extra': {'r0': np.float32, 'tag': np.int32}},
    'jpsi':  {'daughters': 2, 'extra': {}},
    'dstar': {'daughters': 3, 'extra': {'dm': np.float32}},
    'd0':    {'daughters': 2, 'extra': {}},
    'dplus': {'daughters': 3, 'extra': {}},
}


# ---------------------------------------------------------------- kinematics
def four_vector(p3, m):
    """(E, px, py, pz) for momentum p3 (..., 3) and mass m."""
    p3 = np.asarray(p3, float)
    energy = np.sqrt((p3**2).sum(-1, keepdims=True) + m**2)
    return np.concatenate([energy, p3], -1)


def invariant_mass(p4):
    return np.sqrt(np.maximum(p4[..., 0]**2 - (p4[..., 1:]**2).sum(-1), 0))


def kaon_helicity_cosine(kaon_p4, d0_p4):
    """cos of the angle between the kaon in the D0 rest frame and the D0 flight direction."""
    beta = d0_p4[:, 1:] / d0_p4[:, :1]
    beta2 = (beta**2).sum(1)
    gamma = 1 / np.sqrt(1 - beta2)
    beta_dot_p = (beta * kaon_p4[:, 1:]).sum(1)
    # Lorentz boost of the kaon momentum into the D0 rest frame
    kaon_rest = kaon_p4[:, 1:] + ((gamma - 1) * beta_dot_p / beta2 - gamma * kaon_p4[:, 0])[:, None] * beta
    d0_direction = d0_p4[:, 1:] / np.linalg.norm(d0_p4[:, 1:], axis=1, keepdims=True)
    return (kaon_rest * d0_direction).sum(1) / np.linalg.norm(kaon_rest, axis=1)


# ---------------------------------------------------------------- tracks
def state_at_ip(particle):
    """The track parameters of a particle at the interaction point, or None."""
    tracks = particle.getTracks()
    if not len(tracks):
        return None
    states = [s for s in tracks[0].getTrackStates() if s.location == AT_IP]
    return states[0] if states else None


def covariance_matrix(lower_triangle):
    """5x5 helix covariance from EDM4hep's 15 stored values: the lower triangle, row by
    row, i.e. (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ..."""
    cov = np.zeros((5, 5))
    n = 0
    for i in range(5):
        for j in range(i + 1):
            cov[i, j] = cov[j, i] = lower_triangle[n]
            n += 1
    return cov


class TrackTable:
    """The charged tracks of one event as arrays, one entry per particle with track
    parameters at the interaction point:
        p3 (GeV), q, mask (DetectorMask), length (cm), d0_pv, z0_pv (impact parameters
        to the primary vertex, mm), helix (D0, phi, omega, Z0, tanLambda), helix_cov
        (5x5), kaon (DELPHI's TPC dE/dx kaon tag, loose or better, with good quality),
        good (track quality, see quality()) and p (|p|)."""

    def __init__(self, frame, particles, charged):
        d0_pv = frame.get('sDST_PV_Tracks_d0PV')                  # indexed by track
        z0_pv = frame.get('sDST_PV_Tracks_z0PV')
        length = frame.get('sDST_MAIN_Particles_TrackLength')     # indexed by particle
        mask = frame.get('sDST_MAIN_Particles_DetectorMask')
        kaon_tag = self.kaon_tags(frame)

        columns = collections.defaultdict(list)
        for k in charged:
            s = state_at_ip(particles[k])
            if s is None:
                continue
            track = particles[k].getTracks()[0].getObjectID().index
            p = particles[k].getMomentum()
            columns['p3'].append([p.x, p.y, p.z])
            columns['q'].append(particles[k].getCharge())
            columns['mask'].append(mask[k] & 0xffffffff)
            columns['length'].append(length[k])
            columns['d0_pv'].append(d0_pv[track])
            columns['z0_pv'].append(z0_pv[track])
            columns['helix'].append([s.D0, s.phi, s.omega, s.Z0, s.tanLambda])
            columns['helix_cov'].append(covariance_matrix(s.covMatrix.values))
            columns['kaon'].append(kaon_tag.get(k, False))
        for name, values in columns.items():
            setattr(self, name, np.array(values))
        self.n = len(columns['q'])
        if self.n:
            self.good, self.p = self.quality()

    @staticmethod
    def kaon_tags(frame):
        """particle index -> DELPHI's TPC dE/dx kaon tag is loose or better with good quality.
        The tag table: parameter 1 = kaon tag (0 no, 1/2/3 loose/standard/tight),
        parameter 5 = quality flag (1 = good). Its name changed between processings."""
        available = set(frame.getAvailableCollections())
        name = next((n for n in ('sDST_RPRODE_DedxTags', 'sDST_RPRODO_DedxTags') if n in available), None)
        tags = {}
        for o in (frame.get(name) if name else []):
            if o.getParticle().isAvailable():
                values = o.getParameters()
                tags[o.getParticle().getObjectID().index] = values[1] >= 1 and values[5] == 1
        return tags

    def quality(self):
        """0.4 < p < 50 GeV, sigma(p)/p < 100%, 20 < theta < 160 deg, track length > 30 cm,
        |d_xy| < 4 cm and |z| < 10 cm at the primary vertex. Returns (mask, |p|)."""
        p = np.linalg.norm(self.p3, axis=1)
        theta = np.degrees(np.arccos(np.clip(self.p3[:, 2] / np.maximum(p, 1e-9), -1, 1)))
        omega = np.abs(self.helix[:, 2])
        sigma_omega = np.sqrt(np.maximum(self.helix_cov[:, 2, 2], 0))
        good = ((p > 0.4) & (p < 50)
                & (sigma_omega < omega)                  # sigma(p)/p = sigma(omega)/omega < 1
                & (theta > 20) & (theta < 160)
                & (self.length > 30)
                & (np.abs(self.d0_pv) < 40) & (np.abs(self.z0_pv) < 100))
        return good, p

    def has(self, bit):
        return (self.mask & bit) != 0


# ---------------------------------------------------------------- vertex fit
@dataclass
class Vertex:
    position: np.ndarray            # (3,) mm
    cov: np.ndarray                 # (3, 3)
    chi2: float
    ndf: int
    track_distance: np.ndarray      # transverse distance of each track to the vertex, mm


class VertexFitter:
    """F. Bedeschi's vertex fitter (Delphes TrackCovariance/VertexFit.cc, the one used by
    FCCAnalyses VertexFitterSimple), for one candidate. The helper names in brackets are
    the Delphes TrkUtil functions they reproduce."""

    def fit(self, tracks, idx):
        """Common vertex of the tracks idx of a TrackTable."""
        par0 = tracks.helix[idx] * TO_FITTER_CONVENTION
        cov = tracks.helix_cov[idx] * np.outer(TO_FITTER_CONVENTION, TO_FITTER_CONVENTION)
        vertex, phase = self.starting_point(par0, cov)
        vertex, vertex_cov, chi2 = self.iterate(par0, cov, vertex, phase)
        distance = self.transverse_distance(tracks.helix[idx], vertex)
        return Vertex(vertex, vertex_cov, chi2, 2 * len(idx) - 3, distance)

    def starting_point(self, par0, cov):
        """Each helix linearised at its point of closest approach to the z axis (phase 0);
        the vertex is the point nearest all these lines, weighted by the position errors
        transverse to each track. Returns the vertex and each track's phase there."""
        phase = np.zeros(len(par0))
        x0 = self.point(par0, phase)
        direction = self.d_point_d_phase(par0, phase)
        position_cov = self.sandwich(self.d_point_d_par(par0, phase), cov)
        weight = self.regularised_inverse(position_cov)
        w = np.einsum('nij,nj->ni', weight, direction)
        w_cov_w = np.einsum('ni,nij,nj->n', w, position_cov, w)
        transverse_weight = weight - w[:, :, None] * w[:, None, :] / w_cov_w[:, None, None]
        vertex = self.regularised_inverse(transverse_weight.sum(0)) @ np.einsum('nij,nj->i', transverse_weight, x0)
        phase = np.einsum('ni,ni->n', w, vertex - x0) / w_cov_w
        return vertex, phase

    def iterate(self, par0, cov, vertex, phase):
        """Iterate on the full helices: each pass moves every track's point to its phase
        nearest the vertex, refits the vertex, and pulls the track parameters towards it
        within their errors. Returns vertex, its covariance and the chi2."""
        par = par0.copy()
        previous = vertex.copy()
        for _ in range(VERTEX_FIT_MAX_ITERATIONS):
            A = self.d_point_d_par(par, phase)
            point_cov = self.sandwich(A, cov)
            point_weight = self.regularised_inverse(point_cov)
            point = self.point(par, phase)
            shift = np.einsum('nij,nj->ni', A, par - par0)          # point shift from the parameter pull
            tangent = self.d_point_d_phase(par, phase)

            # weight of each track with the along-track direction removed
            t_w_t = np.einsum('ni,nij,nj->n', tangent, point_weight, tangent)
            w_t = np.einsum('nij,nj->ni', point_weight, tangent)
            transverse_weight = point_weight - w_t[:, :, None] * w_t[:, None, :] / t_w_t[:, None, None]

            # new vertex and its covariance
            H_inverse = self.regularised_inverse(transverse_weight.sum(0))
            vertex = H_inverse @ np.einsum('nij,nj->i', transverse_weight, point - shift)
            vertex_cov = self.sandwich(H_inverse, self.sandwich(transverse_weight, point_cov).sum(0))

            # residuals, chi2, and the updated phases and track parameters
            lam = np.einsum('nij,nj->ni', transverse_weight, point - vertex - shift)
            chi2 = np.einsum('ni,nij,nj->', lam, point_cov, lam)
            residual = np.einsum('nij,nj->ni', point_weight, vertex - point + shift)
            phase = phase + np.einsum('ni,ni->n', tangent, residual) / t_w_t
            par = par0 - np.einsum('nij,nkj,nk->ni', cov, A, lam)

            step = vertex - previous
            previous = vertex
            if step @ self.regularised_inverse(vertex_cov) @ step <= VERTEX_FIT_TOLERANCE:
                break
        return vertex, vertex_cov, float(chi2)

    @staticmethod
    def transverse_distance(helix, vertex):
        """Distance in xy of each track, as a straight line through its point of closest
        approach to the z axis, to the vertex (EDM4hep helix parameters)."""
        d0, phi = helix[:, 0], helix[:, 1]
        closest = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi)], -1)
        transverse_axis = np.stack([-np.sin(phi), np.cos(phi)], -1)
        return np.abs(((vertex[:2] - closest) * transverse_axis).sum(1))

    @staticmethod
    def point(par, phase):
        """Point (x, y, z) on each helix at the given phase (Xtrack)."""
        D, phi0, C, z0, cot = np.moveaxis(par, -1, 0)
        return np.stack([-D * np.sin(phi0) + (np.sin(phase + phi0) - np.sin(phi0)) / (2 * C),
                         D * np.cos(phi0) - (np.cos(phase + phi0) - np.cos(phi0)) / (2 * C),
                         z0 + cot * phase / (2 * C)], -1)

    @staticmethod
    def d_point_d_par(par, phase):
        """Derivatives of the helix point with respect to the 5 parameters, shape
        (n, 3, 5) (derXdPar)."""
        D, phi0, C, z0, cot = np.moveaxis(par, -1, 0)
        zero, one = np.zeros_like(D), np.ones_like(D)
        s0, c0 = np.sin(phi0), np.cos(phi0)
        s1, c1 = np.sin(phase + phi0), np.cos(phase + phi0)
        d_D = np.stack([-s0, c0, zero], -1)
        d_phi0 = np.stack([-D * c0 + (c1 - c0) / (2 * C), -D * s0 + (s1 - s0) / (2 * C), zero], -1)
        d_C = np.stack([-(s1 - s0) / (2 * C**2), (c1 - c0) / (2 * C**2), -cot * phase / (2 * C**2)], -1)
        d_z0 = np.stack([zero, zero, one], -1)
        d_cot = np.stack([zero, zero, phase / (2 * C)], -1)
        return np.stack([d_D, d_phi0, d_C, d_z0, d_cot], -1)

    @staticmethod
    def d_point_d_phase(par, phase):
        """Derivative of the helix point with respect to the phase (derXds)."""
        D, phi0, C, z0, cot = np.moveaxis(par, -1, 0)
        return np.stack([np.cos(phase + phi0), np.sin(phase + phi0), cot], -1) / (2 * C)[..., None]

    @staticmethod
    def sandwich(A, M):
        """A M A^T."""
        return A @ M @ np.swapaxes(A, -1, -2)

    @staticmethod
    def regularised_inverse(M):
        """Inverse after scaling to unit diagonal, for badly scaled matrices (RegInv)."""
        d = np.sqrt(np.abs(np.diagonal(M, axis1=-2, axis2=-1)))
        d = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 1.0)
        scaled = M * d[..., :, None] * d[..., None, :]
        try:
            inverse = np.linalg.inv(scaled)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(scaled)
        return inverse * d[..., :, None] * d[..., None, :]


# ---------------------------------------------------------------- one event
class Event:
    """One event: hadronic selection and the candidates of every sample. Each candidate
    method returns rows (daughter momenta, daughter charges, extra columns...)."""

    vertex_fitter = VertexFitter()

    def __init__(self, frame):
        self.frame = frame
        self.particles = frame.get('sDST_MAIN_Particles')
        self.ebeam = frame.get_parameter('sDST_EVT_ECMS') / 2
        self.charged = [k for k in range(len(self.particles)) if self.particles[k].getCharge() != 0]
        self._tracks = None

    def is_hadronic(self):
        """>= 5 charged particles and charged energy (as pions) > 12% of sqrt(s)."""
        energy = sum(math.sqrt(self.momentum(k) @ self.momentum(k) + M_PI**2) for k in self.charged)
        return len(self.charged) >= MIN_CHARGED and energy > MIN_CHARGED_ENERGY_FRACTION * 2 * self.ebeam

    def momentum(self, k):
        p = self.particles[k].getMomentum()
        return np.array([p.x, p.y, p.z])

    @property
    def tracks(self):
        """TrackTable of the charged particles, built on first use."""
        if self._tracks is None:
            self._tracks = TrackTable(self.frame, self.particles, self.charged)
        return self._tracks

    def primary_vertex(self):
        """Position (mm) and EDM4hep's 6 stored covariance values, or None."""
        vertices = self.frame.get('sDST_PV_PrimaryVertex')
        if not len(vertices):
            return None
        v, c = vertices[0].getPosition(), vertices[0].getCovMatrix()
        position = np.array([v.x, v.y, v.z])
        return (position, [c.values[j] for j in range(6)]) if np.isfinite(position).all() else None

    def candidates(self):
        """sample -> rows of this event."""
        out = {'k0s': self.k0s(), 'jpsi': self.jpsi(), 'dstar': [], 'd0': [], 'dplus': []}
        if self.tracks.n < 2:
            return out
        out['dstar'] = self.dstar()
        pv = self.primary_vertex()
        if pv is not None:
            out['d0'], out['dplus'] = self.open_charm(*pv)
        return out

    # ---------------- K0S
    def k0s(self):
        """DELPHI's V0s: (daughter momenta at the vertex, charges, vertex radius, tag)."""
        rows = []
        for v0 in self.frame.get('sDST_V0_V0Candidates'):
            tag = int(v0.getParameters()[0])
            daughters = list(v0.getParticles())
            if tag < 1 or len(daughters) != 2:
                continue
            if daughters[0].getCharge() * daughters[1].getCharge() >= 0:
                continue
            if daughters[0].getCharge() < 0:             # positive daughter first
                daughters = daughters[::-1]

            vertex = v0.getPosition()
            momenta = [self.momentum_at_vertex(d, vertex.x, vertex.y) for d in daughters]
            if any(p is None for p in momenta):
                continue
            rows.append((momenta, [1, -1], math.hypot(vertex.x, vertex.y), tag))
        return rows

    @staticmethod
    def momentum_at_vertex(particle, vx, vy):
        """Momentum of a V0 daughter at the V0 vertex (vx, vy in mm).

        The stored momentum is given at the interaction point. At the V0 vertex the track
        points along the tangent of its helix circle at the point nearest the vertex in
        xy; |p_T| and p_z are unchanged along a helix."""
        s = state_at_ip(particle)
        if s is None:
            return None
        # centre of the helix circle in xy (EDM4hep: omega = signed curvature, 1/mm)
        radius = 1 / s.omega
        centre_x = (radius - s.D0) * math.sin(s.phi)
        centre_y = -(radius - s.D0) * math.cos(s.phi)
        # tangent of the circle at the vertex, oriented along the motion of the particle
        tangent_x = s.omega * (vy - centre_y)
        tangent_y = -s.omega * (vx - centre_x)
        norm = math.hypot(tangent_x, tangent_y)

        p = particle.getMomentum()
        pt = math.hypot(p.x, p.y)
        return [pt * tangent_x / norm, pt * tangent_y / norm, p.z]

    # ---------------- J/psi
    def jpsi(self):
        """Opposite-charge pairs of tagged muons above 1.5 GeV: (momenta, charges)."""
        muon_tag = {}
        for o in self.frame.get('sDST_MUID_MuonID'):
            if o.getParticle().isAvailable() and len(o.getParameters()):
                muon_tag[o.getParticle().getObjectID().index] = int(round(o.getParameters()[0]))

        muons = []
        for k in self.charged:
            p = self.momentum(k)
            if muon_tag.get(k, 0) & MUON_TAG_BITS and np.linalg.norm(p) > 1.5:
                muons.append((p.tolist(), self.particles[k].getCharge()))

        rows = []
        for a in range(len(muons)):
            for b in range(a + 1, len(muons)):
                (p_a, q_a), (p_b, q_b) = muons[a], muons[b]
                if q_a * q_b >= 0:
                    continue
                m = invariant_mass(four_vector(p_a, M_MU) + four_vector(p_b, M_MU))
                if JPSI_WINDOW[0] < m < JPSI_WINDOW[1]:
                    rows.append(([p_a, p_b], [q_a, q_b]))
        return rows

    # ---------------- D*
    def dstar(self):
        """D*+ -> D0 pi+, D0 -> K- pi+: (momenta, charges, Delta M).

        K and pi: above 1 GeV with a VD hit, no kaon ID. Soft pion: 0.4-3.5 GeV, the
        charge of the pi, |d0| < 3 mm. x_E(D*) = E(D*)/E_beam > 0.15, M(K pi) and
        Delta M in their windows, and DELPHI's helicity cut."""
        t = self.tracks
        good = np.nonzero(t.good)[0]
        hard = good[(t.p[good] > 1) & t.has(VD_BIT)[good]]
        soft = good[(t.p[good] > 0.4) & (t.p[good] < 3.5) & (np.abs(t.d0_pv[good]) < 3)]
        if len(hard) < 2 or not len(soft):
            return []

        # all (kaon, pion, soft pion) combinations with the right charges
        kaon, pion, soft_pion = (a.ravel() for a in np.meshgrid(hard, hard, soft, indexing='ij'))
        q = t.q
        ok = ((kaon != pion) & (soft_pion != kaon) & (soft_pion != pion)
              & (q[kaon] == -q[pion]) & (q[soft_pion] == q[pion]))
        kaon, pion, soft_pion = kaon[ok], pion[ok], soft_pion[ok]

        kaon_p4 = four_vector(t.p3[kaon], M_K)
        d0_p4 = kaon_p4 + four_vector(t.p3[pion], M_PI)
        dstar_p4 = d0_p4 + four_vector(t.p3[soft_pion], M_PI)
        m_kpi = invariant_mass(d0_p4)
        dm = invariant_mass(dstar_p4) - m_kpi
        x_e = dstar_p4[:, 0] / self.ebeam

        keep = np.nonzero((x_e > 0.15) & (m_kpi > DSTAR_KPI_WINDOW[0]) & (m_kpi < DSTAR_KPI_WINDOW[1])
                          & (dm < DSTAR_MAX_DM))[0]
        if not len(keep):
            return []
        cos_helicity = kaon_helicity_cosine(kaon_p4[keep], d0_p4[keep])
        limit = np.select([x_e[keep] < edge for edge, _ in DSTAR_HELICITY_CUT],
                          [lim for _, lim in DSTAR_HELICITY_CUT])
        keep = keep[cos_helicity > limit]

        rows = []
        for i in keep:
            three = [kaon[i], pion[i], soft_pion[i]]
            rows.append((t.p3[three], q[three], dm[i]))
        return rows

    # ---------------- D0 and D+
    def open_charm(self, pv, pv_cov):
        """D0 -> K- pi+ and D+ -> K- pi+ pi+ with DELPHI's selections: (momenta, charges).

        D0: both tracks above 1 GeV with a VD hit, at least one reaching the OD; kaon ID;
            vertex P(chi2) > 1%; decay length significance > 1, -0.5 < L < 2 cm; x_E > 0.15.
        D+: p_K > 2, p_pi > 1 GeV; >= 2 tracks with VD hits; kaon ID for x_E < 0.55; each
            track within 100 um of the vertex in xy; vertex P(chi2) > 0.1%; decay length
            significance > 3, -0.5 < L < 2 cm; x_E > 0.15; D* veto: M(K pi pi) - M(K pi)
            > 200 MeV for both pions."""
        from scipy.stats import chi2 as chi2_distribution
        t = self.tracks
        tracks = np.nonzero(t.good & (t.p > 1))[0]
        d0_rows, dplus_rows = [], []
        if len(tracks) < 2:
            return d0_rows, dplus_rows

        q, has_vd, has_od = t.q, t.has(VD_BIT), t.has(OD_BIT)
        as_kaon = {i: four_vector(t.p3[i], M_K) for i in tracks}
        as_pion = {i: four_vector(t.p3[i], M_PI) for i in tracks}

        for k in tracks:
            for pi in tracks:
                if pi == k or q[pi] != -q[k]:
                    continue

                # ---- D0 -> K pi
                if t.kaon[k] and has_vd[k] and has_vd[pi] and (has_od[k] or has_od[pi]):
                    p4 = as_kaon[k] + as_pion[pi]
                    m, x_e = invariant_mass(p4), p4[0] / self.ebeam
                    if x_e > 0.15 and D0_WINDOW[0] < m < D0_WINDOW[1]:
                        vertex = self.vertex_fitter.fit(t, [k, pi])
                        L, sigma_L = self.decay_length(vertex, pv, pv_cov, p4[1:])
                        if (chi2_distribution.sf(vertex.chi2, vertex.ndf) > 0.01
                                and L / sigma_L > 1 and -5 < L < 20):
                            d0_rows.append((t.p3[[k, pi]], q[[k, pi]]))

                # ---- D+ -> K pi pi: the second pion has the charge of the first; each
                # pion pair is taken once (pi2 after pi)
                if t.p[k] <= 2:
                    continue
                for pi2 in tracks[tracks > pi]:
                    if pi2 == k or q[pi2] != q[pi]:
                        continue
                    if int(has_vd[k]) + int(has_vd[pi]) + int(has_vd[pi2]) < 2:   # at least 2 with VD hits
                        continue
                    p4 = as_kaon[k] + as_pion[pi] + as_pion[pi2]
                    m, x_e = invariant_mass(p4), p4[0] / self.ebeam
                    if x_e <= 0.15 or not DPLUS_WINDOW[0] < m < DPLUS_WINDOW[1]:
                        continue
                    if x_e < 0.55 and not t.kaon[k]:
                        continue
                    dm_1 = m - invariant_mass(as_kaon[k] + as_pion[pi])
                    dm_2 = m - invariant_mass(as_kaon[k] + as_pion[pi2])
                    if min(dm_1, dm_2) <= 0.200:                    # D* veto
                        continue
                    vertex = self.vertex_fitter.fit(t, [k, pi, pi2])
                    if chi2_distribution.sf(vertex.chi2, vertex.ndf) <= 0.001:
                        continue
                    if (vertex.track_distance > 0.1).any():
                        continue
                    L, sigma_L = self.decay_length(vertex, pv, pv_cov, p4[1:])
                    if L / sigma_L > 3 and -5 < L < 20:
                        dplus_rows.append((t.p3[[k, pi, pi2]], q[[k, pi, pi2]]))
        return d0_rows, dplus_rows

    @staticmethod
    def decay_length(vertex, pv, pv_cov, direction):
        """Signed decay length (mm) and its error.

        The transverse distance from the primary vertex along the D direction, scaled to
        3D by 1 / sin(theta_D). The error uses the transverse vertex and primary-vertex
        errors, so the significance is the transverse one. pv_cov: EDM4hep's 6 stored
        values of the primary-vertex covariance, of which (xx, xy, yy) are the first three."""
        u = direction[:2] / np.linalg.norm(direction[:2])
        pv_cov_xy = np.array([[pv_cov[0], pv_cov[1]], [pv_cov[1], pv_cov[2]]])
        to_3d = np.linalg.norm(direction) / np.linalg.norm(direction[:2])
        length = (vertex.position[:2] - pv[:2]) @ u
        variance = u @ (vertex.cov[:2, :2] + pv_cov_xy) @ u
        return length * to_3d, math.sqrt(max(variance, 1e-12)) * to_3d


# ---------------------------------------------------------------- all files
class Samples:
    """The candidates of all samples, collected over events and files."""

    def __init__(self):
        self.rows = {s: [] for s in SAMPLES}
        self.count = collections.Counter()

    def add(self, other):
        for s in SAMPLES:
            self.rows[s] += other.rows[s]
        self.count.update(other.count)

    def summary(self):
        return (f'{self.count["hadronic"]} hadronic events, '
                + ', '.join(f'{len(self.rows[s])} {s}' for s in SAMPLES))

    @classmethod
    def from_file(cls, path):
        import podio.root_io as rio
        samples = cls()
        events = rio.Reader(path).get('events')
        for i in range(len(events)):
            event = Event(events[i])
            samples.count['events'] += 1
            if not event.is_hadronic():
                continue
            samples.count['hadronic'] += 1
            for s, rows in event.candidates().items():
                samples.rows[s] += rows
        return samples

    def arrays(self):
        """<sample>_p3 (n, daughters, 3), <sample>_q (n, daughters) and the extra columns."""
        out = {}
        for sample, spec in SAMPLES.items():
            rows, n = self.rows[sample], len(self.rows[sample])
            out[f'{sample}_p3'] = np.array([r[0] for r in rows], np.float32).reshape(n, spec['daughters'], 3)
            out[f'{sample}_q'] = np.array([r[1] for r in rows], np.int8).reshape(n, spec['daughters'])
            for j, (name, dtype) in enumerate(spec['extra'].items()):
                out[f'{sample}_{name}'] = np.array([r[2 + j] for r in rows], dtype)
        return out

    def save(self, path, dataset, files):
        np.savez_compressed(path, dataset=dataset, files=np.array(files), n_events=self.count['events'],
                            n_hadronic=self.count['hadronic'], **self.arrays())


def samples_or_error(path):
    """Samples.from_file for the worker pool: an error is returned, not raised, so that
    the main process can list every failed file."""
    try:
        return path, Samples.from_file(path), None
    except Exception as e:
        return path, None, f'{type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser(description='Skim the momentum-scale samples from DELPHI EDM4hep files.')
    ap.add_argument('--dataset', required=True, help='DELPHI processing of the input files, e.g. 94C2')
    ap.add_argument('--output', required=True, help='output .npz')
    ap.add_argument('--jobs', type=int, default=1, help='number of files processed in parallel')
    ap.add_argument('--filelist', help='text file with one input file per line')
    ap.add_argument('files', nargs='*', help='input EDM4hep files')
    args = ap.parse_args()

    files = list(args.files)
    if args.filelist:
        files += [line.strip() for line in open(args.filelist) if line.strip() and not line.startswith('#')]
    if not files:
        ap.error('no input files')
    files = sorted(files)

    total, failed = Samples(), []
    with multiprocessing.Pool(args.jobs) as pool:
        # imap returns the files in input order, so the output does not depend on --jobs
        for n, (path, samples, error) in enumerate(pool.imap(samples_or_error, files), 1):
            if error:
                failed.append((path, error))
                continue
            total.add(samples)
            print(f'[{n}/{len(files)}] {path}: {samples.summary()}', flush=True)

    # a failed file would silently change the sample: list the failures and stop
    if failed:
        for path, error in failed:
            print(f'FAILED {path}: {error}', file=sys.stderr)
        sys.exit(f'{len(failed)} of {len(files)} files failed; no output written')

    total.save(args.output, args.dataset, files)
    print(f'wrote {args.output}: {total.summary()}')


if __name__ == '__main__':
    main()
