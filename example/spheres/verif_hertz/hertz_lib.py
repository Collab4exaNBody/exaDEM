"""Shared helpers for the Hertz verification tests (stdlib only).

- exact (numerical) solution of exaDEM's Hertz normal contact law on a single impact;
- the parameters of the 3 tests (must match the .msp/.xyz files);
- the readers of the exaDEM outputs are the ones of the Hooke verification
  (../verif_hooke/hooke_lib.py): Paraview .pvtp and interaction dump.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verif_hooke"))
from hooke_lib import read_contact_history, read_velocity_history, sphere_mass  # noqa: E402,F401


def reduced_young_modulus(young1, poisson1, young2, poisson2):
    """1 / E* = (1 - nu1^2) / E1 + (1 - nu2^2) / E2"""
    return 1.0 / ((1.0 - poisson1 ** 2) / young1 + (1.0 - poisson2 ** 2) / young2)


# Materials of Table 2 of Chung & Ooi (2011): (Young modulus [Pa], Poisson ratio, density [kg/m3])
GLASS = (48.0e9, 0.20, 2800.0)
LIMESTONE = (20.0e9, 0.25, 2500.0)
AL_ALLOY = (70.0e9, 0.30, 2699.0)
AL_OXIDE = (380.0e9, 0.23, 4000.0)
RIGID = (1.0e40, 0.0, None)  # rigid plane


def _contact(name, i, j, mat1, mat2, radius1, radius2, zeta, v0, paper=None):
    """Normal impact data of the spheres i and j (j=None: plane, made of mat2).
    paper: values of Table 3 of Chung & Ooi (2011): max overlap [m], max force [N], contact time [s]."""
    m1 = sphere_mass(radius1, mat1[2])
    if j is None:
        meff, reff = m1, radius1
    else:
        m2 = sphere_mass(radius2, mat2[2])
        meff = m1 * m2 / (m1 + m2)
        reff = radius1 * radius2 / (radius1 + radius2)
    return {"name": name, "i": i, "j": j, "estar": reduced_young_modulus(mat1[0], mat1[1], mat2[0], mat2[1]),
            "reff": reff, "meff": meff, "zeta": zeta, "v0": v0, "paper": paper}


# damp_rate giving e = 0.9, 0.7, 0.5, 0.3 with the damping of the Hertz law (see damp_rate_for_restitution)
TARGET_E = [0.9, 0.7, 0.5, 0.3]
DAMP_RATES = [0.0312797, 0.1115664, 0.2330340, 0.4539161]

TESTS = {
    "test1": {
        "dir": "VerifHertzTest1Dir",
        "dt": 1e-7,
        "title": "Test 1 (Chung & Ooi 2011) - elastic impact of two identical spheres (Hertz)",
        "contacts": [
            _contact("Glass", 0, 1, GLASS, GLASS, 0.01, 0.01, 0.0, 20.0, paper=(274.1e-6, 10697.0, 40.3e-6)),
            _contact("Limestone", 2, 3, LIMESTONE, LIMESTONE, 0.01, 0.01, 0.0, 20.0, paper=(368.3e-6, 7108.0, 54.2e-6)),
        ],
        # relative velocity along x of the pair (j - i), approaching at v0 before the impact
        "pairs": [{"name": "Glass", "i": 0, "j": 1, "comp": 0, "contact": 0, "e": 1.0},
                  {"name": "Limestone", "i": 2, "j": 3, "comp": 0, "contact": 1, "e": 1.0}],
        "velocities": [],
        # only plotted: identical spheres, each one moves at its v0 times x'(t) / x'(0)
        "plot_velocities": [{"name": "Glass A", "id": 0, "comp": 0, "v0": 10.0, "contact": 0},
                            {"name": "Glass B", "id": 1, "comp": 0, "v0": -10.0, "contact": 0},
                            {"name": "Limestone A", "id": 2, "comp": 0, "v0": 10.0, "contact": 1},
                            {"name": "Limestone B", "id": 3, "comp": 0, "v0": -10.0, "contact": 1}],
    },
    "test2": {
        "dir": "VerifHertzTest2Dir",
        "dt": 1e-6,
        "title": "Test 2 (Chung & Ooi 2011) - elastic impact of a sphere on a rigid plane (Hertz)",
        "contacts": [_contact("Al. alloy", 0, None, AL_ALLOY, RIGID, 0.1, None, 0.0, 0.2,
                              paper=(49.7e-6, 11370.0, 731.6e-6))],
        # particle id, velocity component, initial value, contact, expected restitution coefficient
        "velocities": [{"name": "Al. alloy", "id": 0, "comp": 2, "v0": -0.2, "contact": 0, "e": 1.0}],
    },
    "test3": {
        "dir": "VerifHertzTest3Dir",
        "dt": 5e-9,
        "title": "Test 3 (Chung & Ooi 2011) - impacts on a rigid plane with different restitution coefficients (Hertz)",
        "contacts": [_contact("e=%g" % e, k, None, AL_OXIDE, RIGID, 0.0025, None, z, 3.9)
                     for k, (e, z) in enumerate(zip(TARGET_E, DAMP_RATES))],
        "velocities": [{"name": "e=%g" % e, "id": k, "comp": 2, "v0": -3.9, "contact": k, "e": e}
                       for k, e in enumerate(TARGET_E)],
    },
}


class HertzContact:
    """Normal impact solution of exaDEM's Hertz law, integrated with a 4th order Runge-Kutta scheme.

    meff x'' = -Fn,   Fn = max(0, Fe + c x'),
    Fe = 4/3 E* sqrt(R*) x^(3/2),   c = 2 zeta sqrt(meff k),   k = 2 E* sqrt(R* x)
    where x is the overlap, x(0) = 0 and x'(0) = v0 > 0. The contact ends when the overlap
    comes back to 0 or when the force reaches 0 (damped case), e = -x'(t_end) / v0.
    """

    def __init__(self, estar, reff, meff, zeta, v0, steps=20000):
        self.estar, self.reff, self.meff, self.zeta, self.v0 = estar, reff, meff, zeta, v0
        t_hertz = 2.87 * (meff ** 2 / (reff * estar ** 2 * v0)) ** 0.2  # contact time of the elastic impact
        h = t_hertz / steps
        self.t, self.x, self.f, self.u = [0.0], [0.0], [0.0], [v0]
        t, x, u = 0.0, 0.0, v0
        for _ in range(20 * steps):
            k1x, k1u = self._rhs(x, u)
            k2x, k2u = self._rhs(x + 0.5 * h * k1x, u + 0.5 * h * k1u)
            k3x, k3u = self._rhs(x + 0.5 * h * k2x, u + 0.5 * h * k2u)
            k4x, k4u = self._rhs(x + h * k3x, u + h * k3u)
            x_new = x + h / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
            u_new = u + h / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
            f_new = self._force(x_new, u_new)
            if x_new <= 0.0 or (f_new <= 0.0 and u_new < 0.0):
                # end of the contact, linear interpolation of the last step
                self.t_end = t + h
                self.u_end = u_new
                self.e = -u_new / v0
                break
            t, x, u = t + h, x_new, u_new
            self.t.append(t)
            self.x.append(x)
            self.f.append(f_new)
            self.u.append(u_new)
        else:
            raise RuntimeError("the contact does not end")

    def _force(self, x, u):
        if x <= 0.0:
            return 0.0
        fe = 4.0 / 3.0 * self.estar * math.sqrt(self.reff) * x ** 1.5
        c = 2.0 * self.zeta * math.sqrt(self.meff * 2.0 * self.estar * math.sqrt(self.reff * x))
        return max(0.0, fe + c * u)

    def _rhs(self, x, u):
        return u, -self._force(x, u) / self.meff

    def _interp(self, series, t):
        if t < 0.0 or t > self.t_end:
            return 0.0
        h = self.t[1] - self.t[0]
        k = min(int(t / h), len(self.t) - 2)
        a = (t - self.t[k]) / h
        return series[k] * (1.0 - a) + series[k + 1] * a

    def overlap(self, t):
        return self._interp(self.x, t)

    def force(self, t):
        return self._interp(self.f, t)

    def velocity(self, t):
        """Approach velocity x'(t) (v0 at t = 0, -e v0 after the contact)."""
        if t > self.t_end:
            return self.u_end
        return self._interp(self.u, t)

    def peak_force(self):
        return max(self.f)

    def max_overlap(self):
        return max(self.x)


def analytical(contact):
    return HertzContact(contact["estar"], contact["reff"], contact["meff"], contact["zeta"], contact["v0"])


def damp_rate_for_restitution(e, iterations=30, steps=8000):
    """damp_rate giving the restitution coefficient e with the damping of exaDEM's Hertz law.
    With c = 2 damp_rate sqrt(m k), k = 2 E* sqrt(R delta), e only depends on damp_rate (not on the
    velocity, the mass, the radius or E*), so any impact can be used to compute it (bisection)."""
    estar, reff, meff, v0 = 1.0e9, 1.0e-3, 1.0e-3, 1.0
    lo, hi = 0.0, 2.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if HertzContact(estar, reff, meff, mid, v0, steps).e > e:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
