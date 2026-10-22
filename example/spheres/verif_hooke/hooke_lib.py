"""Shared helpers for the Hooke verification tests (stdlib only).

- readers for exaDEM outputs: Paraview .pvtp/.vtp (base64 + zlib) and the
  interaction dump (ExaDEMAnalyses/InteractionOutputDir-<step>/*.txt);
- exact analytical solution of exaDEM's linear spring-dashpot normal contact;
- the parameters of the 3 tests (must match the .msp/.xyz files).
"""
import base64
import glob
import math
import os
import re
import struct
import xml.etree.ElementTree as ET
import zlib


def sphere_mass(radius, density):
    return 4.0 / 3.0 * math.pi * radius ** 3 * density


# One entry per contact: particle ids (j=None for the rigid plane), kn, damp_rate,
# effective mass, initial normal approach speed v0, and which particles' velocity
# component to check (id, component index, initial value).
_M_GLASS = sphere_mass(0.01, 2800)
_M_LIME = sphere_mass(0.01, 2500)
_M_AL = sphere_mass(0.1, 2699)
_M_OX = sphere_mass(0.0025, 4000)

TESTS = {
    "test1": {
        "msp": "test1_two_spheres.msp",
        "dir": "VerifHookeTest1Dir",
        "dt": 1e-7,
        "title": "Test 1 - elastic impact of two identical spheres",
        "contacts": [
            {"name": "Glass", "i": 0, "j": 1, "kn": 3.902369e7, "zeta": 0.0, "meff": _M_GLASS / 2, "v0": 20.0},
            {"name": "Limestone", "i": 2, "j": 3, "kn": 1.929993e7, "zeta": 0.0, "meff": _M_LIME / 2, "v0": 20.0},
        ],
        "velocities": [
            {"name": "Glass A", "id": 0, "comp": 0, "v0": 10.0, "contact": 0},
            {"name": "Glass B", "id": 1, "comp": 0, "v0": -10.0, "contact": 0},
            {"name": "Limestone A", "id": 2, "comp": 0, "v0": 10.0, "contact": 1},
            {"name": "Limestone B", "id": 3, "comp": 0, "v0": -10.0, "contact": 1},
        ],
    },
    "test2": {
        "msp": "test2_sphere_plane.msp",
        "dir": "VerifHookeTest2Dir",
        "dt": 1e-6,
        "title": "Test 2 - elastic impact of a sphere on a rigid plane",
        "contacts": [
            {"name": "Al. alloy", "i": 0, "j": None, "kn": 2.286908e8, "zeta": 0.0, "meff": _M_AL, "v0": 0.2},
        ],
        "velocities": [
            {"name": "Al. alloy", "id": 0, "comp": 2, "v0": -0.2, "contact": 0},
        ],
    },
    "test3": {
        "msp": "test3_restitution_sweep.msp",
        "dir": "VerifHookeTest3Dir",
        "dt": 5e-7,
        "title": "Test 3 - damped impact on a rigid plane (restitution sweep)",
        "contacts": [
            {"name": "e=0.9", "i": 0, "j": None, "kn": 1e5, "zeta": 0.033518, "meff": _M_OX, "v0": 3.9},
            {"name": "e=0.7", "i": 1, "j": None, "kn": 1e5, "zeta": 0.112808, "meff": _M_OX, "v0": 3.9},
            {"name": "e=0.5", "i": 2, "j": None, "kn": 1e5, "zeta": 0.215454, "meff": _M_OX, "v0": 3.9},
            {"name": "e=0.3", "i": 3, "j": None, "kn": 1e5, "zeta": 0.357857, "meff": _M_OX, "v0": 3.9},
        ],
        "velocities": [
            {"name": "e=0.9", "id": 0, "comp": 2, "v0": -3.9, "contact": 0},
            {"name": "e=0.7", "id": 1, "comp": 2, "v0": -3.9, "contact": 1},
            {"name": "e=0.5", "id": 2, "comp": 2, "v0": -3.9, "contact": 2},
            {"name": "e=0.3", "id": 3, "comp": 2, "v0": -3.9, "contact": 3},
        ],
    },
}


class HookeContact:
    """Exact normal impact solution of exaDEM's linear spring-dashpot law.

    meff x'' = -Fn,  Fn = max(0, kn x + c x'),  c = 2 zeta sqrt(kn meff),
    x(0) = 0, x'(0) = v0 > 0 (x = overlap). While Fn > 0 the motion is the
    underdamped oscillator x = v0/wd e^{-a t} sin(wd t); the force reaches 0
    at t_end (tan(wd t_end) = -c wd / (kn - c a)) and the particles then
    separate at constant speed, so e = -x'(t_end) / v0. zeta = 0 gives the
    undamped case: t_end = pi/w, e = 1.
    """

    def __init__(self, kn, zeta, meff, v0):
        self.kn, self.meff, self.v0 = kn, meff, v0
        self.wn = math.sqrt(kn / meff)
        self.a = zeta * self.wn
        self.wd = self.wn * math.sqrt(1.0 - zeta * zeta)
        self.c = 2.0 * zeta * math.sqrt(kn * meff)
        theta = math.atan2(-self.c * self.wd, kn - self.c * self.a) % math.pi
        self.t_end = (theta if theta > 0 else math.pi) / self.wd
        self.e = -self.velocity(self.t_end) / v0

    def overlap(self, t):
        if t <= self.t_end:
            return self.v0 / self.wd * math.exp(-self.a * t) * math.sin(self.wd * t)
        return self.overlap(self.t_end) + self.velocity(self.t_end) * (t - self.t_end)

    def velocity(self, t):
        if t > self.t_end:
            t = self.t_end
        return self.v0 * math.exp(-self.a * t) * (math.cos(self.wd * t) - self.a / self.wd * math.sin(self.wd * t))

    def force(self, t):
        if t < 0 or t > self.t_end:
            return 0.0
        return max(0.0, self.kn * self.overlap(t) + self.c * self.velocity(t))

    def peak_force(self, n=4000):
        return max(self.force(self.t_end * k / n) for k in range(n + 1))

    def max_overlap(self, n=4000):
        return max(self.overlap(self.t_end * k / n) for k in range(n + 1))


def analytical(contact):
    return HookeContact(contact["kn"], contact["zeta"], contact["meff"], contact["v0"])


# ---------------------------------------------------------------- readers ----

def _decode_binary_array(text, vtk_type):
    """Decode a VTK XML 'binary' DataArray (UInt64 header, zlib compressed)."""
    text = "".join(text.split())
    nblocks = struct.unpack("<Q", base64.b64decode(text[:12])[:8])[0]
    header_len = 4 * math.ceil((3 + nblocks) * 8 / 3)
    header = struct.unpack("<%dQ" % (3 + nblocks), base64.b64decode(text[:header_len]))
    data = base64.b64decode(text[header_len:])
    raw, pos = b"", 0
    for size in header[3:]:
        raw += zlib.decompress(data[pos:pos + size])
        pos += size
    fmt = {"Float64": "d", "Float32": "f", "UInt64": "Q", "Int64": "q",
           "UInt32": "I", "Int32": "i"}[vtk_type]
    return struct.unpack("<%d%s" % (len(raw) // struct.calcsize(fmt), fmt), raw)


def read_pvtp_velocities(pvtp):
    """Return {particle id: (vx, vy, vz)} from one Paraview snapshot."""
    out = {}
    base = os.path.dirname(pvtp)
    for piece in ET.parse(pvtp).getroot().iter("Piece"):
        vtp = os.path.join(base, piece.get("Source"))
        arrays = {}
        for da in ET.parse(vtp).getroot().iter("DataArray"):
            if da.get("Name") in ("id", "velocity"):
                arrays[da.get("Name")] = _decode_binary_array(da.text or "", da.get("type"))
        ids, vel = arrays.get("id", ()), arrays.get("velocity", ())
        for k, pid in enumerate(ids):
            out[int(pid)] = tuple(vel[3 * k:3 * k + 3])
    return out


def read_velocity_history(test_dir, dt):
    """Return {particle id: {"t": [...], "v": [(vx,vy,vz), ...]}}, sorted by time."""
    snaps = []
    for f in glob.glob(os.path.join(test_dir, "ParaviewOutputFiles", "exadem_*.pvtp")):
        m = re.search(r"exadem_(\d+)\.pvtp$", f)
        if m:
            snaps.append((int(m.group(1)), f))
    hist = {}
    for step, f in sorted(snaps):
        for pid, v in read_pvtp_velocities(f).items():
            h = hist.setdefault(pid, {"t": [], "v": []})
            h["t"].append(step * dt)
            h["v"].append(v)
    return hist


def read_contact_history(test_dir, dt, i, j):
    """Overlap and normal force history of the contact between particles i and j
    (j=None: contact of i with a driver). Returns {"t", "overlap", "fn"}, sorted
    by time; overlap = -dn (> 0 in contact), fn = |normal force|."""
    samples = {}
    for d in glob.glob(os.path.join(test_dir, "ExaDEMAnalyses", "InteractionOutputDir-*")):
        m = re.search(r"InteractionOutputDir-(\d+)$", d)
        if not m:
            continue
        step = int(m.group(1))
        for txt in glob.glob(os.path.join(d, "*.txt")):
            with open(txt) as fh:
                for line in fh:
                    v = line.strip().split(",")
                    if len(v) < 12:
                        continue
                    a, b = int(v[0]), int(v[1])
                    if j is None:
                        ok = (a == i and int(v[4]) >= 4)
                    else:
                        ok = {a, b} == {i, j}
                    if ok:
                        fn = math.sqrt(sum(float(x) ** 2 for x in v[9:12]))
                        samples[step] = (-float(v[5]), fn)
    steps = sorted(samples)
    return {"t": [s * dt for s in steps],
            "overlap": [samples[s][0] for s in steps],
            "fn": [samples[s][1] for s in steps]}
