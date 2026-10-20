#!/usr/bin/env python3
"""
Généré par Claude :
Genere un  bloc de cellules de Voronoi (DEM-BCM) pour ExaDEM :
  - voronoi_block.shp : formes (sommets reduits du rayon de Minkowski)
  - voronoi_block.txt : configuration au format Rockable
Unites : mm. Usage : python3 generate_voronoi_block.py [--lx 20 --ly 40 --lz 10 --h 2.8 --seed 1 --dmin 0.65 --lloyd 0]
"""
import argparse
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree

p = argparse.ArgumentParser()
p.add_argument("--lx", type=float, default=20.0)
p.add_argument("--ly", type=float, default=40.0)    # direction de traction
p.add_argument("--lz", type=float, default=10.0)
p.add_argument("--h", type=float, default=2.8)      # taille moyenne de cellule
p.add_argument("--radius", type=float, default=0.05)  # rayon de Minkowski (comme nooru.shp)
p.add_argument("--seed", type=int, default=1)       # graine aleatoire (meme graine = meme bloc)
p.add_argument("--dmin", type=float, default=0.65)  # distance min entre germes / h (max ~0.85)
p.add_argument("--lloyd", type=int, default=0)      # iterations de Lloyd (cellules plus regulieres)
p.add_argument("--prefix", default="voronoi_block")
a = p.parse_args()

L = np.array([a.lx, a.ly, a.lz])
r = a.radius
rng = np.random.default_rng(a.seed)

# --- germes : tirage aleatoire avec distance minimale (cellules regulieres) ---
n_target = int(round(np.prod(L) / a.h**3))
dmin, margin = a.dmin * a.h, 0.3
seeds = []
tries = 0
while len(seeds) < n_target and tries < 2000000:
    x = margin + rng.random(3) * (L - 2 * margin)
    if all(np.linalg.norm(x - s) > dmin for s in seeds):
        seeds.append(x)
    tries += 1
seeds = np.array(seeds)
if len(seeds) < n_target:
    print(f"attention : {len(seeds)} germes seulement (dmin trop grand ?)")
tree = cKDTree(seeds)


def cell_vertices(i, r=r):
    """Sommets de la cellule i, chaque plan decale de r vers l'interieur."""
    s = seeds[i]
    hs = []  # format scipy : A x + b <= 0
    for j in tree.query_ball_point(s, 3.0 * a.h):
        if j == i:
            continue
        nrm = seeds[j] - s
        d = np.linalg.norm(nrm)
        nrm /= d
        hs.append([*nrm, -(np.dot(nrm, s) + 0.5 * d - r)])
    for k in range(3):  # faces du bloc
        e = np.zeros(3); e[k] = 1.0
        hs.append([*e, -(L[k] - r)])
        hs.append([*(-e), r])
    hs = np.array(hs)
    raw = HalfspaceIntersection(hs, s).intersections
    v = []  # fusion des sommets doubles (points ou plus de 3 plans se coupent)
    for x in raw:
        if all(np.linalg.norm(x - y) > 1e-9 for y in v):
            v.append(x)
    # raffinement : chaque sommet est recalcule exactement sur ses plans actifs,
    # pour que les faces soient coplanaires a la precision machine (ExaDEM : tol 1e-10)
    out = []
    for x in v:
        act = np.abs(hs[:, :3] @ x + hs[:, 3]) < 1e-6
        y, _, rank, _ = np.linalg.lstsq(hs[act, :3], -hs[act, 3], rcond=None)
        if rank < 3 or np.linalg.norm(y - x) > 1e-6:
            raise RuntimeError(f"cellule {i} : sommet mal conditionne")
        out.append(y)
    return np.array(out)


def polyhedron(v):
    """Faces (polygones ordonnes), aretes, volume, centre, tenseur d'inertie / masse."""
    hull = ConvexHull(v)
    v = hull.points
    faces = []
    planes = []  # plans distincts (les triangles coplanaires sont regroupes)
    for eq in hull.equations:
        if not any(np.dot(eq[:3], q[:3]) > 1 - 1e-8 and abs(eq[3] - q[3]) < 1e-7 for q in planes):
            planes.append(eq)
    for eq in planes:
        idx = np.where(np.abs(v @ eq[:3] + eq[3]) < 1e-6)[0]
        c = v[idx].mean(axis=0)
        n = eq[:3]
        u = v[idx[0]] - c; u /= np.linalg.norm(u)
        w = np.cross(n, u)
        ang = np.arctan2((v[idx] - c) @ w, (v[idx] - c) @ u)
        faces.append(list(idx[np.argsort(ang)]))  # sens direct autour de la normale sortante
    used = sorted({k for f in faces for k in f})
    remap = {old: new for new, old in enumerate(used)}
    v = v[used]
    faces = [[remap[k] for k in f] for f in faces]
    edges = sorted({tuple(sorted((f[k], f[(k + 1) % len(f)]))) for f in faces for k in range(len(f))})
    # volume, centre et inertie par tetraedres (sommet de reference = moyenne des sommets)
    o = v.mean(axis=0)
    vol, cm, J = 0.0, np.zeros(3), np.zeros((3, 3))
    tets = []
    for f in faces:
        for k in range(1, len(f) - 1):
            p0, p1, p2 = v[f[0]], v[f[k]], v[f[k + 1]]
            dv = abs(np.dot(p0 - o, np.cross(p1 - o, p2 - o))) / 6.0
            vol += dv
            cm += dv * (o + p0 + p1 + p2) / 4.0
            tets.append((dv, (o, p0, p1, p2)))
    cm /= vol
    for dv, pts in tets:  # second moment exact d'un tetraedre
        q = [pt - cm for pt in pts]
        s = sum(q)
        J += dv / 20.0 * (sum(np.outer(x, x) for x in q) + np.outer(s, s))
    Im = np.trace(J) * np.eye(3) - J
    return v - cm, faces, edges, vol, cm, np.diag(Im) / vol


# --- lissage de Lloyd : chaque germe est deplace au centre de sa cellule ---
for it in range(a.lloyd):
    new = np.array([polyhedron(cell_vertices(i, r=0.0))[4] for i in range(len(seeds))])
    print(f"Lloyd {it + 1} : deplacement moyen des germes {np.linalg.norm(new - seeds, axis=1).mean():.3f} mm")
    seeds = np.clip(new, margin, L - margin)
    tree = cKDTree(seeds)

cells = [polyhedron(cell_vertices(i)) for i in range(len(seeds))]

with open(a.prefix + ".shp", "w") as f:
    for i, (v, faces, edges, vol, cm, im) in enumerate(cells):
        f.write(f"<\nname Voronoi{i}\nradius {r}\npreCompDone y\nnv {len(v)}\n")
        f.writelines(f"{x:.17g} {y:.17g} {z:.17g}\n" for x, y, z in v)
        f.write(f"ne {len(edges)}\n")
        f.writelines(f"{e0} {e1}\n" for e0, e1 in edges)
        f.write(f"nf {len(faces)}\n")
        f.writelines(f"{len(fc)} " + " ".join(map(str, fc)) + "\n" for fc in faces)
        f.write("position 0. 0. 0.\norientation 1.0 0.0 0.0 0.0\n")
        f.write(f"volume {vol:.12g}\nI/m {im[0]:.12g} {im[1]:.12g} {im[2]:.12g}\n>\n")

with open(a.prefix + ".txt", "w") as f:
    f.write("t 0\ndt 5e-05\nperiodicity 0 0 0\n")
    f.write(f"shapeFile {a.prefix}.shp\ndensity 0 0.001\n")
    f.write(f"Particles {len(cells)}\n")
    for i, c in enumerate(cells):
        x, y, z = c[4]
        f.write(f"Voronoi{i} 0 0 1 {x:.17g} {y:.17g} {z:.17g} 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0\n")

vols = np.array([c[3] for c in cells])
print(f"{len(cells)} cellules, volume total {vols.sum():.1f} mm3 "
      f"(bloc {np.prod(L):.1f}), volume moyen {vols.mean():.2f} mm3, "
      f"ecart-type relatif {vols.std() / vols.mean():.2f}")

