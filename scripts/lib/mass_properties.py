import numpy as np
import sys


def polyhedron_mass_properties_mc_fast(
    vertices, cell_faces, n_samples=200_000, density=1.0, seed=0, tol=1e-12
):
    """
    Compute the volume, center of mass and inertia tensor of a polyhedron defined by its vertices and faces, using a Monte Carlo method.
    Unlike the polyhedron_mass_properties_mc function, this version is optimized for speed by using vectorized operations and returning only the diagonal elements of the inertia tensor.
    
    Parameters
    ----------
    vertices : array of shape (n_vertices, 3)
        coordinates of the vertices
    cell_faces : list
        each face is a list of vertex indices
    n_samples : int
        number of random points to generate for the estimation
    density : float
        density of the material (to compute mass from volume)
    seed : float
        seed for the random number generator
    tol : float
        tolerance for considering a point as inside the polyhedron (default: 1e-12)

    Returns
    ----------
    volume : float
        estimated volume of the polyhedron
    center : array of shape (3,)
        estimated center of mass of the polyhedron
    I : array of shape (3,)
        estimated inertia tensor of the polyhedron (as a 3-element array with the diagonal elements Ixx, Iyy, Izz)
        Note: the inertia tensor is returned as a 3-element array with the diagonal elements Ixx, Iyy, Izz, assuming the products of inertia are negligible for the shapes considered. This is a common approximation for convex polyhedra
    """
    rng = np.random.default_rng(seed)

    vert_ids = sorted({v for face in cell_faces for v in face})
    pts = np.array([vertices[v] for v in vert_ids])

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    box_volume = np.prod(maxs - mins)

    if box_volume < 1e-14:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    normals, ds = [], []

    for face in cell_faces:
        if len(face) < 3:
            continue

        v0 = np.asarray(vertices[face[0]])
        v1 = np.asarray(vertices[face[1]])
        v2 = np.asarray(vertices[face[2]])

        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            continue

        n /= norm
        d = -np.dot(n, v0)

        normals.append(n)
        ds.append(d)

    if len(normals) < 4:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    normals = np.asarray(normals)
    ds = np.asarray(ds)

    P = rng.uniform(mins, maxs, size=(n_samples, 3))
    signed = normals @ P.T + ds[:, None]
    inside_mask = np.all(signed <= tol, axis=0)

    inside_pts = P[inside_mask]
    inside = inside_pts.shape[0]

    if inside == 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    volume = box_volume * inside / n_samples
    mass = density * volume
    center = inside_pts.mean(axis=0)

    r = inside_pts - center
    r2 = np.sum(r * r, axis=1)

    # I = np.zeros((3, 3))
    I = np.zeros(3)
    # I[0, 0] = np.sum(r2 - r[:, 0] ** 2)
    # I[1, 1] = np.sum(r2 - r[:, 1] ** 2)
    # I[2, 2] = np.sum(r2 - r[:, 2] ** 2)

    # I[0, 1] = I[1, 0] = -np.sum(r[:, 0] * r[:, 1])
    # I[0, 2] = I[2, 0] = -np.sum(r[:, 0] * r[:, 2])
    # I[1, 2] = I[2, 1] = -np.sum(r[:, 1] * r[:, 2])

    I[0] = np.sum(r2 - r[:, 0] ** 2)
    I[1] = np.sum(r2 - r[:, 1] ** 2)
    I[2] = np.sum(r2 - r[:, 2] ** 2)

    I *= mass / inside
    return mass, center, I / mass


def polyhedron_mass_properties_mc(
    vertices, cell_faces, n_samples=20000, density=1.0, seed=0
):
    """
    Compute the volume, center of mass and inertia tensor of a polyhedron defined by its vertices and faces, using a Monte Carlo method.

    Parameters
    ----------
    vertices : array of shape (n_vertices, 3)
        coordinates of the vertices
    cell_faces : list
        each face is a list of vertex indices
    n_samples : int
        number of random points to generate for the estimation
    density : float
        density of the material (to compute mass from volume)
    seed : float
        seed for the random number generator
    tol : float
        tolerance for considering a point as inside the polyhedron (default: 1e-12)

    Returns
    ----------
    volume : float
        estimated volume of the polyhedron
    center : array of shape (3,)
        estimated center of mass of the polyhedron
    I : array of shape (3,)
        estimated inertia tensor of the polyhedron (as a 3-element array with the diagonal elements Ixx, Iyy, Izz)
        Note: the inertia tensor is returned as a 3-element array with the diagonal elements Ixx, Iyy, Izz, assuming the products of inertia are negligible for the shapes considered. This is a common approximation for convex polyhedra
    """
    rng = np.random.default_rng(seed)

    vert_ids = sorted({v for face in cell_faces for v in face})
    pts = np.array([vertices[v] for v in vert_ids])

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    box_volume = np.prod(maxs - mins)

    if box_volume < 1e-14:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    planes = []
    for face in cell_faces:
        if len(face) < 3:
            continue

        v0 = np.array(vertices[face[0]])
        v1 = np.array(vertices[face[1]])
        v2 = np.array(vertices[face[2]])

        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            continue

        n /= norm
        d = -np.dot(n, v0)
        planes.append((n, d))

    if len(planes) < 4:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    inside = 0
    C = np.zeros(3)
    I = np.zeros((3, 3))

    for _ in range(n_samples):
        p = rng.uniform(mins, maxs)
        if all(np.dot(n, p) + d <= 1e-12 for n, d in planes):
            inside += 1
            C += p

    if inside == 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    volume = box_volume * inside / n_samples
    mass = density * volume
    center = C / inside

    for _ in range(n_samples):
        p = rng.uniform(mins, maxs)
        if all(np.dot(n, p) + d <= 1e-12 for n, d in planes):
            r = p - center
            I += np.dot(r, r) * np.eye(3) - np.outer(r, r)

    I *= mass / inside
    return mass, center, I / mass
