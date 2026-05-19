import numpy as np
import sys


def polyhedron_mass_properties(
    vertices,
    faces,
    n_samples=50000,
    density=1.0,
    tol=1e-12
):
    """
    Compute the mass properties (mass, center of mass, inertia tensor) of a polyhedron defined by its vertices and faces using a Monte Carlo method.    
    Parameters
    ----------
    vertices : array-like of shape (n_vertices, 3)
        The coordinates of the vertices of the polyhedron.
    faces : list of lists
        Each face is defined by a list of vertex indices (0-based) that form the face. The faces should be defined in a consistent order (e.g., counterclockwise) to ensure correct normal direction.
    n_samples : int, optional
        The number of random points to sample for the Monte Carlo estimation (default: 50000).
    density : float, optional
        The density of the material (default: 1.0). The mass will be computed as density * volume.
    tol : float, optional
        The tolerance for considering a point as inside the polyhedron (default: 1e-12). This is used to account for numerical precision issues when testing if points are inside the polyhedron. A point is considered inside if it satisfies the plane equations of all faces within this tolerance.  
    Returns
    -------
    mass : float
        The estimated mass of the polyhedron, computed as density * volume.
    center : array of shape (3,)
        The estimated center of mass of the polyhedron.
    inertia : array of shape (3,)
        The estimated inertia tensor of the polyhedron, returned as a 3-element array containing the diagonal elements (Ixx, Iyy, Izz). The products of inertia are assumed to be negligible for the shapes considered, which is a common approximation for convex polyhedra.   
    """

    vertices = np.asarray(vertices)

    # --- 1. Bounding box ---
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    box_size = maxs - mins
    Vbox = np.prod(box_size)

    if Vbox < 1e-14:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    # --- 2. Build plane equations (convex polyhedron assumed) ---
    normals = []
    ds = []

    for face in faces:
        if len(face) < 3:
            continue

        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            continue

        n /= norm
        d = -np.dot(n, v0)

        normals.append(n)
        ds.append(d)

    normals = np.asarray(normals)
    ds = np.asarray(ds)

    if len(normals) < 4:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    # --- 3. radom sampling --- 
    rng = np.random.default_rng(0) 
    u = rng.random((n_samples, 3))

    P = mins + u * box_size

    # --- 4. Inside test ---
    signed = normals @ P.T + ds[:, None]
    inside_mask = np.all(signed <= tol, axis=0)

    inside_pts = P[inside_mask]
    n_inside = inside_pts.shape[0]

    if n_inside == 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    # --- 5. Volume & center of mass ---
    volume = Vbox * n_inside / n_samples
    mass = density * volume
    center = inside_pts.mean(axis=0)

    # --- 6. Inertia tensor ---
    r = inside_pts - center
    x, y, z = r[:, 0], r[:, 1], r[:, 2]

    Ixx = np.sum(y*y + z*z)
    Iyy = np.sum(x*x + z*z)
    Izz = np.sum(x*x + y*y)

    Ixy = -np.sum(x*y)
    Ixz = -np.sum(x*z)
    Iyz = -np.sum(y*z)

    # Normalisation (comme le C++)
    factor = mass / n_inside

    I = factor * np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])

    # --- 7. Diagonalisation (axes principaux) ---
    eigvals, eigvecs = np.linalg.eigh(I)

    return mass, center, eigvals/mass
