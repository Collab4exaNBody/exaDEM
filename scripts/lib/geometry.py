import numpy as np
import itertools


def intersect_planes(p1, p2, p3):
    """
    Intersect three planes defined by (n, d) where n is the normal vector and d is the distance to the origin.

    Parameters
    ----------
    p1, p2, p3 : tuples
        tuples of (n, d) where n is the normal vector of the plane and d is the distance to the origin

    Returns
    ----------
    point: array of shape (3,)
        coordinates of the intersection point, or None if the planes do not intersect in a single point
    """

    A = np.vstack([p1[0], p2[0], p3[0]])
    b = np.array([p1[1], p2[1], p3[1]])
    if abs(np.linalg.det(A)) < 1e-12:
        return None
    return np.linalg.solve(A, b)


def inside_all_planes(cell_center, p, planes, tol=1e-10):
    """
    Check if a point p is inside the half-spaces defined by a list of planes (n, d) where n is the normal vector and d is the distance to the origin, with the normals oriented towards the cell center.

    Parameters
    ----------
    cell_center : array
        coordinates of the cell center [x, y, z]
    p : array
        coordinates of the point to check [x, y, z]
    planes : list of tuples (n, d)
        n is the normal vector of the plane and d is the distance to the origin, with n oriented towards the cell center
    tol : float
        tolerance for considering a point as inside the plane (default: 1e-10)

    Returns
    ----------
    bool
        True if the point is inside all planes, False otherwise
    """
    for n, d in planes:
        if np.dot(n, p) - d > tol:
            return False
    return True


def order_face_vertices(face_vids, vertices, n):
    """
    Order the vertices of a face in a consistent way (counterclockwise when looking from outside the cell) given the normal vector of the face.

    Parameters
    ----------
    face_vids : list
        vertex indices of the face
    vertices : array of shape (n_vertices, 3)
        coordinates of the vertices
    n : array of shape (3,)
        normal vector of the face, oriented towards the cell center

    Returns
    ----------
    list of vertex indices
        face ordered in a consistent way (counterclockwise when looking from outside the cell)
    """
    pts = vertices[face_vids]
    center = pts.mean(axis=0)

    u = pts[0] - center
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    angles = [np.arctan2(np.dot(p - center, v), np.dot(p - center, u)) for p in pts]
    return [face_vids[i] for i in np.argsort(angles)]


def unique_points(points, tol=1e-8):
    """
    Return unique points from a list of points, considering two points as identical if they are within a certain distance (tol).

    Parameters
    ----------
    points : list of arrays of shape (3,)
        coordinates of the points
    tol : float
        tolerance for considering two points as identical (default: 1e-8)

    Returns
    ----------
    array of shape (n_unique, 3)
        coordinates of the unique points
    """
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) < tol for q in unique):
            unique.append(p)
    return np.array(unique)


def face_area(vertices, face):
    """
    Compute the area of a face defined by its vertices.

    Parameters
    ----------
    vertices : array of shape (n_vertices, 3)
        coordinates of the vertices
    face : list of vertex indices
        indices of the vertices that define the face

    Returns
    ----------
    area : float
        area of the face
    """
    pts = vertices[face]
    area = 0.0
    for i in range(1, len(face) - 1):
        area += np.linalg.norm(np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])) / 2
    return area


def face_normal(vertices, face):
    """
    Compute the normal vector of a face defined by its vertices, using the first three vertices to define the plane of the face.

    Parameters
    ----------
    vertices : array of shape (n_vertices, 3)
        coordinates of the vertices
    face : list of vertex indices
        indices of the vertices that define the face

    Returns
    ----------
    n : array of shape (3,)
        normal vector of the face, not necessarily oriented towards the cell center
    """
    v0, v1, v2 = vertices[face[:3]]
    n = np.cross(v1 - v0, v2 - v0)
    return n / np.linalg.norm(n)
