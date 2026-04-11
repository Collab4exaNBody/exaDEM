from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional

# ----------------------------
# -- ROCKABLE DATA CLASSES ---
# ----------------------------


# --- PARTICLE ---
@dataclass
class Particle:
    """
    Particle data class to store the properties of each particle in the system.

    Attributes
    ----------
       name : str
           name of the particle
       group : int
           group id of the particle (for interaction purposes)
       cluster : int
           cluster id of the particle (for clustering purposes)
       homothety : float
           homothety factor for scaling the particle

       pos : Tuple[float,float,float]
           position of the particle in 3D space
       vel : Tuple[float,float,float]
           velocity of the particle in 3D space
       acc : Tuple[float,float,float]
           acceleration of the particle in 3D space

       quat : Tuple[float,float,float,float]
           orientation of the particle as a quaternion
       vrot : Tuple[float,float,float]
           rotational velocity of the particle
       arot : Tuple[float,float,float]
           rotational acceleration of the particle
    """

    name: str
    group: int
    cluster: int
    homothety: float

    pos: Tuple[float, float, float]
    vel: Tuple[float, float, float]
    acc: Tuple[float, float, float]

    quat: Tuple[float, float, float, float]
    vrot: Tuple[float, float, float]
    arot: Tuple[float, float, float]


# --- PARAMS ---
@dataclass
class Params:
    """
    Params data class to store the simulation parameters.

    Attributes
    ----------
    values : Dict[str, Union[float, List[float], str]]
        dictionary to store any other parameters needed for the simulation
    """

    values: Dict[str, Union[float, List[float], str]]


# --- INTERACTIONS ---
InteractionKey = Union[int, Tuple[int, int]]


@dataclass
class Interactions:
    """
    Interactions data class to store the interaction parameters between particles.

    Attributes
    ----------
    tables : Dict[str, Dict[InteractionKey, float]]
        dictionary to store interaction parameters for different types of interactions (e.g., 'normal', 'tangential', 'rolling')
    """

    tables: Dict[str, Dict[InteractionKey, float]]


@dataclass
class Contact:
    '''
    Contact data class to store the properties of a contact between two particles.
    Attributes
    ----------
    i : int
        index of the first particle in the contact
    j : int
        index of the second particle in the contact
    type : int
        type of the contact
    force : Tuple[float, float, float]
        force vector acting on the contact
    pos_i : Tuple[float, float, float]
        position of the first particle in the contact  
    pos_j : Tuple[float, float, float]
        position of the second particle in the contact
    '''
    i: int
    j: int
    type: int
    force: Tuple[float, float, float]
    pos_i: Tuple[float, float, float]
    pos_j: Tuple[float, float, float]


# --- GLOBAL ROCKABLE DATA ---
@dataclass
class RockableData:
    """
    RockableData data class to store all the data related to the rockable system.
    
    Attributes
    ----------
    params : Params
       simulation parameters
    interactions : Interactions
        interaction parameters between particles
    particles : List[Particle]
        list of particles in the system
    n_particles : int
        number of particles in the system
    stick_distance : Optional[float]
        distance threshold for sticking particles together (if applicable)
    time : Optional[float]
        current simulation time (can be used for time-dependent parameters or interactions)
    """

    params: Params
    interactions: Interactions
    particles: List[Particle]
    n_particles: int
    stick_distance: Optional[float] = None
    time: Optional[float] = None


# --- SHAPE ---
@dataclass
class Shape:
    """
    Shape data class to store the geometric data of a shape.
    
    Attributes
    ----------
    vertices : List[Tuple[float, float, float]]
        list of vertex coordinates
    edges : List[Tuple[int, int]]
        list of edges defined by their vertex indices
    faces : List[List[int]]
        list of faces defined by their vertex indices
    volume : float
        volume of the shape
    inertia_tensor : List[List[float]]
        inertia tensor of the shape (3x3 matrix)
    """

    vertices: List[Tuple[float, float, float]]
    faces: List[List[int]]
    edges: List[Tuple[int, int]]
    volume: float
    center: Tuple[float, float, float]
    inertia_tensor: List[Tuple[float, float, float]]


@dataclass
class Shapes:
    """
    Shapes data class to store the geometric data of multiple shapes.

    Attributes
    ----------
    shapes : Dict[int, Shape]
        mapping shape ID to its Shape data
    """

    shapes: Dict[int, Shape]


# ----------------------------
# -- OTHER DATA CLASSES ---
# ----------------------------


# -- NEPER CELLS DATA ---
@dataclass
class CellsData:
    """
    NEPER : CellData data class to store the geometric data of each cell in the system.

    Attributes
    ----------
    vertices : dict
        mapping vertex ID to its coordinates [x, y, z]
    edges : dict
        mapping edge ID to its vertex indices (v0, v1)
    faces : dict
        mapping face ID to its vertex indices [v0, v1, ...]
    face_normals : dict
       mapping face ID to its normal vector (ax, ay, az, d) for the plane equation ax + by + cz + d = 0
    polyhedra : dict
        mapping polyhedron ID to its face indices [face_id0, face_id1, ...]
    radius : float
        radius of the particles (for stickVerticesInClusters)
    gap : float
         gap to apply for the Minkowski erosion (should be >= 2*radius)
    """

    vertices: Dict[int, List[float]]
    edges: Dict[int, Tuple[int, int]]
    faces: Dict[int, List[int]]
    face_normals: Dict[int, Tuple[float, float, float, float]]
    polyhedra: Dict[int, List[int]]
    radius: float
