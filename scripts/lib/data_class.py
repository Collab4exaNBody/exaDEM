from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional

# ###############################################
# -- ROCKABLE DATA CLASSES --
# ###############################################

# PARTICLE
# -----------------------------------------------
@dataclass
class Particle:
    """
    Particle data class to store the properties of each particle in the system.

    Attributes
    ----------
       name : Optional[str]
           name of the particle
       id : Optional[int]
           unique identifier for the particle
       group : int
           group id of the particle (for interaction purposes)
       cluster : int
           cluster id of the particle (for clustering purposes)
       homothety : float
           homothety factor for scaling the particle

       pos : Optional[Tuple[float,float,float]]
           position of the particle in 3D space
       vel : Optional[Tuple[float,float,float]]
           velocity of the particle in 3D space
       acc : Optional[Tuple[float,float,float]]
           acceleration of the particle in 3D space

       force : Optional[Tuple[float,float,float]]
           total force acting on the particle
       stress : Optional[Tuple[float,float,float,float,float,float]]
           stress tensor components (sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz) for the particle (can be None if not provided)

       quat : Optional[Tuple[float,float,float,float]]
           orientation of the particle as a quaternion
       vrot : Optional[Tuple[float,float,float]]
           rotational velocity of the particle
       arot : Optional[Tuple[float,float,float]]
           rotational acceleration of the particle

       I : Optional[Tuple[float,float,float]]
           principal moments of inertia of the particle (can be None if not provided)
       mass : Optional[float]
           mass of the particle (can be None if not provided)
       volume : Optional[float]
           volume of the particle (can be None if not provided)
    """

    name: Optional[str]=None
    id: Optional[int]=None
    group: int = 0
    cluster: int = 0    
    homothety: float = 1.0

    pos: Optional[Tuple[float, float, float]] = None
    vel: Optional[Tuple[float, float, float]] = None
    acc: Optional[Tuple[float, float, float]] = None

    force: Optional[Tuple[float, float, float]] = None


    quat: Optional[Tuple[float, float, float, float]] = None
    vrot: Optional[Tuple[float, float, float]] = None
    arot: Optional[Tuple[float, float, float]] = None

    stress: Optional[Tuple[float, float, float, float, float, float]] = None
    I: Optional[Tuple[float, float, float]] = None
    mass: Optional[float] = None
    volume: Optional[float] = None



# --- PARAMS ---
# -----------------------------------------------

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
# -----------------------------------------------

InteractionKey = Union[int, Tuple[int, int]]

@dataclass
class InteractionsParameters:
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
    si : int
        subindex of the first particle in the contact (for non-spherical particles)
    sj : int
        subindex of the second particle in the contact (for non-spherical particles)
    type : int
        type of the contact
    pos : Tuple[float, float, float]
        position of the contact point in 3D space
    force : Tuple[float, float, float]
        force vector acting on the contact
    fn : Tuple[float, float, float]
        normal force vector acting on the contact
    ft : Tuple[float, float, float]
        tangential force vector acting on the contact
    pos_i : Tuple[float, float, float]
        position of the first particle in the contact  
    pos_j : Tuple[float, float, float]
        position of the second particle in the contact
    '''
    i: int
    j: int
    si: int
    sj: int
    type: int
    pos: Tuple[float, float, float]
    force: Tuple[float, float, float]
    fn: Tuple[float, float, float]
    ft: Tuple[float, float, float]
    pos_i: Tuple[float, float, float]
    pos_j: Tuple[float, float, float]

@dataclass
class Interactions:
    parameters: InteractionsParameters
    contacts: List[Contact] = field(default_factory=list)


# --- GLOBAL ROCKABLE DATA ---
# -----------------------------------------------

@dataclass
class RockableData:
    """
    RockableData data class to store all the data related to the rockable system.
    
    Attributes
    ----------
    params : Params
       simulation parameters
    interactions : Interactions
        interaction parameters and contact data
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
# -----------------------------------------------

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


# ##############################################
# -- OTHER DATA CLASSES ---
# ##############################################


# -- NEPER CELLS DATA ---
# -----------------------------------------------
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
