from collections import defaultdict
import re
import glob
from lib.data_class import (
    Particle, Params, InteractionsParameters,Interactions, 
    RockableData, CellsData, Contact
)
from typing import Optional, List, Dict, Union, Tuple


# ##############################################
# -- DATA UTILITIES FOR EXADEM/ROCKABLE --
# ##############################################

# --- ROCKABLE DATA CREATION ---
# -----------------------------------------------
def make_rockable_data(    
    params: Optional[Dict[str, Union[float, List[float], str]]] = None,
    interactions: Optional[Dict[str, Dict]] = None,
    particles: Optional[List[Particle]] = None,
    stick_distance: Optional[float] = None,
) -> RockableData:
    '''
Create an empty RockableData object with default values.
This can be used as a starting point for building a RockableData object with specific parameters

Parameters
---------- 
params : dict
    dictionary of simulation parameters (optional)
interactions : dict
    dictionary of interaction parameters (optional)
particles : list
    list of Particle objects (optional)
stick_distance : float
    distance threshold for sticking particles together (optional)

Returns
---------- 
RockableData
    an empty RockableData object with default values
    '''
    return RockableData(
        params=Params(values=params or {}),
        interactions=Interactions(
            parameters=InteractionsParameters(tables=interactions or {}),
            contacts=[]
        ),
        particles=particles or [],
        n_particles=len(particles) if particles else 0,
        stick_distance=stick_distance,
    )

# --- STICKED PARTICLES CONFIGURATION ---
# -----------------------------------------------
def make_sticked_conf(cell_centers, shapefile, gap):
    '''
    Create a RockableData object with the parameters, interactions and particles needed for a simulation with sticked particles.
    The particles are created at the positions of the cell centers, and the interactions are defined to allow sticking between particles that are within a certain distance (stick_distance).

    Parameters
    ----------
    cell_centers : dict 
        mapping cell ID to its center coordinates (x, y, z)
    shapefile : str 
        path to the shapefile containing the cell geometries (used for visualization and interaction definitions)
    gap: float 
        distance threshold for sticking particles together (should be >= 2*radius of the particles)

    Returns
    ----------
    RockableData 
        a RockableData object with the parameters, interactions and particles defined for a sticked particle simulation
    '''
    data = make_rockable_data()

    # --- PARAMS ---
    data.params.values = {
        "t": 0,
        "tmax": 2,
        "dt": 1e-6,
        "interVerlet": 1e-4,
        "interConf": 0.01,
        "DVerlet": 0.02,
        "dVerlet": 0.01,
        "gravity": [0, 0, -9.81],
        "forceLaw": "StickedLinks",
        "periodicity": [0, 0, 0],
        "shapeFile": shapefile,
    }

    # --- INTERACTIONS ---
    data.interactions.parameters.tables = {
        "density": {0: 0.006, 1: 0.006},
        "knContact": {(0, 0): 1e4, (0, 1): 1e4},
        "en2Contact": {(0, 0): 0.001, (0, 1): 0.001},
        "ktContact": {(0, 0): 8e3, (0, 1): 8e3},
        "muContact": {(0, 0): 0.3, (0, 1): 0.3},
        "knInnerBond": {(0, 0): 1e4},
        "ktInnerBond": {(0, 0): 8e3},
        "en2InnerBond": {(0, 0): 1e-4},
        "powInnerBond": {(0, 0): 2.0},
        "GInnerBond": {(0, 0): 2e-4},
    }

    # --- PARTICLES ---
    for pid, (x, y, z) in cell_centers.items():
        data.particles.append(Particle(
            name=f"Voronoi{pid}",
            group=1,
            cluster=0,
            homothety=1,
            pos=(x, y, z),
            vel=(0, 0, 0),
            acc=(0, 0, 0),
            quat=(1, 0, 0, 0),
            vrot=(0, 0, 0),
            arot=(0, 0, 0),
        ))
 
    data.n_particles = len(data.particles)

    # --- STICK ---
    data.stick_distance  = 4 * gap

    return data

# --- PARSING UTILITIES ---
# -----------------------------------------------
def parse_time(header_line):
    '''
    Parse the time from a header line in the ExaDEM output file.
    
    Parameters
    ----------
    header_line : str
        the header line containing the time information (e.g., "Time=0.001")
    
    Returns
    ----------
    float
        the parsed time value, or None if the time information is not found in the header line
    '''
    match = re.search(r"Time=([0-9.eE+-]+)", header_line)
    return float(match.group(1)) if match else None
    

def parse_properties(header_line):
    '''
    Parse the properties from a header line in the ExaDEM output file.
    Parameters
    ----------
    header_line : str
        the header line containing the properties information (e.g., "Properties=pos:float:3:vel:float:3:acc:float:3")  
    Returns
    ----------
    list of tuples
        a list of tuples containing the property name, type and dimension (e.g., [("pos", "float", 3), ("vel", "float", 3), ("acc",
        "float", 3)])
    '''

    import re
    
    match = re.search(r'Properties=([^ ]+)', header_line)
    if not match:
        raise ValueError("No Properties field found")

    tokens = match.group(1).split(':')
    
    props = []
    i = 0
    while i < len(tokens):
        name = tokens[i]
        typ = tokens[i+1]
        dim = int(tokens[i+2])
        props.append((name, typ, dim))
        i += 3

    return props


def parse_vec3(string):
    '''Parse a string containing three float values into a tuple of three floats.
    
    Parameters
    ----------
    string : str
        the string containing the three float values (e.g., "(1.0, 2.0, 3.0)")

    Returns
    ----------
    np.ndarray
        the parsed vector as a numpy array
    '''
    string = string.strip().replace("(", "").replace(")", "")
    return np.array([float(x) for x in string.split()])


# --- PARTICLE INDEXING AND CLUSTERING ---
# -----------------------------------------------

def build_particle_index(particles):
    """Map particle id -> Particle
    
    Parameters
    ----------
    particles : List[Particle]
        list of Particle objects to index
        
    Returns
    ----------
    Dict[int, Particle]
        dictionary mapping particle id to Particle object
    """
    return {p.id: p for p in particles}


def build_clusters(particles):
    """Group particles by cluster id
    
    Parameters
    ----------
    particles : List[Particle]
        list of Particle objects to group
        
    Returns
    ----------
    Dict[int, List[Particle]]
        dictionary mapping cluster id to list of Particle objects
    """
    clusters = defaultdict(list)
    for p in particles:
        clusters[p.cluster].append(p)
    return clusters


def build_index_map(props):
    '''
    Build an index map from property names to their corresponding indices in the ExaDEM output file.
    Parameters
    ----------
    props : list of tuples
        a list of tuples containing the property name, type and dimension (e.g., [("pos", "float", 3), ("vel", "float", 3), ("acc",
        "float", 3)])
    Returns 
    ----------
    dict
        a dictionary mapping property names to their corresponding index ranges in the ExaDEM output file (e.g., {"pos": (0, 3), "vel": (3, 6), "acc": (6, 9)})
    '''
    index_map = {}
    current = 0

    for name, typ, dim in props:
        index_map[name] = (current, current + dim)
        current += dim

    return index_map

def extract_field(data, index_map, field, cast=float):
    '''
    Extract a specific field from a line of data in the ExaDEM output file using the index map.
    Parameters
    ----------
    data : list of str
        the line of data from which to extract the field, split into a list of strings
    index_map : dict
        a dictionary mapping property names to their corresponding index ranges in the ExaDEM output file (e.g., {"pos": (0, 3), "vel": (3, 6), "acc": (6, 9)})
    field : str
        the name of the field to extract (e.g., "pos", "vel", "acc")    
    cast : function
        a function to cast the extracted values to the desired type (default is float)
    Returns
    ----------
    list or single value
        the extracted field values, cast to the desired type; returns a list if the field has multiple dimensions, or a single value if it has only one dimension; returns None if the field is not found in the index map
    '''
    if field not in index_map:
        return None
    
    start, end = index_map[field]
    values = data[start:end]

    if cast:
        values = list(map(cast, values))

    return values if len(values) > 1 else values[0]
