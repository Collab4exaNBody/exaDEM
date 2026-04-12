from lib.data_class import (
    Particle, Params, InteractionsParameters,Interactions, 
    RockableData, CellsData, Contact
)
from typing import Optional, List, Dict, Union, Tuple

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
        "periodicity": [0, 1, 0],
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