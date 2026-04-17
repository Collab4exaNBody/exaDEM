Scripts ExaDEM Docs






[Skip to content](#documentation-scripts-exadem)

Scripts ExaDEM Docs

Home





Initializing search

Scripts ExaDEM Docs

* Home

  [Home](.)



  Table of contents
  + [TOP](#top)
  + [data\_utils](#lib.data_utils)
  + [build\_clusters](#lib.data_utils.build_clusters)
  + [build\_particle\_index](#lib.data_utils.build_particle_index)
  + [make\_rockable\_data](#lib.data_utils.make_rockable_data)
  + [make\_sticked\_conf](#lib.data_utils.make_sticked_conf)
  + [parse\_time](#lib.data_utils.parse_time)
  + [parse\_vec3](#lib.data_utils.parse_vec3)
  + [geometry](#lib.geometry)
  + [face\_area](#lib.geometry.face_area)
  + [face\_normal](#lib.geometry.face_normal)
  + [inside\_all\_planes](#lib.geometry.inside_all_planes)
  + [intersect\_planes](#lib.geometry.intersect_planes)
  + [order\_face\_vertices](#lib.geometry.order_face_vertices)
  + [unique\_points](#lib.geometry.unique_points)
  + [io\_utils](#lib.io_utils)
  + [read\_interactions](#lib.io_utils.read_interactions)
  + [read\_rockable\_file](#lib.io_utils.read_rockable_file)
  + [read\_tess](#lib.io_utils.read_tess)
  + [read\_xyzdem\_snapshot](#lib.io_utils.read_xyzdem_snapshot)
  + [write\_rockable\_file](#lib.io_utils.write_rockable_file)
  + [write\_shp\_file](#lib.io_utils.write_shp_file)
  + [data\_utils](#lib.data_utils)
  + [build\_clusters](#lib.data_utils.build_clusters)
  + [build\_particle\_index](#lib.data_utils.build_particle_index)
  + [make\_rockable\_data](#lib.data_utils.make_rockable_data)
  + [make\_sticked\_conf](#lib.data_utils.make_sticked_conf)
  + [parse\_time](#lib.data_utils.parse_time)
  + [parse\_vec3](#lib.data_utils.parse_vec3)
  + [data\_class](#lib.data_class)
  + [CellsData](#lib.data_class.CellsData)
  + [Contact](#lib.data_class.Contact)
  + [InteractionsParameters](#lib.data_class.InteractionsParameters)
  + [Params](#lib.data_class.Params)
  + [Particle](#lib.data_class.Particle)
  + [RockableData](#lib.data_class.RockableData)
  + [Shape](#lib.data_class.Shape)
  + [Shapes](#lib.data_class.Shapes)
  + [mass\_properties](#lib.mass_properties)
  + [polyhedron\_mass\_properties\_mc](#lib.mass_properties.polyhedron_mass_properties_mc)
  + [polyhedron\_mass\_properties\_mc\_fast](#lib.mass_properties.polyhedron_mass_properties_mc_fast)
  + [topology](#lib.topology)
  + [check\_interface](#lib.topology.check_interface)
  + [check\_interfaces](#lib.topology.check_interfaces)
  + [count\_interfaces](#lib.topology.count_interfaces)
  + [get\_interfaces](#lib.topology.get_interfaces)
* [Geometry](geometry/)
* [IO](io_utils/)
* [Data\_Utils](data_utils/)
* [Data\_Class](data_class/)
* [Mass\_properties](mass_properties/)
* [topology](topology/)

Table of contents

* [TOP](#top)
* [data\_utils](#lib.data_utils)
* [build\_clusters](#lib.data_utils.build_clusters)
* [build\_particle\_index](#lib.data_utils.build_particle_index)
* [make\_rockable\_data](#lib.data_utils.make_rockable_data)
* [make\_sticked\_conf](#lib.data_utils.make_sticked_conf)
* [parse\_time](#lib.data_utils.parse_time)
* [parse\_vec3](#lib.data_utils.parse_vec3)
* [geometry](#lib.geometry)
* [face\_area](#lib.geometry.face_area)
* [face\_normal](#lib.geometry.face_normal)
* [inside\_all\_planes](#lib.geometry.inside_all_planes)
* [intersect\_planes](#lib.geometry.intersect_planes)
* [order\_face\_vertices](#lib.geometry.order_face_vertices)
* [unique\_points](#lib.geometry.unique_points)
* [io\_utils](#lib.io_utils)
* [read\_interactions](#lib.io_utils.read_interactions)
* [read\_rockable\_file](#lib.io_utils.read_rockable_file)
* [read\_tess](#lib.io_utils.read_tess)
* [read\_xyzdem\_snapshot](#lib.io_utils.read_xyzdem_snapshot)
* [write\_rockable\_file](#lib.io_utils.write_rockable_file)
* [write\_shp\_file](#lib.io_utils.write_shp_file)
* [data\_utils](#lib.data_utils)
* [build\_clusters](#lib.data_utils.build_clusters)
* [build\_particle\_index](#lib.data_utils.build_particle_index)
* [make\_rockable\_data](#lib.data_utils.make_rockable_data)
* [make\_sticked\_conf](#lib.data_utils.make_sticked_conf)
* [parse\_time](#lib.data_utils.parse_time)
* [parse\_vec3](#lib.data_utils.parse_vec3)
* [data\_class](#lib.data_class)
* [CellsData](#lib.data_class.CellsData)
* [Contact](#lib.data_class.Contact)
* [InteractionsParameters](#lib.data_class.InteractionsParameters)
* [Params](#lib.data_class.Params)
* [Particle](#lib.data_class.Particle)
* [RockableData](#lib.data_class.RockableData)
* [Shape](#lib.data_class.Shape)
* [Shapes](#lib.data_class.Shapes)
* [mass\_properties](#lib.mass_properties)
* [polyhedron\_mass\_properties\_mc](#lib.mass_properties.polyhedron_mass_properties_mc)
* [polyhedron\_mass\_properties\_mc\_fast](#lib.mass_properties.polyhedron_mass_properties_mc_fast)
* [topology](#lib.topology)
* [check\_interface](#lib.topology.check_interface)
* [check\_interfaces](#lib.topology.check_interfaces)
* [count\_interfaces](#lib.topology.count_interfaces)
* [get\_interfaces](#lib.topology.get_interfaces)

Documentation scripts ExaDEM
============================

TOP
---

`build_clusters(particles)`
---------------------------

Group particles by cluster id

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `particles` | `List[Particle]` | list of Particle objects to group | *required* |

Returns:

| Type | Description |
| --- | --- |
| `Dict[int, List[Particle]]` | dictionary mapping cluster id to list of Particle objects |

`build_particle_index(particles)`
---------------------------------

Map particle id -> Particle

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `particles` | `List[Particle]` | list of Particle objects to index | *required* |

Returns:

| Type | Description |
| --- | --- |
| `Dict[int, Particle]` | dictionary mapping particle id to Particle object |

`make_rockable_data(params=None, interactions=None, particles=None, stick_distance=None)`
-----------------------------------------------------------------------------------------

Create an empty RockableData object with default values.
This can be used as a starting point for building a RockableData object with specific parameters

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `params` | `dict` | dictionary of simulation parameters (optional) | `None` |
| `interactions` | `dict` | dictionary of interaction parameters (optional) | `None` |
| `particles` | `list` | list of Particle objects (optional) | `None` |
| `stick_distance` | `float` | distance threshold for sticking particles together (optional) | `None` |

Returns:

| Type | Description |
| --- | --- |
| `RockableData` | an empty RockableData object with default values |

`make_sticked_conf(cell_centers, shapefile, gap)`
-------------------------------------------------

Create a RockableData object with the parameters, interactions and particles needed for a simulation with sticked particles.
The particles are created at the positions of the cell centers, and the interactions are defined to allow sticking between particles that are within a certain distance (stick\_distance).

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cell_centers` | `dict` | mapping cell ID to its center coordinates (x, y, z) | *required* |
| `shapefile` | `str` | path to the shapefile containing the cell geometries (used for visualization and interaction definitions) | *required* |
| `gap` |  | distance threshold for sticking particles together (should be >= 2\*radius of the particles) | *required* |

Returns:

| Type | Description |
| --- | --- |
| `RockableData` | a RockableData object with the parameters, interactions and particles defined for a sticked particle simulation |

`parse_time(header_line)`
-------------------------

Parse the time from a header line in the ExaDEM output file.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `header_line` | `str` | the header line containing the time information (e.g., "Time=0.001") | *required* |

Returns:

| Type | Description |
| --- | --- |
| `float` | the parsed time value, or None if the time information is not found in the header line |

`parse_vec3(string)`
--------------------

Parse a string containing three float values into a tuple of three floats.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `string` | `str` | the string containing the three float values (e.g., "(1.0, 2.0, 3.0)") | *required* |

Returns:

| Type | Description |
| --- | --- |
| `ndarray` | the parsed vector as a numpy array |

`face_area(vertices, face)`
---------------------------

Compute the area of a face defined by its vertices.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `vertices` | `array of shape (n_vertices, 3)` | coordinates of the vertices | *required* |
| `face` | `list of vertex indices` | indices of the vertices that define the face | *required* |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `area` | `float` | area of the face |

`face_normal(vertices, face)`
-----------------------------

Compute the normal vector of a face defined by its vertices, using the first three vertices to define the plane of the face.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `vertices` | `array of shape (n_vertices, 3)` | coordinates of the vertices | *required* |
| `face` | `list of vertex indices` | indices of the vertices that define the face | *required* |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `n` | `array of shape (3,)` | normal vector of the face, not necessarily oriented towards the cell center |

`inside_all_planes(cell_center, p, planes, tol=1e-10)`
------------------------------------------------------

Check if a point p is inside the half-spaces defined by a list of planes (n, d) where n is the normal vector and d is the distance to the origin, with the normals oriented towards the cell center.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cell_center` | `array` | coordinates of the cell center [x, y, z] | *required* |
| `p` | `array` | coordinates of the point to check [x, y, z] | *required* |
| `planes` | `list of tuples (n, d)` | n is the normal vector of the plane and d is the distance to the origin, with n oriented towards the cell center | *required* |
| `tol` | `float` | tolerance for considering a point as inside the plane (default: 1e-10) | `1e-10` |

Returns:

| Type | Description |
| --- | --- |
| `bool` | True if the point is inside all planes, False otherwise |

`intersect_planes(p1, p2, p3)`
------------------------------

Intersect three planes defined by (n, d) where n is the normal vector and d is the distance to the origin.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `p1` | `tuples` | tuples of (n, d) where n is the normal vector of the plane and d is the distance to the origin | *required* |
| `p2` | `tuples` | tuples of (n, d) where n is the normal vector of the plane and d is the distance to the origin | *required* |
| `p3` | `tuples` | tuples of (n, d) where n is the normal vector of the plane and d is the distance to the origin | *required* |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `point` | `array of shape (3,)` | coordinates of the intersection point, or None if the planes do not intersect in a single point |

`order_face_vertices(face_vids, vertices, n)`
---------------------------------------------

Order the vertices of a face in a consistent way (counterclockwise when looking from outside the cell) given the normal vector of the face.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `face_vids` | `list` | vertex indices of the face | *required* |
| `vertices` | `array of shape (n_vertices, 3)` | coordinates of the vertices | *required* |
| `n` | `array of shape (3,)` | normal vector of the face, oriented towards the cell center | *required* |

Returns:

| Type | Description |
| --- | --- |
| `list of vertex indices` | face ordered in a consistent way (counterclockwise when looking from outside the cell) |

`unique_points(points, tol=1e-08)`
----------------------------------

Return unique points from a list of points, considering two points as identical if they are within a certain distance (tol).

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `points` | `list of arrays of shape (3,)` | coordinates of the points | *required* |
| `tol` | `float` | tolerance for considering two points as identical (default: 1e-8) | `1e-08` |

Returns:

| Type | Description |
| --- | --- |
| `array of shape (n_unique, 3)` | coordinates of the unique points |

`read_interactions(filename)`
-----------------------------

Read an interactions file and return a list of Contact objects representing the interactions between particles.
The function parses the interactions file, extracting the relevant data for each contact, and organizes it
into a list of Contact objects for use in simulations.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `filename` | `str` | path to the interactions file, with the following format: i j type fx fy fz pos\_i\_x pos\_i\_y pos\_i\_z pos\_j\_x pos\_j\_y pos\_j\_z | *required* |

Returns:

| Type | Description |
| --- | --- |
| `list[Contact]` | a list of Contact objects, each containing the properties of a contact between two particles, including the indices of the particles involved, the type of contact, the force vector, and the positions of the contact points on each particle. |

`read_rockable_file(filepath)`
------------------------------

read a Rockable configuration file and return a structured data object.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `filepath` | `str` | path to the Rockable configuration file | *required* |

Returns:

| Type | Description |
| --- | --- |
| `RockableData` | structured data object containing all the information from the configuration file |

`read_tess(filename, radius=0.0)`
---------------------------------

Read a Neper .tess file and return structured data for vertices, edges, faces, face normals, and polyhedra.
The function parses the .tess file, extracting the relevant sections for vertices, edges, faces, and polyhedra, and organizes the data into dictionaries for easy access.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `filename` | `str` | path to the Neper .tess file to be read | *required* |
| `radius` | `float` | minkowski radius of the particles | `0.0` |

Returns:

| Type | Description |
| --- | --- |
| `CellsData` | a structured data object containing the vertices, edges, faces, face normals, and polyhedra defined in the .tess file, organized as follows: CellsData(vertices={vertex\_id: [x, y, z], ...}, edges={edge\_id: (v0, v1), ...}, faces={face\_id: [v0, v1, ...], ...}, face\_normals={face\_id: (a, b, c, d), ...}, polyhedra={polyhedron\_id: [face\_id0, face\_id1, ...], ...}, radius=radius) |

`read_xyzdem_snapshot(particle_file, interaction_file=None)`
------------------------------------------------------------

Read a DEM snapshot from a text file and return a structured RockableData object.
The function parses the particle data from the given file, extracting the particle properties and optionally the
interaction parameters from a separate file, and organizes the data into a RockableData object for use in simulations.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `particle_file` | `str` | path to the text file containing the particle data, with the following format: n\_particles header line (ignored) name group cluster homothety pos\_x pos\_y pos\_z vel\_x vel\_y vel\_z acc\_x acc\_y acc\_z quat\_w quat\_x quat\_y quat\_z vrot\_x vrot\_y vrot\_z arot\_x arot\_y arot\_z pid | *required* |
| `interaction_file` | `str` | path to the text file containing the interaction parameters, with a format that can be parsed by the read\_interactions function (default: None, meaning no interactions will be read) | `None` |

Returns:

| Type | Description |
| --- | --- |
| `RockableData` | a RockableData object containing the parameters, interactions and particles defined in the input files, structured as follows: RockableData(params=Params({...}), interactions=Interactions({...}), particles=[Particle(...)], n\_particles=int, stick\_distance=None, time=float or None) |

`write_rockable_file(filepath, data)`
-------------------------------------

Write a Rockable configuration file from a structured data object.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `filepath` | `str` | path to the output Rockable configuration file | *required* |
| `data` | `RockableData` | data object containing all the information to be written in the configuration file, structured as follows: data = {"params": {...},"interactions": {...},"particles": [...],"n\_particles": int,"stick\_distance": float or None}  Returns | *required* |
| `None` |  | The function writes the configuration file in the format expected by Rockable, with sections for parameters, interactions and particles, and includes the stickVerticesInClusters line if stick\_distance is provided. | *required* |

`write_shp_file(shapes, outfile, radius)`
-----------------------------------------

Write a shapefile compatible with Rockable, containing the geometric data of the cells.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `shapes` | `Shapes` | data class containing the geometric data of the cells (vertices, faces, edges, volume, inertia tensor) | *required* |
| `outfile` | `str` | path to the output shapefile | *required* |
| `radius` | `float` | Minkowski radius | *required* |

Returns:

| Type | Description |
| --- | --- |
| `None` | The function writes a shapefile in the format expected by Rockable, with sections for each cell containing its vertices, edges, faces, volume and inertia tensor. The Minkowski radius is included as a parameter for each cell. |

`build_clusters(particles)`
---------------------------

Group particles by cluster id

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `particles` | `List[Particle]` | list of Particle objects to group | *required* |

Returns:

| Type | Description |
| --- | --- |
| `Dict[int, List[Particle]]` | dictionary mapping cluster id to list of Particle objects |

`build_particle_index(particles)`
---------------------------------

Map particle id -> Particle

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `particles` | `List[Particle]` | list of Particle objects to index | *required* |

Returns:

| Type | Description |
| --- | --- |
| `Dict[int, Particle]` | dictionary mapping particle id to Particle object |

`make_rockable_data(params=None, interactions=None, particles=None, stick_distance=None)`
-----------------------------------------------------------------------------------------

Create an empty RockableData object with default values.
This can be used as a starting point for building a RockableData object with specific parameters

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `params` | `dict` | dictionary of simulation parameters (optional) | `None` |
| `interactions` | `dict` | dictionary of interaction parameters (optional) | `None` |
| `particles` | `list` | list of Particle objects (optional) | `None` |
| `stick_distance` | `float` | distance threshold for sticking particles together (optional) | `None` |

Returns:

| Type | Description |
| --- | --- |
| `RockableData` | an empty RockableData object with default values |

`make_sticked_conf(cell_centers, shapefile, gap)`
-------------------------------------------------

Create a RockableData object with the parameters, interactions and particles needed for a simulation with sticked particles.
The particles are created at the positions of the cell centers, and the interactions are defined to allow sticking between particles that are within a certain distance (stick\_distance).

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cell_centers` | `dict` | mapping cell ID to its center coordinates (x, y, z) | *required* |
| `shapefile` | `str` | path to the shapefile containing the cell geometries (used for visualization and interaction definitions) | *required* |
| `gap` |  | distance threshold for sticking particles together (should be >= 2\*radius of the particles) | *required* |

Returns:

| Type | Description |
| --- | --- |
| `RockableData` | a RockableData object with the parameters, interactions and particles defined for a sticked particle simulation |

`parse_time(header_line)`
-------------------------

Parse the time from a header line in the ExaDEM output file.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `header_line` | `str` | the header line containing the time information (e.g., "Time=0.001") | *required* |

Returns:

| Type | Description |
| --- | --- |
| `float` | the parsed time value, or None if the time information is not found in the header line |

`parse_vec3(string)`
--------------------

Parse a string containing three float values into a tuple of three floats.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `string` | `str` | the string containing the three float values (e.g., "(1.0, 2.0, 3.0)") | *required* |

Returns:

| Type | Description |
| --- | --- |
| `ndarray` | the parsed vector as a numpy array |

`CellsData`

`dataclass`
------------------------

NEPER : CellData data class to store the geometric data of each cell in the system.

Attributes:

| Name | Type | Description |
| --- | --- | --- |
| `vertices` | `dict` | mapping vertex ID to its coordinates [x, y, z] |
| `edges` | `dict` | mapping edge ID to its vertex indices (v0, v1) |
| `faces` | `dict` | mapping face ID to its vertex indices [v0, v1, ...] |
| `face_normals` | `dict` | mapping face ID to its normal vector (ax, ay, az, d) for the plane equation ax + by + cz + d = 0 |
| `polyhedra` | `dict` | mapping polyhedron ID to its face indices [face\_id0, face\_id1, ...] |
| `radius` | `float` | radius of the particles (for stickVerticesInClusters) |
| `gap` | `float` | gap to apply for the Minkowski erosion (should be >= 2\*radius) |

`Contact`

`dataclass`
----------------------

Contact data class to store the properties of a contact between two particles.

Attributes:

| Name | Type | Description |
| --- | --- | --- |
| `i` | `int` | index of the first particle in the contact |
| `j` | `int` | index of the second particle in the contact |
| `type` | `int` | type of the contact |
| `force` | `Tuple[float, float, float]` | force vector acting on the contact |
| `pos_i` | `Tuple[float, float, float]` | position of the first particle in the contact |
| `pos_j` | `Tuple[float, float, float]` | position of the second particle in the contact |

`InteractionsParameters`

`dataclass`
-------------------------------------

Interactions data class to store the interaction parameters between particles.

Attributes:

| Name | Type | Description |
| --- | --- | --- |
| `tables` | `Dict[str, Dict[InteractionKey, float]]` | dictionary to store interaction parameters for different types of interactions (e.g., 'normal', 'tangential', 'rolling') |

`Params`

`dataclass`
---------------------

Params data class to store the simulation parameters.

Attributes:

| Name | Type | Description |
| --- | --- | --- |
| `values` | `Dict[str, Union[float, List[float], str]]` | dictionary to store any other parameters needed for the simulation |

`Particle`

`dataclass`
-----------------------

Particle data class to store the properties of each particle in the system.

Attributes:

| Name | Type | Description |
| --- | --- | --- |
| `name` | `str` | name of the particle id : int unique identifier for the particle group : int group id of the particle (for interaction purposes) cluster : int cluster id of the particle (for clustering purposes) homothety : float homothety factor for scaling the particle  pos : Tuple[float,float,float] position of the particle in 3D space vel : Tuple[float,float,float] velocity of the particle in 3D space acc : Tuple[float,float,float] acceleration of the particle in 3D space  quat : Tuple[float,float,float,float] orientation of the particle as a quaternion vrot : Tuple[float,float,float] rotational velocity of the particle arot : Tuple[float,float,float] rotational acceleration of the particle mass : Optional[float] mass of the particle (can be None if not provided) volume : Optional[float] volume of the particle (can be None if not provided) |

`RockableData`

`dataclass`
---------------------------

RockableData data class to store all the data related to the rockable system.

Attributes:

| Name | Type | Description |
| --- | --- | --- |
| `params` | `Params` | simulation parameters |
| `interactions` | `Interactions` | interaction parameters and contact data |
| `particles` | `List[Particle]` | list of particles in the system |
| `n_particles` | `int` | number of particles in the system |
| `stick_distance` | `Optional[float]` | distance threshold for sticking particles together (if applicable) |
| `time` | `Optional[float]` | current simulation time (can be used for time-dependent parameters or interactions) |

`Shape`

`dataclass`
--------------------

Shape data class to store the geometric data of a shape.

Attributes:

| Name | Type | Description |
| --- | --- | --- |
| `vertices` | `List[Tuple[float, float, float]]` | list of vertex coordinates |
| `edges` | `List[Tuple[int, int]]` | list of edges defined by their vertex indices |
| `faces` | `List[List[int]]` | list of faces defined by their vertex indices |
| `volume` | `float` | volume of the shape |
| `inertia_tensor` | `List[List[float]]` | inertia tensor of the shape (3x3 matrix) |

`Shapes`

`dataclass`
---------------------

Shapes data class to store the geometric data of multiple shapes.

Attributes:

| Name | Type | Description |
| --- | --- | --- |
| `shapes` | `Dict[int, Shape]` | mapping shape ID to its Shape data |

`polyhedron_mass_properties_mc(vertices, cell_faces, n_samples=20000, density=1.0, seed=0)`
-------------------------------------------------------------------------------------------

Compute the volume, center of mass and inertia tensor of a polyhedron defined by its vertices and faces, using a Monte Carlo method.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `vertices` | `array of shape (n_vertices, 3)` | coordinates of the vertices | *required* |
| `cell_faces` | `list` | each face is a list of vertex indices | *required* |
| `n_samples` | `int` | number of random points to generate for the estimation | `20000` |
| `density` | `float` | density of the material (to compute mass from volume) | `1.0` |
| `seed` | `float` | seed for the random number generator | `0` |
| `tol` | `float` | tolerance for considering a point as inside the polyhedron (default: 1e-12) | *required* |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `volume` | `float` | estimated volume of the polyhedron |
| `center` | `array of shape (3,)` | estimated center of mass of the polyhedron |
| `I` | `array of shape (3,)` | estimated inertia tensor of the polyhedron (as a 3-element array with the diagonal elements Ixx, Iyy, Izz) Note: the inertia tensor is returned as a 3-element array with the diagonal elements Ixx, Iyy, Izz, assuming the products of inertia are negligible for the shapes considered. This is a common approximation for convex polyhedra |

`polyhedron_mass_properties_mc_fast(vertices, cell_faces, n_samples=200000, density=1.0, seed=0, tol=1e-12)`
------------------------------------------------------------------------------------------------------------

Compute the volume, center of mass and inertia tensor of a polyhedron defined by its vertices and faces, using a Monte Carlo method.
Unlike the polyhedron\_mass\_properties\_mc function, this version is optimized for speed by using vectorized operations and returning only the diagonal elements of the inertia tensor.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `vertices` | `array of shape (n_vertices, 3)` | coordinates of the vertices | *required* |
| `cell_faces` | `list` | each face is a list of vertex indices | *required* |
| `n_samples` | `int` | number of random points to generate for the estimation | `200000` |
| `density` | `float` | density of the material (to compute mass from volume) | `1.0` |
| `seed` | `float` | seed for the random number generator | `0` |
| `tol` | `float` | tolerance for considering a point as inside the polyhedron (default: 1e-12) | `1e-12` |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `volume` | `float` | estimated volume of the polyhedron |
| `center` | `array of shape (3,)` | estimated center of mass of the polyhedron |
| `I` | `array of shape (3,)` | estimated inertia tensor of the polyhedron (as a 3-element array with the diagonal elements Ixx, Iyy, Izz) Note: the inertia tensor is returned as a 3-element array with the diagonal elements Ixx, Iyy, Izz, assuming the products of inertia are negligible for the shapes considered. This is a common approximation for convex polyhedra |

`check_interface(face_i, verts_i, face_j, verts_j)`
---------------------------------------------------

Check if two faces (face\_i and face\_j) from two different cells represent the same interface, by comparing their geometry (area, normal, number of vertices).

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `face_i` | `list` | vertex indices of the first face | *required* |
| `verts_i` | `array of shape (n_vertices_i, 3)` | coordinates of the vertices of the first cell | *required* |
| `face_j` | `list` | vertex indices of the second face | *required* |
| `verts_j` | `array of shape (n_vertices_j, 3)` | coordinates of the vertices of the second cell | *required* |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `ok` | `bool` | True if the faces represent the same interface, False otherwise |
| `reason` | `str` | reason for failure if ok is False ("nb\_vertices", "area", "normal") Criteria for matching: - number of vertices must be the same - area must be within 1% of each other - normals must be parallel (cross product close to zero) |

`check_interfaces(cell_shapes, polyhedra)`
------------------------------------------

Check that the interfaces defined by Neper (from the polyhedra definitions) are correctly represented in the computed cell shapes.
For each interface (face shared by two cells), we check if there are corresponding faces in the shapes of the two cells that match in terms of geometry (area, normal, number of vertices).

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cell_shapes` | `Shapes` | data class containing the geometric data of the cells (vertices, faces, edges, volume, inertia tensor) | *required* |
| `polyhedra` | `dict` | mapping polyhedron ID to its face indices [face\_id0, face\_id1, ...] (from Neper) | *required* |

Returns:

| Type | Description |
| --- | --- |
| `None` |  |

`count_interfaces(polyhedra)`
-----------------------------

Count the number of interfaces between cells in a topology.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `polyhedra` | `dict` | mapping polyhedron ID to its face indices [fid0, fid1, ...] | *required* |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `n_interface` | `int` | number of interfaces |

`get_interfaces(polyhedra)`
---------------------------

Return a list of interfaces between cells in a topology, defined as faces that are shared by exactly two cells.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `polyhedra` | `dict` | mapping polyhedron ID to its face indices [fid0, fid1, ...] | *required* |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `interfaces` | `list` | tuples (fid, cell1, cell2) where fid is the face ID of the interface, and cell1 and cell2 are the IDs of the two cells sharing the interface. |



Made with
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)