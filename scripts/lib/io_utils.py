from fileinput import filename
from nr.date import parse_time
from collections import defaultdict
import numpy as np

from lib.data_class import (
    Contact,
    Particle,
    Params,
    InteractionsParameters,
    Interactions,
    RockableData,
    CellsData,
    Shape,
    Shapes,
)

# ---------------------------------------------------------------------------
# READ ROCKABLE CONF FILE (generic)
# ---------------------------------------------------------------------------
def read_rockable_file(filepath: str) -> RockableData:
    """
    read a Rockable configuration file and return a structured data object.

    Parameters
    ----------
    filepath : str
        path to the Rockable configuration file

    Returns
    ----------
    RockableData
        structured data object containing all the information from the configuration file
    """
    param = {}
    interactions = {}
    particles = []
    n_particles = 0
    stick_distance = None

    with open(filepath, "r") as f:
        lines = f.readlines()

    reading_particles = False
    particle_count = 0

    for line in lines:
        line = line.strip()

        # ignorer commentaires et lignes vides
        if not line or line.startswith("#"):
            continue

        tokens = line.split()

        # --- PARTICLES ---
        if reading_particles:
            if line.startswith("stickVerticesInClusters"):
                stick_distance = float(tokens[1])
                reading_particles = False
                continue

            # parse particule (23 colonnes)
            p = Particle(
                name=tokens[0],
                group=int(tokens[1]),
                cluster=int(tokens[2]),
                homothety=float(tokens[3]),
                pos=tuple(map(float, tokens[4:7])),
                vel=tuple(map(float, tokens[7:10])),
                acc=tuple(map(float, tokens[10:13])),
                quat=tuple(map(float, tokens[13:17])),
                vrot=tuple(map(float, tokens[17:20])),
                arot=tuple(map(float, tokens[20:23])),
            )
            particles.append(p)
            continue

        # --- HEADER ---
        key = tokens[0]

        if key == "Particles":
            n_particles = int(tokens[1])
            reading_particles = True

        elif key == "gravity":
            param[key] = list(map(float, tokens[1:]))

        elif key in [
            "t",
            "tmax",
            "dt",
            "interVerlet",
            "interConf",
            "DVerlet",
            "dVerlet",
        ]:
            param[key] = float(tokens[1])

        elif key == "forceLaw":
            param[key] = tokens[1]

        elif key == "periodicity":
            param[key] = list(map(int, tokens[1:]))

        elif key == "shapeFile":
            param[key] = tokens[1]

        # --- interactions type (avec groupes) ---

        elif key in [
            "density",
            "knContact",
            "en2Contact",
            "ktContact",
            "muContact",
            "knInnerBond",
            "ktInnerBond",
            "en2InnerBond",
            "powInnerBond",
            "GInnerBond",
        ]:

            if key not in interactions:
                interactions[key] = {}

            if len(tokens) == 3:
                # format: key g value
                g = int(tokens[1])
                val = float(tokens[2])
                interactions[key][g] = val

            elif len(tokens) == 4:
                # format: key g1 g2 value
                g1 = int(tokens[1])
                g2 = int(tokens[2])
                val = float(tokens[3])
                interactions[key][(g1, g2)] = val

            else:
                raise ValueError(f"Format inconnu pour {key}: {tokens}")

    return RockableData(
        params=Params(param),
        interactions=Interactions(
            parameters=InteractionsParameters(interactions),
            contacts=[],
        ),
        particles=particles,
        n_particles=n_particles,
        stick_distance=stick_distance,
    )


# ---------------------------------------------------------------------------
# WRITE ROCKABLE CONF FILE (generic)
# ---------------------------------------------------------------------------


def write_rockable_file(filepath: str, data: RockableData):
    """
    Write a Rockable configuration file from a structured data object.

    Parameters
    ----------
    filepath : str
        path to the output Rockable configuration file
    data : RockableData
        data object containing all the information to be written in the configuration file, structured as follows:
        data = {"params": {...},"interactions": {...},"particles": [...],"n_particles": int,"stick_distance": float or None}

     Returns
    ----------
    None
        The function writes the configuration file in the format expected by Rockable, with sections for parameters, interactions and particles, and includes the stickVerticesInClusters line if stick_distance is provided.
    """
    with open(filepath, "w") as f:

        # --- PARAMS ---
        for key, val in data.params.values.items():
            if isinstance(val, list):
                f.write(f"{key} " + " ".join(map(str, val)) + "\n")
            else:
                f.write(f"{key} {val}\n")

        # --- INTERACTIONS ---
        for key, table in data.interactions.parameters.tables.items():
            for k, val in table.items():
                if isinstance(k, tuple):
                    g1, g2 = k
                    f.write(f"{key} {g1} {g2} {val}\n")
                else:
                    f.write(f"{key} {k} {val}\n")

        # --- PARTICLES ---
        f.write(f"Particles {data.n_particles}\n")

        for p in data.particles:
            line = [
                p.name,
                p.group,
                p.cluster,
                p.homothety,
                *p.pos,
                *p.vel,
                *p.acc,
                *p.quat,
                *p.vrot,
                *p.arot,
            ]
            f.write(" ".join(map(str, line)) + "\n")

        # --- FIN ---
        if data.stick_distance is not None:
            f.write(f"stickVerticesInClusters {data.stick_distance}\n")


# ---------------------------------------------------------------------------
# READ INTERACTIONS FILE (generic)
# ---------------------------------------------------------------------------
def read_interactions(filename)-> list[Contact]:
    '''
    Read an interactions file and return a list of Contact objects representing the interactions between particles.
    The function parses the interactions file, extracting the relevant data for each contact, and organizes it
    into a list of Contact objects for use in simulations.
    
    Parameters
    ----------
    filename : str
        path to the interactions file, with the following format:
        i j type fx fy fz pos_i_x pos_i_y pos_i_z pos_j_x pos_j_y pos_j_z

    Returns
    ----------
    list[Contact]
        a list of Contact objects, each containing the properties of a contact between two particles, including the indices of the particles involved, the type of contact, the force vector, and the positions of the contact points on each particle.
    '''
    contacts = []

    with open(filename, 'r') as f:
        for line in f:
            data = line.strip().split(",")

            i = int(data[0])
            j = int(data[1])
            itype = int(data[4])

            normal = np.array(list(map(float, data[9:12])))
            tangential = np.array(list(map(float, data[12:15])))

            contacts.append(
                Contact(
                    i=i,
                    j=j,
                    type=itype,
                    force=normal + tangential,
                    pos_i=np.array(list(map(float, data[15:18]))),
                    pos_j=np.array(list(map(float, data[18:21]))),

                )
            )

    return contacts

# ---------------------------------------------------------------------------
# READ XYZ FILE (generic)
# ---------------------------------------------------------------------------
def read_xyzdem_snapshot(particle_file, interaction_file=None) -> RockableData:
    '''
    Read a DEM snapshot from a text file and return a structured RockableData object.
    The function parses the particle data from the given file, extracting the particle properties and optionally the
    interaction parameters from a separate file, and organizes the data into a RockableData object for use in simulations.
    
    Parameters
    ----------
    particle_file : str
        path to the text file containing the particle data, with the following format:
        n_particles
        header line (ignored)
        name group cluster homothety pos_x pos_y pos_z vel_x vel_y vel_z acc_x acc_y acc_z quat_w quat_x quat_y quat_z vrot_x vrot_y vrot_z arot_x arot_y arot_z pid
    interaction_file : str, optional
        path to the text file containing the interaction parameters, with a format that can be parsed by the read_interactions function (default: None, meaning no interactions will be read)
        
    Returns
    ----------
    RockableData
        a RockableData object containing the parameters, interactions and particles defined in the input files, structured as follows:
        RockableData(params=Params({...}), interactions=Interactions({...}), particles=[Particle(...)], n_particles=int, stick_distance=None, time=float or None)
    '''
    particles = []
    clusters = {}

    with open(particle_file, "r") as f:
        n = int(f.readline())
        header = f.readline()
        time = parse_time(header)

        for line in f:
            data = line.split()
            pid = int(data[-1])

            p = Particle(
                name=f"p{pid}",
                group=0,
                cluster=int(data[7]),
                homothety=1.0,
                pos=tuple(map(float, data[1:4])),
                vel=tuple(map(float, data[4:7])),
                acc=(0, 0, 0),
                quat=(1, 0, 0, 0),
                vrot=(0, 0, 0),
                arot=(0, 0, 0),
            )

            particles.append(p)
            clusters.setdefault(p.cluster, []).append(p)

    contacts = {}
    if interaction_file:
        contacts = read_interactions(interaction_file)

    return RockableData(
        params=Params({}),  # vide ici
        interactions=Interactions(
            parameters=InteractionsParameters({}),
            contacts=contacts),
        particles=particles,
        n_particles=len(particles),
        stick_distance=None,
        time=time,
    )

# ---------------------------------------------------------------------------
# READ TESS FILE (FROM NEPER)
# ---------------------------------------------------------------------------
def read_tess(filename, radius=0.0):
    """
    Read a Neper .tess file and return structured data for vertices, edges, faces, face normals, and polyhedra.
    The function parses the .tess file, extracting the relevant sections for vertices, edges, faces, and polyhedra, and organizes the data into dictionaries for easy access.

    Parameters
    ----------
    filename : str
        path to the Neper .tess file to be read
    radius : float
        minkowski radius of the particles

    Returns
    ----------
    vertices : dict
        mapping vertex ID to its coordinates [x, y, z]
    edges : dict
        mapping edge ID to its vertex indices (v0, v1)
    faces : dict
        mapping face ID to its vertex indices [v0, v1, ...]
    face_normals : dict
        mapping face ID to its plane coefficients (a, b, c, d) for the plane equation ax + by + cz + d = 0
    polyhedra : dict
        mapping polyhedron ID to its face indices [face_id0, face_id1, ...]
    """
    vertices = {}
    edges = {}
    faces = {}
    face_normals = {}
    polyhedra = {}

    section = None
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # Sections detection
            if line.startswith("**"):
                section = line.lower()
                continue

            # Don't read beyond domain section
            if section == "**domain":
                break

            # Read vertices coordinates
            if section == "**vertex":
                parts = line.split()
                if len(parts) < 5:
                    continue  # ligne vide ou malformée
                try:
                    vid = int(parts[0]) - 1  # 0-based
                    x, y, z = map(float, parts[1:4])
                    vertices[vid] = [x, y, z]
                except ValueError:
                    continue

            # Read edges
            elif section == "**edge":
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    eid = int(parts[0]) - 1
                    v0, v1 = int(parts[1]) - 1, int(parts[2]) - 1
                    edges[eid] = (v0, v1)
                except ValueError:
                    continue

            # Read faces
            elif section == "**face":
                # on lit la ligne principale
                parts = line.split()
                if len(parts) < 3:
                    continue

                fid = int(parts[0]) - 1
                nverts = int(parts[1])
                vids = [int(v) - 1 for v in parts[2 : 2 + nverts]]
                faces[fid] = vids

                next(f)  # edges

                # --- ligne plane : ax by cz d
                plane = next(f).split()
                a, b, c, d = map(float, plane[:4])

                # normalisation
                norm = (a * a + b * b + c * c) ** 0.5
                face_normals[fid] = (a, b, c, d)

                next(f)  # domain info

            # Read polyhedra
            elif section == "**polyhedron":
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    pid = int(parts[0]) - 1
                    nfaces = int(parts[1])
                    face_ids = [abs(int(f)) - 1 for f in parts[2 : 2 + nfaces]]
                    polyhedra[pid] = face_ids
                except ValueError:
                    continue

    print(
        f"Read {len(vertices)} vertices, {len(edges)} edges, {len(faces)} faces, {len(polyhedra)} polyhedra from '{filename}'"
    )
    return CellsData(
        vertices=vertices,
        edges=edges,
        faces=faces,
        face_normals=face_normals,
        polyhedra=polyhedra,
        radius=radius,
    )


# ---------------------------------------------------------------------------
# WRITE SHP FILE (FOR ROCKABLE)
# ---------------------------------------------------------------------------
def write_shp_file(shapes: Shapes, outfile: str, radius: float):
    """
    Write a shapefile compatible with Rockable, containing the geometric data of the cells.

    Parameters
    ----------
    shapes : Shapes
        data class containing the geometric data of the cells (vertices, faces, edges, volume, inertia tensor)
    outfile : str
        path to the output shapefile
    radius : float
        Minkowski radius

    Returns
    ----------
    None
        The function writes a shapefile in the format expected by Rockable, with sections for each cell containing its vertices, edges, faces, volume and inertia tensor. The Minkowski radius is included as a parameter for each cell.
    """

    with open(outfile, "w") as fout:

        for pid, shape in shapes.items():
            vertices = shape.vertices
            edges = shape.edges
            faces = shape.faces
            volume = shape.volume
            I = shape.inertia_tensor

            fout.write("<\n")
            fout.write(f"name Voronoi{pid}\n")
            fout.write(f"radius {radius}\n")
            fout.write("preCompDone y\n")

            # --- vertices
            fout.write(f"nv {len(vertices)}\n")
            for v in vertices:
                fout.write(f"{v[0]} {v[1]} {v[2]}\n")

            # --- edges
            fout.write(f"ne {len(edges)}\n")
            for e in edges:
                fout.write(f"{e[0]} {e[1]}\n")

            # --- faces
            fout.write(f"nf {len(faces)}\n")
            for f in faces:
                fout.write(f'{len(f)} {" ".join(map(str, f))}\n')

            fout.write("obb.extent 0.5 0.5 0.5\n")
            fout.write("obb.e1 1.0 0.0 0.0\n")
            fout.write("obb.e2 0.0 1.0 0.0\n")
            fout.write("obb.e3 0.0 0.0 1.0\n")
            fout.write("obb.center 0. 0. 0.\n")
            fout.write("position 0. 0. 0.\n")
            fout.write("orientation 1.0 0.0 0.0 0.0\n")

            fout.write(f"volume {volume}\n")
            fout.write(f"I/m {I[0]} {I[1]} {I[2]}\n")
            fout.write(">\n")


# ---------------------------------------------------------------------------
# WRITE ROCKABLE CONF FILE FOR STICKED PLANES
# ---------------------------------------------------------------------------
def write_sticked_conf(filename, shapefile, cell_centers, gap):
    """
    write a Rockable configuration file for a simulation with sticked particles, based on the cell centers and a shapefile containing the cell geometries.
    """
    with open(filename, "w") as f:

        f.write(f" Rockable 29-11-2018\n")
        f.write(f" #temps initial\n")
        f.write(f" t 0  \n")
        f.write(f" #temps max\n")
        f.write(f" tmax 2\n")
        f.write(f" #pas de temps\n")
        f.write(f" dt 1.e-06\n")
        f.write(f" #pas de temps pour mettre à jour la liste de voisin\n")
        f.write(f" interVerlet 1.e-4\n")
        f.write(f" #pas de temps pour enregistrer\n")
        f.write(f" interConf 0.01\n")
        f.write(f" #Distance de detection de voisin pour les boites englobantes\n")
        f.write(f" DVerlet 0.02\n")
        f.write(
            f" #Distance de detection de voisin (entre les sous éléments des particules)\n"
        )
        f.write(f" dVerlet 0.01\n")
        f.write(f" #densité des particules : les 0 et 1 c'est les groupes\n")
        f.write(f" density 0 0.006\n")
        f.write(f" density 1 0.006\n")
        f.write(f" #gravité\n")
        f.write(f" gravity 0 0 -9.81\n")
        f.write(f" #type de loi de force utilisée\n")
        f.write(f" forceLaw StickedLinks\n")
        f.write(f" #les rigidités normales de contact\n")
        f.write(f" knContact 0 0 1e+04\n")
        f.write(f" knContact 0 1 1e+04\n")
        f.write(f" #les coefs de ristitutions au carré\n")
        f.write(f" en2Contact 0 0 0.001\n")
        f.write(f" en2Contact 0 1 0.001\n")
        f.write(f" #les rigidités tangentielles de contact\n")
        f.write(f" ktContact 0 0 8e+3\n")
        f.write(f" ktContact 0 1 8e+3\n")
        f.write(f" #les coefs de frottement\n")
        f.write(f" muContact 0 0 0.3\n")
        f.write(f" muContact 0 1 0.3\n")
        f.write(f" periodicity 0 1 0\n")
        f.write(f" \n")
        f.write(f" #des ressorts pour ragrder le lien cohesif\n")
        f.write(f" knInnerBond 0 0 1e+04\n")
        f.write(f" ktInnerBond 0 0 8e+03\n")
        f.write(f" en2InnerBond 0 0 0.0001\n")
        f.write(f" powInnerBond 0 0 2.0\n")
        f.write(f" \n")
        f.write(f" #les energies de rupture par unité de surface\n")
        f.write(f" GInnerBond 0 0 2.e-4\n")
        f.write(f" shapeFile {shapefile}\n")

        f.write(f"Particles {len(cell_centers)}\n")

        for pid, center in cell_centers.items():
            # print("Writing particle", pid, "center", center)
            x, y, z = center
            f.write(
                f"Voronoi{pid} 1 0 1 "
                f"{x:.15e} {y:.15e} {z:.15e} "
                f"0 0 0  "
                f"0 0 0  "
                f"1 0 0 0  "
                f"0 0 0  "
                f"0 0 0\n"
            )

        f.write(f"stickVerticesInClusters {4*gap}\n")
