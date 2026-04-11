#!/usr/bin/env python3

import sys
import numpy as np
import itertools


from lib.data_class import Particle, Params, Interactions, RockableData,CellsData, Shape, Shapes
from lib.io_utils import read_tess, write_shp_file, write_sticked_conf, write_rockable_file
from lib.geometry import intersect_planes, inside_all_planes, order_face_vertices, unique_points
from lib.topology import count_interfaces,check_interfaces
from lib.mass_properties import polyhedron_mass_properties_mc_fast
from lib.data_utils import make_sticked_conf

def get_local_face_normal(face_id, faces, vertices, face_normals, cell_center):
    '''
    Compute the local normal of a face, oriented towards the cell center.
    
    Parameters
    ----------
    face_id : int 
        global ID of the face
    faces : dict 
        mapping face ID to its vertex indices [v0, v1, ...]
    vertices : dict 
        mapping vertex ID to its coordinates [x, y, z]
    face_normals : dict 
        mapping face ID to its normal vector (ax, ay, az, d) for the plane equation ax + by + cz + d = 0
    cell_center : array 
        coordinates of the cell center [x, y, z]

    Returns
    ----------
    n : array 
        normal vector of the face, oriented towards the cell center
    '''
    
    n = np.array(face_normals[face_id][1:4])
    n /= np.linalg.norm(n)

    face = faces[face_id]
    face_center = np.mean([vertices[v] for v in face], axis=0)

    if np.dot(n, face_center - cell_center) < 0:
        n = -n

    return n

def minkowski_erosion_cell_from_planes(cell_center, cell_vertices_coords_local, cell_faces, local_normals, gap):
    '''
    Compute the Minkowski erosion of a cell defined by its faces and local normals, by intersecting the half-spaces defined by the planes corresponding to each face, shifted by the gap distance in the direction of the normal.
    
    Parameters
    ----------
    cell_center : array 
        coordinates of the cell center [x, y, z]
    cell_vertices_coords_local : list
        list of vertex coordinates of the cell in local indices 
    cell_faces : dict 
        mapping local face ID to its vertex indices [v0, v1, ...]
    local_normals : dict 
        mapping local face ID to its normal vector (ax, ay, az) oriented towards the cell center
    gap : float 
        distance to shift the planes for the Minkowski erosion (should be >= 2*radius of the particles)

    Returns
    ----------
    new_vertices : array 
        array of shape (n_vertices, 3) with the coordinates of the vertices of the eroded cell
    new_faces: list 
        list of faces defined by their vertex indices [v0, v1, ...] for the eroded cell
    '''
    planes = []
    for fid, face in cell_faces.items():
        n = local_normals[fid]
        p0 = cell_vertices_coords_local[face[0]] - cell_center
        d = np.dot(n, p0)
        planes.append((n, d - gap))

    vertices_list = []
    for p1, p2, p3 in itertools.combinations(planes, 3):
        p = intersect_planes(p1, p2, p3)
        if p is not None and inside_all_planes(cell_center, p, planes):
            vertices_list.append(p)

    new_vertices = unique_points(vertices_list)

    new_faces = []
    for (n, d) in planes:
        face_vids = [i for i, v in enumerate(new_vertices) if abs(np.dot(n, v) - d) < 1e-8]
        if len(face_vids) >= 3:
            new_faces.append(order_face_vertices(face_vids, new_vertices, n))

    return new_vertices, new_faces

def compute_cells(cells_data: CellsData, gap) -> Shapes:
    '''
    Compute the cell data (vertices, faces, volume, inertia tensor) for each cell in the topology, applying a Minkowski erosion to account for the particle radius.
    
    Parameters
    ----------
    cells_data : CellsData 
        data class containing the vertices, edges, faces, face normals, and polyhedra definitions of the system
    gap : float 
        gap to apply for the Minkowski erosion (should be >= 2
   
    Returns
    ----------
    Shapes 
        data class containing the geometric data of the eroded cells (vertices, faces, volume, inertia tensor)
    cells_out : CellsData 
        data class containing the original geometric data of the cells (vertices, edges, faces, face normals, polyhedra definitions) to be used for writing the shapefile and defining interactions
    '''

    vertices = cells_data.vertices
    edges = cells_data.edges
    faces = cells_data.faces
    face_normals = cells_data.face_normals
    polyhedra = cells_data.polyhedra
    radius = cells_data.radius

    cell_shapes = {}

    #loop over polyhedra
    for pid, face_ids in polyhedra.items():
        #RETRIEVE VERTICES OF THE CELL
        vert_set = set()
        #loop over faces of the cell
        for fid in face_ids:#global face ids (from Neper)
            if fid in faces:#check if the face id is in the faces dictinary
                vert_set.update([v for v in faces[fid] if v in vertices])#add the vertices of the face to the set
                # <=>
                #lst = []
                #for v in faces[fid]:
                #  if v in vertices:
                #    lst.append(v)
        vert_list = sorted(list(vert_set))#sort global id vertices of the cell
        
        #CONVERTION TO LOCAL INDICES
        # convert vertex ids to local indices (dictionnary)
        vid_map = {v: idx for idx, v in enumerate(vert_list)} # global vertex id -> local index
        #<=>
        #for i in range(len(vert_list)):
        #    v = vert_list[i]
        cell_vertices_coords_local = [vertices[v] for v in vert_list]  # local vertex coordinates of the cell
        
        # BARYCENTER OF THE CELL
        # DOUBLON cell_vertices = list({v for fid in face_ids for v in faces[fid]}) #list of global vertex ids of the cell
        #<=>
        #cellvertices = set()
        #for fid in face_ids:        # pour chaque face de la cellule
        #for v in faces[fid]:    # pour chaque sommet de cette face
        #cell_vertices.add(v)
                    # Compute cell center
        cell_center = np.mean(
                [np.array(vertices[v]) for v in vert_list], axis=0
            )

        #LOCAL FACE DEFINITION
        cell_faces = {} #dictionnary of local face definitions: cell_faces[local_face_id] = [local_vertex_id0, local_vertex_id1, ...]
        local_normals = {} #dictionnary of local face normals: local_normals[local_face_id] = normal_vector
        for fid in face_ids:
            if fid not in faces:
                continue
            cell_faces[fid] = [vid_map[v] for v in faces[fid]]
            local_normals[fid] = get_local_face_normal(
                fid, faces, vertices, face_normals, cell_center
            )# compute local normal of the face (oriented towards the cell center)
                
        # --- Minkowski
        eroded_vertices, eroded_faces = minkowski_erosion_cell_from_planes(
            cell_center,
            cell_vertices_coords_local,
            cell_faces,
            local_normals,
            gap=gap
        )

        if eroded_vertices.shape[0] < 4:
            continue

        cell_edges = set()
        for f in eroded_faces:
            for i in range(len(f)):
                v1 = f[i]
                v2 = f[(i + 1) % len(f)]
                edge = tuple(sorted((v1, v2)))
                cell_edges.add(edge)

        volume, center, I = polyhedron_mass_properties_mc_fast(
            eroded_vertices, eroded_faces, n_samples=5000
        )

        center = cell_center
        cell_shapes[pid] = Shape(
            vertices=eroded_vertices,
            faces=eroded_faces,
            edges=list(cell_edges),
            volume=volume,
            center=center,
            inertia_tensor=I
        )


    cells_out = CellsData(
        vertices={v: vertices[v] for v in vert_list},
        edges=edges,
        faces=faces,
        face_normals=face_normals,
        polyhedra=polyhedra,
        radius=radius
    )

    return cell_shapes, cells_out

#***************************
# MAIN
#***************************

def main():
    if len(sys.argv) != 4:
        print('Usage: python3 tess2rockable.py input.tess radius output.shp')
        sys.exit(1)
    '''
    Main function to convert a Neper tessellation file to a Rockable configuration file.
        
    Parameters
    ----------
    input.tess : str 
        file path to the Neper tessellation file
    radius : float
        radius of the particles to be generated (for stickVerticesInClusters)
    
    Returns
    ----------
    output.shp : str
        output file path for the generated shapefile (to be used in the Rockable configuration)
    output_sticked.conf : str
        output file path for the generated Rockable configuration file with sticked particles
    '''

    tess_file = sys.argv[1]
    radius = float(sys.argv[2])
    shp_file = sys.argv[3]

    gap= radius*2

    print(f"Reading {tess_file}...")
    Neper_cell_datas = read_tess(tess_file, radius=radius)
    n_interfaces = count_interfaces(Neper_cell_datas.polyhedra)
    print(f"Number of Neper interfaces: {n_interfaces}")

    print("Computing cells...")
    cell_shapes, cells_out = compute_cells(Neper_cell_datas, gap)
    write_shp_file(cell_shapes, shp_file, radius)
    check_interfaces(cell_shapes, Neper_cell_datas.polyhedra)
    cell_centers = {
      pid: shape.center
      for pid, shape in cell_shapes.items()
    }
    data = make_sticked_conf(cell_centers, shp_file, gap)
    write_rockable_file(shp_file.replace('.shp', '_sticked.conf'), data)  

if __name__ == '__main__':
    main()