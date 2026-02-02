/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#pragma once

#include <exaDEM/drivers.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/shape_detection_driver.hpp>

namespace exaDEM {
/**
 * @brief Add interactions between particles and a driver defined by an RShape driver.
 *
 * This function detects contacts between all particles (vertices, edges, faces)
 * and the primitives (vertices, edges, faces) of a driver mesh. It relies on
 * Oriented Bounding Boxes (OBB) for broad-phase filtering and then applies
 * fine-grained filters for vertex-vertex, vertex-edge, vertex-face, edge-edge,
 * edge-vertex, and face-vertex interactions.
 *
 * @tparam Func   Functor type used to register contacts
 *                (signature: void(size_t pid, Interaction&, int sub_i, int sub_j)).
 *
 * @param mesh        RShape driver containing geometry and precomputed OBBs.
 * @param cell_a      Index of the mesh grid cell to process (must be < mesh.grid_indexes.size()).
 * @param add_contact Functor used to register detected contacts.
 * @param item        Reusable interaction object (fields are updated during processing).
 * @param n_particles Number of particles to process.
 * @param rVerlet     Verlet radius (distance threshold for contact detection).
 * @param type        Array mapping particle index -> type id.
 * @param id          Array of unique particle identifiers.
 * @param rx, ry, rz  Arrays of particle positions.
 * @param vertices    Vertex field storing per-particle vertex positions.
 * @param type        Array of particle homoethety.
 * @param orient      Array of particle orientations (as quaternions).
 * @param shps        Shape container indexed by particle type.
 *
 * @note The interaction type is encoded in `item.type`:
 *       - 7 : particle-vertex vs driver-vertex
 *       - 8 : particle-vertex vs driver-edge
 *       - 9 : particle-vertex vs driver-face
 *       - 10: particle-edge vs driver-edge
 *       - 11: particle-edge vs driver-vertex
 *       - 12: particle-face vs driver-vertex
 */
template <typename Func>
ONIKA_HOST_DEVICE_FUNC inline void add_driver_interaction(
    RShapeDriver& mesh, size_t cell_a, Func& add_contact, PlaceholderInteraction& item, const size_t n_particles,
    const double rVerlet, const ParticleTypeInt* __restrict__ type, const uint64_t* __restrict__ id,
    const double* __restrict__ rx, const double* __restrict__ ry, const double* __restrict__ rz, VertexField& vertices,
    const double* __restrict__ homothety, const exanb::Quaternion* __restrict__ orient, shapes& shps) {
  using onika::cuda::vector_data;
  constexpr double dhomothety = 1.0;
#define __particle__ vertices_i, hi, i, shpi
#define __driver__ mesh.vertices.data(), dhomothety, idx, &mesh.shp
  assert(cell_a < mesh.grid_indexes.size());
  auto& list = mesh.grid_indexes[cell_a];
  const size_t rshape_nv = list.vertices.size();
  const size_t rshape_ne = list.edges.size();
  const size_t rshape_nf = list.faces.size();

  if (rshape_nv == 0 && rshape_ne == 0 && rshape_nf == 0) {
    return;
  }

  // Get OBBs from rshape mesh
  auto& rshape_shp = mesh.shp;
  OBB* __restrict__ rshape_obb_vertices = onika::cuda::vector_data(rshape_shp.m_obb_vertices);
  [[maybe_unused]] OBB* __restrict__ rshape_obb_edges = vector_data(rshape_shp.m_obb_edges);
  [[maybe_unused]] OBB* __restrict__ rshape_obb_faces = vector_data(rshape_shp.m_obb_faces);

  // particle i (id, cell id, particle position, sub vertex)
  auto& pi = item.i();

  for (size_t p = 0; p < n_particles; p++) {
    Vec3d r = {rx[p], ry[p], rz[p]};  // position
    double hi = homothety[p];
    ParticleVertexView vertices_i = {p, vertices};
    const Quaternion& orient_i = orient[p];
    pi.p = p;
    pi.id = id[p];
    auto ti = type[p];
    const shape* shpi = shps[ti];
    const size_t nv = shpi->get_number_of_vertices();
    const size_t ne = shpi->get_number_of_edges();
    const size_t nf = shpi->get_number_of_faces();

    // Compute particle OBB
    OBB obb_i = shpi->obb;
    quat conv_orient_i = quat{vec3r{orient_i.x, orient_i.y, orient_i.z}, orient_i.w};
    obb_i.rotate(conv_orient_i);
    obb_i.translate(vec3r{r.x, r.y, r.z});
    obb_i.enlarge(rVerlet);

    // Note:
    // loop i = particle p
    // loop j = rshape mesh
    for (size_t i = 0; i < nv; i++) {
      vec3r v = conv_to_vec3r(vertices_i[i]);
      OBB obb_v_i;
      obb_v_i.center = v;
      obb_v_i.enlarge(rVerlet + shpi->minskowski(hi));

      // vertex - vertex
      item.pair.type = 7;
      pi.sub = i;
      for (size_t j = 0; j < rshape_nv; j++) {
        size_t idx = list.vertices[j];
        if (filter_vertex_vertex_v2(rVerlet, __particle__, __driver__)) {
          add_contact(item, i, idx);
        }
      }
      // vertex - edge
      item.pair.type = 8;
      for (size_t j = 0; j < rshape_ne; j++) {
        size_t idx = list.edges[j];
        if (filter_vertex_edge(rVerlet, __particle__, __driver__)) {
          add_contact(item, i, idx);
        }
      }
      // vertex - face
      item.pair.type = 9;
      for (size_t j = 0; j < rshape_nf; j++) {
        size_t idx = list.faces[j];
        const OBB& obb_f_rshape_j = rshape_obb_faces[idx];
        if (obb_f_rshape_j.intersect(obb_v_i)) {
          if (filter_vertex_face(rVerlet, __particle__, __driver__)) {
            add_contact(item, i, idx);
          }
        }
      }
    }

    for (size_t i = 0; i < ne; i++) {
      item.pair.type = 10;
      pi.sub = i;
      // edge - edge
      for (size_t j = 0; j < rshape_ne; j++) {
        const size_t idx = list.edges[j];
        if (filter_edge_edge(rVerlet, __particle__, __driver__)) {
          add_contact(item, i, idx);
        }
      }
    }

    for (size_t j = 0; j < rshape_nv; j++) {
      const size_t idx = list.vertices[j];
      // rejects vertices that are too far from the rshape mesh.
      const OBB& obb_v_rshape_j = rshape_obb_vertices[idx];
      if (!obb_v_rshape_j.intersect(obb_i)) {
        continue;
      }
      item.pair.type = 11;
      // edge - vertex
      for (size_t i = 0; i < ne; i++) {
        if (filter_vertex_edge(rVerlet, __driver__, __particle__)) {
          add_contact(item, i, idx);
        }
      }
      // face vertex
      item.pair.type = 12;
      for (size_t i = 0; i < nf; i++) {
        if (filter_vertex_face(rVerlet, __driver__, __particle__)) {
          add_contact(item, i, idx);
        }
      }
    }
  }  // end loop p
#undef __particle__
#undef __driver__
}  // end funcion

/**
 * @brief Add interactions between particles and a driver (boundary or external object).
 *
 * This function loops over all particles, fetches their associated shape,
 * and tests each vertex against the given driver to determine if a contact
 * should be created. If a contact is detected, the provided functor
 * `add_contact` is invoked with the corresponding interaction data.
 *
 * @tparam DriverT   Type of the driver (e.g., surface, ball, cylinder).
 * @tparam Func      Functor type used to register a contact (signature: void(size_t, Interaction&, int, int)).
 *
 * @param driver      Reference to the driver instance.
 * @param add_contact Functor used to register new contacts.
 * @param item        Reusable interaction object (its fields p_i and id_i are updated).
 * @param n_particles Total number of particles.
 * @param rVerlet     Verlet radius (tolerance distance for contact detection).
 * @param type        Pointer to array mapping particle index -> type id.
 * @param id          Pointer to array of unique particle ids.
 * @param vertices    Vertex field storing particle vertex positions.
 * @param shps        Shape container indexed by particle type.
 */
template <typename DriverT, typename Func>
ONIKA_HOST_DEVICE_FUNC inline void add_driver_interaction(DriverT& driver, Func& add_contact,
                                                          PlaceholderInteraction& item, const size_t n_particles,
                                                          const double rVerlet,
                                                          const ParticleTypeInt* __restrict__ type,
                                                          const uint64_t* __restrict__ id, VertexField& vertices,
                                                          const double* __restrict__ homothety, shapes& shps) {
  constexpr int DRIVER_VERTEX_SUB_IDX = -1;  // Convention
  auto& pi = item.i();                       // particle i (id, cell id, particle position, sub vertex)

  for (size_t pid = 0; pid < n_particles; pid++) {
    pi.p = pid;
    pi.id = id[pid];
    const double h = homothety[pid];
    ParticleVertexView vertex_view = {pid, vertices};

    const shape* shp = shps[type[pid]];
    assert(shp != nullptr);
    int num_vertices = shp->get_number_of_vertices();
    for (int vertex_index = 0; vertex_index < num_vertices; vertex_index++) {
      if (filter_vertex_driver(driver, rVerlet, vertex_view, h, vertex_index, shp)) {
        add_contact(item, vertex_index, DRIVER_VERTEX_SUB_IDX);
      }
    }
  }
}
}  // namespace exaDEM
