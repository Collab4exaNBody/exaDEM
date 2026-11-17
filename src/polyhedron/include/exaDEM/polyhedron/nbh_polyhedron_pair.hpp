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

#include <exaDEM/basic_types.hpp>
#include <exaDEM/interaction/placeholder_interaction.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/shape.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/basic_type.hpp>

namespace exaDEM {

/**
 * @brief Per-particle data needed to detect the interactions between two polyhedra.
 */
struct PolyhedronPairSide {
  uint64_t id_;                  ///< particle id
  uint32_t cell_;                ///< cell index
  uint16_t p_;                   ///< particle position in its cell
  Vec3d r_;                      ///< position (already transformed if the domain is deformed)
  Quaternion orient_;            ///< orientation
  double h_;                     ///< homothety
  const shape* shp_;             ///< shape
  ParticleVertexView vertices_;  ///< vertex positions (precomputed by compute_vertices)
};

/**
 * @brief Detects the potential contacts (VV, VE, VF, EE) between two polyhedra and registers them through
 * add_contact(item, sub_i, sub_j). a -> b interactions have swap_ = false, b -> a interactions (VE, VF) swap_ = true.
 * item.pair_.ghost_ is set by the caller.
 * @return false if the OBBs of the two polyhedra do not intersect.
 */
template <typename Func>
inline bool add_polyhedron_pair_interactions(PolyhedronPairSide& a, PolyhedronPairSide& b, const double rVerlet,
                                             PlaceholderInteraction& item, Func& add_contact) {
  const shape* const shp = a.shp_;
  const shape* const shp_nbh = b.shp_;
  const double hi = a.h_;
  const double h_nbh = b.h_;
  ParticleVertexView& vertices_a = a.vertices_;
  ParticleVertexView& vertices_b = b.vertices_;

  // Eliminate if two polyhedra are two far away if there is not intersection between their OBBs.
  OBB obb_i = compute_obb(shp->obb_, a.r_, a.orient_, hi);
  OBB obb_j = compute_obb(shp_nbh->obb_, b.r_, b.orient_, h_nbh);

  obb_i.enlarge(0.5 * rVerlet);
  obb_j.enlarge(0.5 * rVerlet);

  if (!obb_i.intersect(obb_j)) {
    return false;
  }

  obb_i.enlarge(0.5 * rVerlet);
  obb_j.enlarge(0.5 * rVerlet);

  // particle i (id, cell id, particle position, sub vertex)
  auto& pi = item.i();
  // particle j (id, cell id, particle position, sub vertex)
  auto& pj = item.j();

  pi.id_ = a.id_;
  pi.p_ = a.p_;
  pi.cell_ = a.cell_;

  pj.id_ = b.id_;
  pj.p_ = b.p_;
  pj.cell_ = b.cell_;

  const int nv = shp->get_number_of_vertices();
  const int ne = shp->get_number_of_edges();
  const int nf = shp->get_number_of_faces();
  const int nv_nbh = shp_nbh->get_number_of_vertices();
  const int ne_nbh = shp_nbh->get_number_of_edges();
  const int nf_nbh = shp_nbh->get_number_of_faces();

#define PARAMETERS_SWAP_FALSE rVerlet, vertices_a, hi, i, shp, vertices_b, h_nbh, j, shp_nbh
#define PARAMETERS_SWAP_TRUE rVerlet, vertices_b, h_nbh, j, shp_nbh, vertices_a, hi, i, shp

  // exclude possibilities with obb
  item.pair_.swap_ = false;
  for (int i = 0; i < nv; i++) {
    auto vi = vertices_a[i];
    OBB obbvi;
    obbvi.center = {vi.x, vi.y, vi.z};
    obbvi.enlarge(shp->minkowski(hi));
    if (obb_j.intersect(obbvi)) {
      item.pair_.type_ = InteractionTypeId::VertexVertex;
      for (int j = 0; j < nv_nbh; j++) {
        if (filter_vertex_vertex(PARAMETERS_SWAP_FALSE)) {
          add_contact(item, i, j);
        }
      }
      item.pair_.type_ = InteractionTypeId::VertexEdge;
      for (int j = 0; j < ne_nbh; j++) {
        if (filter_vertex_edge(PARAMETERS_SWAP_FALSE)) {
          add_contact(item, i, j);
        }
      }

      item.pair_.type_ = InteractionTypeId::VertexFace;
      for (int j = 0; j < nf_nbh; j++) {
        if (filter_vertex_face(PARAMETERS_SWAP_FALSE)) {
          add_contact(item, i, j);
        }
      }
    }
  }

  item.pair_.type_ = InteractionTypeId::EdgeEdge;
  for (int i = 0; i < ne; i++) {
    for (int j = 0; j < ne_nbh; j++) {
      if (filter_edge_edge(PARAMETERS_SWAP_FALSE)) {
        add_contact(item, i, j);
      }
    }
  }

  // interaction of from particle j to particle i
  item.pair_.swap_ = true;
  pj.cell_ = a.cell_;
  pj.id_ = a.id_;
  pj.p_ = a.p_;

  pi.cell_ = b.cell_;
  pi.id_ = b.id_;
  pi.p_ = b.p_;

  for (int j = 0; j < nv_nbh; j++) {
    auto vj = vertices_b[j];
    OBB obbvj;
    obbvj.center = {vj.x, vj.y, vj.z};
    obbvj.enlarge(shp_nbh->minkowski(h_nbh));

    if (obb_i.intersect(obbvj)) {
      item.pair_.type_ = InteractionTypeId::VertexEdge;
      for (int i = 0; i < ne; i++) {
        if (filter_vertex_edge(PARAMETERS_SWAP_TRUE)) {
          add_contact(item, j, i);
        }
      }

      item.pair_.type_ = InteractionTypeId::VertexFace;
      for (int i = 0; i < nf; i++) {
        if (filter_vertex_face(PARAMETERS_SWAP_TRUE)) {
          add_contact(item, j, i);
        }
      }
    }
  }
#undef PARAMETERS_SWAP_FALSE
#undef PARAMETERS_SWAP_TRUE
  return true;
}
}  // namespace exaDEM
