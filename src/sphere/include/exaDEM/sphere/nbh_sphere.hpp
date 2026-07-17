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
#include <vector>
#include <exaDEM/shape_detection.hpp>

namespace exaDEM {
// rVerletMax = rVerlet + sphere radius
std::vector<exaDEM::PlaceholderInteraction> detection_sphere_driver(const RShapeDriver& driver, const size_t cell,
                                                                    const size_t p, const uint64_t id,
                                                                    const size_t drv_id, const double rx,
                                                                    const double ry, const double rz,
                                                                    const double radius, const double rVerletMax) {
  using onika::cuda::vector_data;
  std::vector<exaDEM::PlaceholderInteraction> res;
  exaDEM::PlaceholderInteraction item;
  auto& pi = item.i();       // particle i (id, cell, pos, sub)
  auto& pd = item.driver();  // driver (id, cell, pos, sub)
  pi.cell_ = cell;
  pi.p_ = p;
  pi.id_ = id;
  pd.id_ = drv_id;  // convention: id of the driver is the index of the driver in the drivers vector.
  pd.cell_ = 0;     // convention: driver is not stored in a cell.
  pi.sub_ = 0;      // convention: not used for spherical particles.

  // Get info from the rshape driver.
  const Vec3d* dvertices = vector_data(driver.vertices_);
  // grid of projected vertices, edges, and faces for the rshape driver.
  RShapeDriverCellAccessor grid_rshape_driver_accessor(cell, driver.grid_indexes_);
  auto& shp = driver.shp_;

  exanb::Vec3d v = {rx, ry, rz};
  constexpr double dhomothety = 1.0;
  double dradius = shp.minkowski(dhomothety);
  // vertex - vertex
  item.pair_.type_ = InteractionTypeId::VertexRshapeDriverVertex;
  for (size_t j = 0; j < grid_rshape_driver_accessor.rshape_nv_; j++) {
    // driver sub is the vertex index in the rshape vertices array.
    size_t didx = grid_rshape_driver_accessor.grid_id_vertices_[j];
    if (filter_vertex_vertex_v2(rVerletMax, v, radius, dvertices, dradius, didx)) {
      pd.sub_ = didx;
      res.push_back(item);
    }
  }
  // vertex - edge
  item.pair_.type_ = InteractionTypeId::VertexRshapeDriverEdge;
  for (size_t j = 0; j < grid_rshape_driver_accessor.rshape_ne_; j++) {
    // driver sub is the edge index in the rshape edges array.
    size_t didx = grid_rshape_driver_accessor.grid_id_edges_[j];
    if (filter_vertex_edge(rVerletMax, v, radius, dvertices, dhomothety, didx, &shp)) {
      pd.sub_ = didx;
      res.push_back(item);
    }
  }
  // vertex - face
  item.pair_.type_ = InteractionTypeId::VertexRshapeDriverFace;
  for (size_t j = 0; j < grid_rshape_driver_accessor.rshape_nf_; j++) {
    // driver sub is the face index in the rshape faces array.
    size_t didx = grid_rshape_driver_accessor.grid_id_faces_[j];
    if (filter_vertex_face(rVerletMax, v, radius, dvertices, dhomothety, didx, &shp)) {
      pd.sub_ = didx;
      res.push_back(item);
    }
  }
  return res;
}
}  // namespace exaDEM
