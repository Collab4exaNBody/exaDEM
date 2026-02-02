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
std::vector<exaDEM::PlaceholderInteraction> detection_sphere_driver(const RShapeDriver& mesh, const size_t cell,
                                                                    const size_t p, const uint64_t id,
                                                                    const size_t drv_id, const double rx,
                                                                    const double ry, const double rz,
                                                                    const double radius, const double rVerletMax) {
  using onika::cuda::vector_data;
  std::vector<exaDEM::PlaceholderInteraction> res;
  exaDEM::PlaceholderInteraction item;
  auto& pi = item.i();       // particle i (id, cell, pos, sub)
  auto& pd = item.driver();  // driver (id, cell, pos, sub)
  pi.cell = cell;
  pi.p = p;
  pi.id = id;
  pd.id = drv_id;
  pi.sub = 0;  // not used

  // Get info from the rshape mesh.
  const Vec3d* dvertices = vector_data(mesh.vertices);
  auto& list = mesh.grid_indexes[cell];
  auto& shp = mesh.shp;
  const size_t rshape_driver_nv = list.vertices.size();
  const size_t rshape_driver_ne = list.edges.size();
  const size_t rshape_driver_nf = list.faces.size();

  exanb::Vec3d v = {rx, ry, rz};
  constexpr double dhomothety = 1.0;
  double dradius = shp.minskowski(dhomothety);
  // vertex - vertex
  item.pair.type = 7;
  for (size_t j = 0; j < rshape_driver_nv; j++) {
    size_t didx = list.vertices[j];
    if (filter_vertex_vertex_v2(rVerletMax, v, radius, dvertices, dradius, didx)) {
      pd.sub = didx;
      res.push_back(item);
    }
  }
  // vertex - edge
  item.pair.type = 8;
  for (size_t j = 0; j < rshape_driver_ne; j++) {
    size_t didx = list.edges[j];
    if (filter_vertex_edge(rVerletMax, v, radius, dvertices, dhomothety, didx, &shp)) {
      pd.sub = didx;
      res.push_back(item);
    }
  }
  // vertex - face
  item.pair.type = 9;
  for (size_t j = 0; j < rshape_driver_nf; j++) {
    size_t didx = list.faces[j];
    if (filter_vertex_face(rVerletMax, v, radius, dvertices, dhomothety, didx, &shp)) {
      pd.sub = didx;
      res.push_back(item);
    }
  }
  return res;
}
}  // namespace exaDEM
