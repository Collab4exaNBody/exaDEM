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

#include <exaDEM/shape_detection.hpp>

namespace exaDEM
{
  using namespace exanb;

  // rVerletMax = rVerlet + sphere radius
  std::vector<exaDEM::Interaction> detection_sphere_driver(
      const Stl_mesh& mesh,
      const size_t cell, 
      const size_t p, 
      const uint64_t id, 
      const size_t drv_id, 
      const double rx, 
      const double ry, 
      const double rz, 
      const double radius,
      const double rVerletMax)
  {
    std::vector<exaDEM::Interaction> res;
    exaDEM::Interaction item;
    item.cell_i = cell;
    item.p_i = p;
    item.id_i = id;
    item.id_j = drv_id;
    item.sub_i = 0; // not used
    const Vec3d* dvertices = onika::cuda::vector_data(mesh.vertices);

    auto &list = mesh.grid_indexes[cell];
    auto &shp  = mesh.shp;
    const size_t stl_nv = list.vertices.size();
    const size_t stl_ne = list.edges.size();
    const size_t stl_nf = list.faces.size();

    exanb::Vec3d v = {rx, ry, rz};
    // vertex - vertex
    item.type = 7;
    for (size_t j = 0; j < stl_nv; j++)
    {
      size_t idx = list.vertices[j];
      if(filter_vertex_vertex_v2(rVerletMax, v, radius, dvertices, idx, &shp))
      {
        item.sub_j = idx;
        res.push_back(item);
      }
    }
    // vertex - edge
    item.type = 8;
    for (size_t j = 0; j < stl_ne; j++)
    {
      size_t idx = list.edges[j];
      if(filter_vertex_edge(rVerletMax, v, radius, dvertices, idx, &shp))
      {
        item.sub_j = idx;
        res.push_back(item);
      }
    }
    // vertex - face
    item.type = 9;
    for (size_t j = 0; j < stl_nf; j++)
    {
      size_t idx = list.faces[j];
      if(filter_vertex_face(rVerletMax, v, radius, dvertices, idx, &shp))
      {
        item.sub_j = idx;
        res.push_back(item);
      }
    }
    return res;
  }
}

