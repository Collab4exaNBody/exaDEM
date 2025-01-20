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

#include <exaDEM/shape.hpp>

namespace exaDEM
{
  // data collection of shapes
  struct shapes
  {
    template <typename T> using VectorT = onika::memory::CudaMMVector<T>;
    VectorT<shape> m_data;

    inline const shape *data() const { return onika::cuda::vector_data(m_data); }

    inline size_t get_size() { return onika::cuda::vector_size(m_data); }

    inline size_t get_size() const { return onika::cuda::vector_size(m_data); }

    ONIKA_HOST_DEVICE_FUNC
    inline const shape *operator[](const uint32_t idx) const
    {
      const shape *data = onika::cuda::vector_data(m_data);
      return data + idx;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline shape *operator[](const std::string name)
    {
      for (auto &shp : this->m_data)
      {
        if (shp.m_name == name)
        {
          return &shp;
        }
      }
      // std::cout << "Warning, the shape: " << name << " is not included in this collection of shapes. We return a nullptr." << std::endl;
      return nullptr;
    }

    inline void add_shape(shape *shp)
    {
      this->m_data.push_back(*shp); // copy
    }
  };
  
  struct shapes_GPU
  {
  	template<typename T> using VectorT = onika::memory::CudaMMVector<T>;
  	
  	VectorT<double> m_vertices_x;
  	VectorT<double> m_vertices_y;
  	VectorT<double> m_vertices_z;
  	VectorT<double> m_radius;
  	VectorT<int> m_edges;

  	
  	VectorT<int> start_m_vertices;
  	VectorT<int> end_m_vertices;
  	
  	VectorT<int> start_m_edges;
  	VectorT<int> end_m_edges;
  	

  	int b = true;
  	
    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_vertices(const uint32_t idx) { int *end = onika::cuda::vector_data(end_m_vertices); int *start = onika::cuda::vector_data(start_m_vertices); return end[idx] - start[idx]; }

    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_vertices(const uint32_t idx) const { const int *end = onika::cuda::vector_data(end_m_vertices); const int *start = onika::cuda::vector_data(start_m_vertices); return end[idx] - start[idx]; } 
      
    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_edges(const uint32_t idx) { const int* end = onika::cuda::vector_data(end_m_edges); const int* start = onika::cuda::vector_data(start_m_edges); return (end[idx] - start[idx]) / 2; }

    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_edges(const uint32_t idx) const { const int* end = onika::cuda::vector_data(end_m_edges); const int* start = onika::cuda::vector_data(start_m_edges); return (end[idx] - start[idx]) / 2; }
      

      
      inline void add_shape(shape *shp)
      {
         if(b)
         {
         	start_m_vertices.push_back(0);
         	start_m_edges.push_back(0);
         	b = false;
         }
         else
         {
      	 	start_m_vertices.push_back(end_m_vertices[end_m_vertices.size() - 1]);
      	 	start_m_edges.push_back(end_m_edges[end_m_edges.size() - 1]);
      	 }
      	 
      	int end = 0;
      	
      	for(auto &v : shp->m_vertices)
      	{
      		m_vertices_x.push_back(v.x);
      		m_vertices_y.push_back(v.y);
      		m_vertices_z.push_back(v.z);
      		end++;
      	}
      	
      	int end2 = 0;
      	
      	for(auto &e : shp->m_edges)
      	{
      		m_edges.push_back(e);
      		end2++;
      	}
    	      	
      	
      	end_m_vertices.push_back(start_m_vertices[start_m_vertices.size() - 1] + end);
      	
      	end_m_edges.push_back(start_m_edges[start_m_edges.size() - 1]  + end2);
      	
      	m_radius.push_back(shp->m_radius);
      	
      }
      
          ONIKA_HOST_DEVICE_FUNC
      inline exanb::Vec3d get_vertex(const uint32_t i, size_t j, const exanb::Vec3d &p, const exanb::Quaternion &orient) const 
      {
        auto *start = onika::cuda::vector_data(start_m_vertices);
        auto *vx = onika::cuda::vector_data(m_vertices_x);
        auto *vy = onika::cuda::vector_data(m_vertices_y);
        auto *vz = onika::cuda::vector_data(m_vertices_z);
        
        Vec3d v = {vx[start[i] + j], vy[start[i] + j], vz[start[i] + j]};
        
        return p + orient * v; 
      }
      
    ONIKA_HOST_DEVICE_FUNC
      inline std::pair<int, int> get_edge(const uint32_t type, const int i)
      {
        const int *__restrict__ edges = onika::cuda::vector_data(m_edges);
        const int *__restrict__ start = onika::cuda::vector_data(start_m_edges);
        const int *__restrict__ end = onika::cuda::vector_data(end_m_edges);
        return {edges[start[type] + 2 * i], edges[start[type] + 2 * i + 1]};
      }
      
    ONIKA_HOST_DEVICE_FUNC
      inline std::pair<int, int> get_edge(const uint32_t type, const int i) const
      {
        const int *__restrict__ edges = onika::cuda::vector_data(m_edges);
        const int *__restrict__ start = onika::cuda::vector_data(start_m_edges);
        const int *__restrict__ end = onika::cuda::vector_data(end_m_edges);
        return {edges[start[type] + 2 * i], edges[start[type] + 2 * i + 1]};
      }

      
      ONIKA_HOST_DEVICE_FUNC
      inline double radius( const uint32_t i )
      {
      	auto *radius = onika::cuda::vector_data(m_radius);
      	
      	return radius[i];
      }
 	
  	
  	
  };
  
} // namespace exaDEM
