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

#include<exaDEM/shape/shape.hpp>

namespace exaDEM
{
	// data collection of shapes
	struct shapesSOA
	{
		template <typename T> using VectorT =  onika::memory::CudaMMVector<T>; 
		
		int num_shapes = 0; //Number of shapes
		
		//VectorT <exanb::Vec3d> m_vertices; ///<
		VectorT<int> m_vertices_start;
		VectorT<double> m_vertices_x;
		VectorT<double> m_vertices_y;
		VectorT<double> m_vertices_z;
		VectorT<int> m_vertices_end;
		  
		//exanb::Vec3d m_inertia_on_mass;
		
		VectorT<double> m_inertia_on_mass_x;
		VectorT<double> m_inertia_on_mass_y;
		VectorT<double> m_inertia_on_mass_z;
		
		VectorT<double> obb_center_x;
		VectorT<double> obb_center_y;
		VectorT<double> obb_center_z;
		VectorT<double> obb_e1_x;
		VectorT<double> obb_e1_y;
		VectorT<double> obb_e1_z;
		VectorT<double> obb_e2_x;
		VectorT<double> obb_e2_y;
		VectorT<double> obb_e2_z;
		VectorT<double> obb_e3_x;
		VectorT<double> obb_e3_y;
		VectorT<double> obb_e3_z;
		VectorT<double> obb_extent_x;
		VectorT<double> obb_extent_y;
		VectorT<double> obb_extent_z;
		
		VectorT<double> m_radius;
		VectorT<double> m_volume;
		
		VectorT<std::string> m_name;


		//OBB		
		//vec3r center;  //< Center
		//vec3r e1;    //< 3 directions (normalized vectors)
		//vec3r e2;    //< 3 directions (normalized vectors)
		//vec3r e3;    //< 3 directions (normalized vectors)
		//vec3r extent;  //< 3 extents (in the the 3 directions)
		
		//VectorT <OBB> m_obb_vertices; ///< only used for stl meshes
		VectorT<int> m_obb_vertices_start;
		//center
		VectorT<double> m_obb_vertices_center_x;
		VectorT<double> m_obb_vertices_center_y;
		VectorT<double> m_obb_vertices_center_z;
		//e1
		VectorT<double> m_obb_vertices_e1_x;
		VectorT<double> m_obb_vertices_e1_y;
		VectorT<double> m_obb_vertices_e1_z;
		//e2
		VectorT<double> m_obb_vertices_e2_x;
		VectorT<double> m_obb_vertices_e2_y;
		VectorT<double> m_obb_vertices_e2_z;
		//e3
		VectorT<double> m_obb_vertices_e3_x;
		VectorT<double> m_obb_vertices_e3_y;
		VectorT<double> m_obb_vertices_e3_z;
		//extent
		VectorT<double> m_obb_vertices_extent_x;
		VectorT<double> m_obb_vertices_extent_y;
		VectorT<double> m_obb_vertices_extent_z;
		//end
		VectorT<int> m_obb_vertices_end;
		
		
		//VectorT <OBB> m_obb_edges;
		VectorT<int> m_obb_edges_start;
		//center
		VectorT<double> m_obb_edges_center_x;
		VectorT<double> m_obb_edges_center_y;
		VectorT<double> m_obb_edges_center_z;
		//e1
		VectorT<double> m_obb_edges_e1_x;
		VectorT<double> m_obb_edges_e1_y;
		VectorT<double> m_obb_edges_e1_z;
		//e2
		VectorT<double> m_obb_edges_e2_x;
		VectorT<double> m_obb_edges_e2_y;
		VectorT<double> m_obb_edges_e2_z;
		//e3
		VectorT<double> m_obb_edges_e3_x;
		VectorT<double> m_obb_edges_e3_y;
		VectorT<double> m_obb_edges_e3_z;
		//extent
		VectorT<double> m_obb_edges_extent_x;
		VectorT<double> m_obb_edges_extent_y;
		VectorT<double> m_obb_edges_extent_z;
		//end
		VectorT<int> m_obb_edges_end;
		
		
		//VectorT <OBB> m_obb_faces;
		VectorT<int> m_obb_faces_start;
		//center
		VectorT<double> m_obb_faces_center_x;
		VectorT<double> m_obb_faces_center_y;
		VectorT<double> m_obb_faces_center_z;
		//e1
		VectorT<double> m_obb_faces_e1_x;
		VectorT<double> m_obb_faces_e1_y;
		VectorT<double> m_obb_faces_e1_z;
		//e2
		VectorT<double> m_obb_faces_e2_x;
		VectorT<double> m_obb_faces_e2_y;
		VectorT<double> m_obb_faces_e2_z;
		//e3
		VectorT<double> m_obb_faces_e3_x;
		VectorT<double> m_obb_faces_e3_y;
		VectorT<double> m_obb_faces_e3_z;
		//extent
		VectorT<double> m_obb_faces_extent_x;
		VectorT<double> m_obb_faces_extent_y;
		VectorT<double> m_obb_faces_extent_z;
		//end
		VectorT<int> m_obb_faces_end;
		
		
		//OBB obb;
		
		//VectorT <int> m_edges; ///<
		VectorT<int> m_edges_start;
		VectorT<int> m_edges;
		VectorT<int> m_edges_end;
		  
		//VectorT <int> m_faces; ///<
		VectorT<int> m_faces_start;
		VectorT<int> m_faces;
		VectorT<int> m_faces_end;
		  
		//double m_radius; ///< use for detection
		//double m_volume; ///< use for detection
		//std::string m_name = "undefined";


    /*		inline const shape* data() const
    {
			return onika::cuda::vector_data( m_data );
    }*/

		inline size_t get_size()
		{
			return num_shapes;
		}

		inline size_t get_size() const
		{
			return num_shapes;
		}

		ONIKA_HOST_DEVICE_FUNC
			inline const shape operator[] (const uint32_t idx) const
			{
				shape res;
				
				VectorT<exanb::Vec3d> m_vertices_res;
				exanb::Vec3d m_inertia_on_mass_res;
				VectorT<OBB> m_obb_vertices_res;
				VectorT<OBB> m_obb_edges_res;
				VectorT<OBB> m_obb_faces_res;
				OBB obb_res;
				VectorT<int> m_edges_res;
				VectorT<int> m_faces_res;
				double m_radius_res;
				double m_volume_res;
				std::string m_name_res;
				
				m_inertia_on_mass_res = {m_inertia_on_mass_x[idx - 1], m_inertia_on_mass_y[idx - 1], m_inertia_on_mass_z[idx - 1]};
				obb_res.center = {obb_center_x[idx - 1], obb_center_y[idx - 1], obb_center_z[idx - 1]};
				obb_res.e1 = {obb_e1_x[idx - 1], obb_e1_y[idx - 1], obb_e1_z[idx - 1]};
				obb_res.e2 = {obb_e2_x[idx - 1], obb_e2_y[idx - 1], obb_e2_z[idx - 1]};
				obb_res.e3 = {obb_e3_x[idx - 1], obb_e3_y[idx - 1], obb_e3_z[idx - 1]};
				obb_res.extent = {obb_extent_x[idx - 1], obb_extent_y[idx - 1], obb_extent_z[idx - 1]};
				m_radius_res = m_radius[idx - 1];
				m_volume_res = m_volume[idx - 1];
				m_name_res = m_name[idx - 1];
				
				int start = m_vertices_start[idx - 1];
				int end = m_vertices_end[idx - 1];
				for(int i = start; i< end; i++)
				{
					m_vertices_res.push_back({m_vertices_x[i], m_vertices_y[i], m_vertices_z[i]});
				}
				
				start = m_edges_start[idx - 1];
				end = m_edges_end[idx - 1];
				for(int i = start; i< end; i++)
				{
					m_edges_res.push_back(m_edges[i]);
				}
				
				start = m_faces_start[idx - 1];
				end = m_faces_end[idx - 1];
				for(int i = start; i< end; i++)
				{
					m_faces_res.push_back(m_faces[i]);
				}
				
				start = m_obb_vertices_start[idx - 1];
				end = m_obb_vertices_end[idx - 1];
				for(int i = start; i < end; i++)
				{
					OBB obb;
					obb.center = {m_obb_vertices_center_x[idx - 1], m_obb_vertices_center_y[idx - 1], m_obb_vertices_center_z[idx - 1]};
					obb.e1 = {m_obb_vertices_e1_x[idx - 1], m_obb_vertices_e1_y[idx - 1], m_obb_vertices_e1_z[idx - 1]};
					obb.e2 = {m_obb_vertices_e2_x[idx - 1], m_obb_vertices_e2_y[idx - 1], m_obb_vertices_e2_z[idx - 1]};
					obb.e3 = {m_obb_vertices_e3_x[idx - 1], m_obb_vertices_e3_y[idx - 1], m_obb_vertices_e3_z[idx - 1]};
					obb.extent = {m_obb_vertices_extent_x[idx - 1], m_obb_vertices_extent_y[idx - 1], m_obb_vertices_extent_z[idx - 1]};
					m_obb_vertices_res.push_back(obb);					
				}
				
				start = m_obb_edges_start[idx - 1];
				end = m_obb_edges_end[idx - 1];
				for(int i = start; i < end; i++)
				{
					OBB obb;
					obb.center = {m_obb_edges_center_x[idx - 1], m_obb_edges_center_y[idx - 1], m_obb_edges_center_z[idx - 1]};
					obb.e1 = {m_obb_edges_e1_x[idx - 1], m_obb_edges_e1_y[idx - 1], m_obb_edges_e1_z[idx - 1]};
					obb.e2 = {m_obb_edges_e2_x[idx - 1], m_obb_edges_e2_y[idx - 1], m_obb_edges_e2_z[idx - 1]};
					obb.e3 = {m_obb_edges_e3_x[idx - 1], m_obb_edges_e3_y[idx - 1], m_obb_edges_e3_z[idx - 1]};
					obb.extent = {m_obb_edges_extent_x[idx - 1], m_obb_edges_extent_y[idx - 1], m_obb_edges_extent_z[idx - 1]};
					m_obb_edges_res.push_back(obb);					
				}
				
				start = m_obb_faces_start[idx - 1];
				end = m_obb_faces_end[idx - 1];
				for(int i = start; i < end; i++)
				{
					OBB obb;
					obb.center = {m_obb_faces_center_x[idx - 1], m_obb_faces_center_y[idx - 1], m_obb_faces_center_z[idx - 1]};
					obb.e1 = {m_obb_faces_e1_x[idx - 1], m_obb_faces_e1_y[idx - 1], m_obb_faces_e1_z[idx - 1]};
					obb.e2 = {m_obb_faces_e2_x[idx - 1], m_obb_faces_e2_y[idx - 1], m_obb_faces_e2_z[idx - 1]};
					obb.e3 = {m_obb_faces_e3_x[idx - 1], m_obb_faces_e3_y[idx - 1], m_obb_faces_e3_z[idx - 1]};
					obb.extent = {m_obb_faces_extent_x[idx - 1], m_obb_faces_extent_y[idx - 1], m_obb_faces_extent_z[idx - 1]};
					m_obb_faces_res.push_back(obb);					
				}
				
						
				res.m_vertices = m_vertices_res;
				res.m_inertia_on_mass = m_inertia_on_mass_res;
				res.m_obb_vertices = m_obb_vertices_res;
				res.m_obb_edges = m_obb_edges_res;
				res.m_obb_faces = m_obb_faces_res;
				res.obb = obb_res;
				res.m_edges = m_edges_res;
				res.m_faces = m_faces_res;
				res.m_radius = m_radius_res;
				res.m_volume = m_volume_res;
				res.m_name = m_name_res;
				return res;				
			}

		ONIKA_HOST_DEVICE_FUNC
			inline shape operator[] (const std::string name)
			{
				shape res;
				
				for (int idx = 0; idx < num_shapes; idx++)
				{
					if(m_name[idx] == name)
					{
						
						VectorT<exanb::Vec3d> m_vertices_res;
						exanb::Vec3d m_inertia_on_mass_res;
						VectorT<OBB> m_obb_vertices_res;
						VectorT<OBB> m_obb_edges_res;
						VectorT<OBB> m_obb_faces_res;
						OBB obb_res;
						VectorT<int> m_edges_res;
						VectorT<int> m_faces_res;
						double m_radius_res;
						double m_volume_res;
						std::string m_name_res;
				
						m_inertia_on_mass_res = {m_inertia_on_mass_x[idx - 1], m_inertia_on_mass_y[idx - 1], m_inertia_on_mass_z[idx - 1]};
						obb_res.center = {obb_center_x[idx - 1], obb_center_y[idx - 1], obb_center_z[idx - 1]};
						obb_res.e1 = {obb_e1_x[idx - 1], obb_e1_y[idx - 1], obb_e1_z[idx - 1]};
						obb_res.e2 = {obb_e2_x[idx - 1], obb_e2_y[idx - 1], obb_e2_z[idx - 1]};
						obb_res.e3 = {obb_e3_x[idx - 1], obb_e3_y[idx - 1], obb_e3_z[idx - 1]};
						obb_res.extent = {obb_extent_x[idx - 1], obb_extent_y[idx - 1], obb_extent_z[idx - 1]};
						m_radius_res = m_radius[idx - 1];
						m_volume_res = m_volume[idx - 1];
						m_name_res = m_name[idx - 1];
				
						int start = m_vertices_start[idx - 1];
						int end = m_vertices_end[idx - 1];
						for(int i = start; i< end; i++)
						{
							m_vertices_res.push_back({m_vertices_x[i], m_vertices_y[i], m_vertices_z[i]});
						}
				
						start = m_edges_start[idx - 1];
						end = m_edges_end[idx - 1];
						for(int i = start; i< end; i++)
						{
							m_edges_res.push_back(m_edges[i]);
						}
				
						start = m_faces_start[idx - 1];
						end = m_faces_end[idx - 1];
						for(int i = start; i< end; i++)
						{
							m_faces_res.push_back(m_faces[i]);
						}
				
						start = m_obb_vertices_start[idx - 1];
						end = m_obb_vertices_end[idx - 1];
						for(int i = start; i < end; i++)
						{
							OBB obb;
							obb.center = {m_obb_vertices_center_x[idx - 1], m_obb_vertices_center_y[idx - 1], m_obb_vertices_center_z[idx - 1]};
							obb.e1 = {m_obb_vertices_e1_x[idx - 1], m_obb_vertices_e1_y[idx - 1], m_obb_vertices_e1_z[idx - 1]};
							obb.e2 = {m_obb_vertices_e2_x[idx - 1], m_obb_vertices_e2_y[idx - 1], m_obb_vertices_e2_z[idx - 1]};
							obb.e3 = {m_obb_vertices_e3_x[idx - 1], m_obb_vertices_e3_y[idx - 1], m_obb_vertices_e3_z[idx - 1]};
							obb.extent = {m_obb_vertices_extent_x[idx - 1], m_obb_vertices_extent_y[idx - 1], m_obb_vertices_extent_z[idx - 1]};
							m_obb_vertices_res.push_back(obb);					
						}
				
						start = m_obb_edges_start[idx - 1];
						end = m_obb_edges_end[idx - 1];
						for(int i = start; i < end; i++)
						{
							OBB obb;
							obb.center = {m_obb_edges_center_x[idx - 1], m_obb_edges_center_y[idx - 1], m_obb_edges_center_z[idx - 1]};
							obb.e1 = {m_obb_edges_e1_x[idx - 1], m_obb_edges_e1_y[idx - 1], m_obb_edges_e1_z[idx - 1]};
							obb.e2 = {m_obb_edges_e2_x[idx - 1], m_obb_edges_e2_y[idx - 1], m_obb_edges_e2_z[idx - 1]};
							obb.e3 = {m_obb_edges_e3_x[idx - 1], m_obb_edges_e3_y[idx - 1], m_obb_edges_e3_z[idx - 1]};
							obb.extent = {m_obb_edges_extent_x[idx - 1], m_obb_edges_extent_y[idx - 1], m_obb_edges_extent_z[idx - 1]};
							m_obb_edges_res.push_back(obb);					
						}
				
						start = m_obb_faces_start[idx - 1];
						end = m_obb_faces_end[idx - 1];
						for(int i = start; i < end; i++)
						{
							OBB obb;
							obb.center = {m_obb_faces_center_x[idx - 1], m_obb_faces_center_y[idx - 1], m_obb_faces_center_z[idx - 1]};
							obb.e1 = {m_obb_faces_e1_x[idx - 1], m_obb_faces_e1_y[idx - 1], m_obb_faces_e1_z[idx - 1]};
							obb.e2 = {m_obb_faces_e2_x[idx - 1], m_obb_faces_e2_y[idx - 1], m_obb_faces_e2_z[idx - 1]};
							obb.e3 = {m_obb_faces_e3_x[idx - 1], m_obb_faces_e3_y[idx - 1], m_obb_faces_e3_z[idx - 1]};
							obb.extent = {m_obb_faces_extent_x[idx - 1], m_obb_faces_extent_y[idx - 1], m_obb_faces_extent_z[idx - 1]};
							m_obb_faces_res.push_back(obb);					
						}						
						
						res.m_vertices = m_vertices_res;
						res.m_inertia_on_mass = m_inertia_on_mass_res;
						res.m_obb_vertices = m_obb_vertices_res;
						res.m_obb_edges = m_obb_edges_res;
						res.m_obb_faces = m_obb_faces_res;
						res.obb = obb_res;
						res.m_edges = m_edges_res;
						res.m_faces = m_faces_res;
						res.m_radius = m_radius_res;
						res.m_volume = m_volume_res;
						res.m_name = m_name_res;
						return res;
					}
				}
				//std::cout << "Warning, the shape: " << name << " is not included in this collection of shapes. We return a nullptr." << std::endl;
				return res;
			}

		inline void add_shape(shape* shp)
		{
			//this->m_data.push_back(*shp); // copy
			VectorT<exanb::Vec3d> m_vertices_shp = shp->m_vertices;
			exanb::Vec3d m_inertia_on_mass_shp = shp->m_inertia_on_mass;
			VectorT<OBB> m_obb_vertices_shp = shp->m_obb_vertices;
			VectorT<OBB> m_obb_edges_shp = shp->m_obb_edges;
			VectorT<OBB> m_obb_faces_shp = shp->m_obb_faces;
			OBB obb_shp = shp->obb;
			VectorT<int> m_edges_shp = shp->m_edges;
			VectorT<int> m_faces_shp = shp->m_faces;
			double m_radius_shp = shp->m_radius;
			double m_volume_shp = shp->m_volume;
			std::string m_name_shp = shp->m_name;
			
			m_inertia_on_mass_x.push_back(m_inertia_on_mass_shp.x);
			m_inertia_on_mass_y.push_back(m_inertia_on_mass_shp.y);
			m_inertia_on_mass_z.push_back(m_inertia_on_mass_shp.z);
			
			obb_center_x.push_back(obb_shp.center.x);
			obb_center_y.push_back(obb_shp.center.y);
			obb_center_z.push_back(obb_shp.center.z);
			obb_e1_x.push_back(obb_shp.e1.x);
			obb_e1_y.push_back(obb_shp.e1.y);
			obb_e1_z.push_back(obb_shp.e1.z);
			obb_e2_x.push_back(obb_shp.e2.x);
			obb_e2_y.push_back(obb_shp.e2.y);
			obb_e2_z.push_back(obb_shp.e2.z);
			obb_e3_x.push_back(obb_shp.e3.x);
			obb_e3_y.push_back(obb_shp.e3.y);
			obb_e3_z.push_back(obb_shp.e3.z);
			obb_extent_x.push_back(obb_shp.extent.x);
			obb_extent_y.push_back(obb_shp.extent.y);
			obb_extent_z.push_back(obb_shp.extent.z);
			
			m_radius.push_back(m_radius_shp);
			m_volume.push_back(m_volume_shp);
			
			m_name.push_back(m_name_shp);
			
			int start;// = m_vertices_end[num_shapes - 1];
			if(num_shapes == 0)
			{
				start = 0;
			}
			else
			{
				start = m_vertices_end[num_shapes - 1];
			}
			m_vertices_start.push_back(start);
			int end = start + m_vertices_shp.size();
			m_vertices_end.push_back(end);
			for(auto m : m_vertices_shp)
			{
				m_vertices_x.push_back(m.x);
				m_vertices_y.push_back(m.y);
				m_vertices_z.push_back(m.z);
			}
			
			if(num_shapes == 0)
			{
				start = 0;
			}
			else
			{
				start = m_obb_vertices_end[num_shapes - 1];
			}
			m_obb_vertices_start.push_back(start);
			end = start + m_obb_vertices_shp.size();
			m_obb_vertices_end.push_back(end);
			for(auto o : m_obb_vertices_shp)
			{
				m_obb_vertices_center_x.push_back(o.center.x);
				m_obb_vertices_center_y.push_back(o.center.y);
				m_obb_vertices_center_z.push_back(o.center.z);
				m_obb_vertices_e1_x.push_back(o.e1.x);
				m_obb_vertices_e1_y.push_back(o.e1.y);
				m_obb_vertices_e1_z.push_back(o.e1.z);
				m_obb_vertices_e2_x.push_back(o.e2.x);
				m_obb_vertices_e2_y.push_back(o.e2.y);
				m_obb_vertices_e2_z.push_back(o.e2.z);
				m_obb_vertices_e3_x.push_back(o.e3.x);
				m_obb_vertices_e3_y.push_back(o.e3.y);
				m_obb_vertices_e3_z.push_back(o.e3.z);
				m_obb_vertices_extent_x.push_back(o.extent.x);
				m_obb_vertices_extent_y.push_back(o.extent.y);
				m_obb_vertices_extent_z.push_back(o.extent.z);
			}
			
			if(num_shapes == 0)
			{
				start = 0;
			}
			else
			{
				start = m_obb_edges_end[num_shapes - 1];
			}
			m_obb_edges_start.push_back(start);
			end = start + m_obb_edges_shp.size();
			m_obb_edges_end.push_back(end);
			for(auto o : m_obb_edges_shp)
			{
				m_obb_edges_center_x.push_back(o.center.x);
				m_obb_edges_center_y.push_back(o.center.y);
				m_obb_edges_center_z.push_back(o.center.z);
				m_obb_edges_e1_x.push_back(o.e1.x);
				m_obb_edges_e1_y.push_back(o.e1.y);
				m_obb_edges_e1_z.push_back(o.e1.z);
				m_obb_edges_e2_x.push_back(o.e2.x);
				m_obb_edges_e2_y.push_back(o.e2.y);
				m_obb_edges_e2_z.push_back(o.e2.z);
				m_obb_edges_e3_x.push_back(o.e3.x);
				m_obb_edges_e3_y.push_back(o.e3.y);
				m_obb_edges_e3_z.push_back(o.e3.z);
				m_obb_edges_extent_x.push_back(o.extent.x);
				m_obb_edges_extent_y.push_back(o.extent.y);
				m_obb_edges_extent_z.push_back(o.extent.z);
			}
			
			if(num_shapes == 0)
			{
				start = 0;
			}
			else
			{
				start = m_obb_faces_end[num_shapes - 1];
			}
			m_obb_faces_start.push_back(start);
			end = start + m_obb_faces_shp.size();
			m_obb_faces_end.push_back(end);
			for(auto o : m_obb_faces_shp)
			{
				m_obb_faces_center_x.push_back(o.center.x);
				m_obb_faces_center_y.push_back(o.center.y);
				m_obb_faces_center_z.push_back(o.center.z);
				m_obb_faces_e1_x.push_back(o.e1.x);
				m_obb_faces_e1_y.push_back(o.e1.y);
				m_obb_faces_e1_z.push_back(o.e1.z);
				m_obb_faces_e2_x.push_back(o.e2.x);
				m_obb_faces_e2_y.push_back(o.e2.y);
				m_obb_faces_e2_z.push_back(o.e2.z);
				m_obb_faces_e3_x.push_back(o.e3.x);
				m_obb_faces_e3_y.push_back(o.e3.y);
				m_obb_faces_e3_z.push_back(o.e3.z);
				m_obb_faces_extent_x.push_back(o.extent.x);
				m_obb_faces_extent_y.push_back(o.extent.y);
				m_obb_faces_extent_z.push_back(o.extent.z);
			}
			
			if(num_shapes == 0)
			{
				start = 0;
			}
			else
			{
				start = m_edges_end[num_shapes - 1];
			}
			m_edges_start.push_back(start);
			end = start + m_edges_shp.size();
			m_edges_end.push_back(end);
			for(auto m : m_edges_shp)
			{
				m_edges.push_back(m);
			}
			
			if(num_shapes == 0)
			{
				start = 0;
			}
			else
			{
				start = m_faces_end[num_shapes - 1];
			}
			m_faces_start.push_back(start);
			end = start + m_faces_shp.size();
			m_faces_end.push_back(end);
			for(auto m : m_faces_shp)
			{
				m_faces.push_back(m);
			}
			
			num_shapes++;						
		}
		
		ONIKA_HOST_DEVICE_FUNC
		inline int get_number_of_vertices(const uint32_t idx)
		{
			auto* start = onika::cuda::vector_data(m_vertices_start);
			auto* end = onika::cuda::vector_data(m_vertices_end);
			
			return end[idx] - start[idx];
		}
		
		ONIKA_HOST_DEVICE_FUNC
		inline int get_number_of_vertices(const uint32_t idx) const
		{
			auto* start = onika::cuda::vector_data(m_vertices_start);
			auto* end = onika::cuda::vector_data(m_vertices_end);
			
			return end[idx] - start[idx];			
		}
		
		ONIKA_HOST_DEVICE_FUNC
		inline exanb::Vec3d get_vertex(const uint32_t idx, const int i, const exanb::Vec3d& p, const exanb::Quaternion& orient)
		{
			auto* start = onika::cuda::vector_data(m_vertices_start);
			auto* vertices_x = onika::cuda::vector_data(m_vertices_x);
			auto* vertices_y = onika::cuda::vector_data(m_vertices_y);
			auto* vertices_z = onika::cuda::vector_data(m_vertices_z);
			
			Vec3d vertex = {vertices_x[start[idx] + i], vertices_y[start[idx] + i], vertices_z[start[idx] + i]};
			 
			return p + orient * vertex;
		}
		
		ONIKA_HOST_DEVICE_FUNC
		inline exanb::Vec3d get_vertex(const uint32_t idx, const int i, const exanb::Vec3d& p, const exanb::Quaternion& orient) const
		{
			auto* start = onika::cuda::vector_data(m_vertices_start);
			auto* vertices_x = onika::cuda::vector_data(m_vertices_x);
			auto* vertices_y = onika::cuda::vector_data(m_vertices_y);
			auto* vertices_z = onika::cuda::vector_data(m_vertices_z);
			
			Vec3d vertex = {vertices_x[start[idx] + i], vertices_y[start[idx] + i], vertices_z[start[idx] + i]};
			 
			return p + orient * vertex;
		}
	};
}
