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
//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <memory>

#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/compute_contact_force.h>

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/interactionSOA.hpp>
#include <exaDEM/interaction/interactionAOS.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/classifier.hpp>
#include <exaDEM/interaction/classifier_for_all.hpp>
#include <exaDEM/itools/itools.hpp>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/shape/shape_detection_driver.hpp>
#include <exaDEM/drivers.h>
#include <exaDEM/contact_polyhedron.h>

namespace exaDEM
{
  using namespace exanb;
  using namespace polyhedron;
  
  ONIKA_HOST_DEVICE_FUNC
  inline std::tuple<bool, double, Vec3d, Vec3d> detection_precompute(const uint16_t type, const VertexArray &pi, const int i, const shape *shpi, const VertexArray &pj, const int j, const shape *shpj)
  {
#       define __params__     pi, i, shpi, pj, j, shpj
  	if(type==0)
  	{
  		return detection_vertex_vertex_precompute(__params__);
  	}
  	else if(type==1)
  	{
  		return detection_vertex_edge_precompute(__params__);
  	}
  	else if(type==2)
  	{
  		return detection_vertex_face_precompute(__params__);
  	}
  	else if(type==3)
  	{
  		return detection_edge_edge_precompute(__params__);
  	}
  	
  	return std::tuple<bool, double, Vec3d, Vec3d>();
  }
  
  
  __device__ void kernelContact1(Interaction2& item,
  					double* rx,
  					double* ry,
  					double* rz,
  					double* vrotx,
  					double* vroty,
  					double* vrotz,
  					double* vx,
  					double* vy,
  					double* vz,
  					double* mass,
  					double* momx,
  					double* momy,
  					double* momz,
  					double* fx,
  					double* fy,
  					double* fz,
  					uint32_t* type,
  					VertexArray* vertices,
  					const shape *const shps,
  					const double time,
  					const ContactParams hkp)
  {
  	auto& idi = item.id_i;
  	auto& idj = item.id_j;
  	
  	auto& rx_i = rx[idi];
  	auto& ry_i = ry[idi];
  	auto& rz_i = rz[idi];
  	
  	auto& rx_j = rx[idj];
  	auto& ry_j = ry[idj];
  	auto& rz_j = rz[idj];
  	
  	Vec3d ri = {rx_i, ry_i, rz_i};
  	Vec3d rj = {rx_j, ry_j, rz_j};
  	
  	auto& vrotx_i = vrotx[idi];
  	auto& vroty_i = vroty[idi];
  	auto& vrotz_i = vrotz[idi];
  	
  	auto& vrotx_j = vrotx[idj];
  	auto& vroty_j = vroty[idj];
  	auto& vrotz_j = vrotz[idj];
  	
  	Vec3d vrot_i = {vrotx_i, vroty_i, vrotz_i};
  	Vec3d vrot_j = {vrotx_j, vroty_j, vrotz_j};
  	
  	auto& type_i = type[idi];
  	auto& type_j = type[idj];
  	
  	auto& vertices_i = vertices[idi];
  	auto& vertices_j = vertices[idj];
  	
  	const shape& shp_i = shps[type_i];
  	const shape& shp_j = shps[type_j];
  	
  	auto [contact, dn, n, contact_position] = detection_vertex_vertex_precompute(vertices_i, item.sub_i, &shp_i, vertices_j, item.sub_j, &shp_j);
  	
  	Vec3d f = {0, 0, 0};
  	Vec3d fn = {0, 0, 0};
  	if(contact)
  	{
  		printf("CONTACT\n");
  		const Vec3d vi = {vx[idi], vy[idi], vz[idi]};
  		const Vec3d vj = {vx[idj], vy[idj], vz[idj]};
  		const auto& m_i = mass[idi];
  		const auto& m_j = mass[idj];
  		
  		const double meff = compute_effective_mass(m_i, m_j);
  		
  		contact_force_core(dn, n, time, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item.friction, contact_position, ri, vi, f, item.moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);
          	
          	fn = f - item.friction;
          	
  		Vec3d mom_i = {momx[idi], momy[idi], momz[idi]};
  		Vec3d mom_i_res = compute_moments(contact_position, ri, f, item.moment);
  		atomicAdd(&momx[idi], mom_i_res.x);
  		atomicAdd(&momy[idi], mom_i_res.y);
  		atomicAdd(&momz[idi], mom_i_res.z);
  		atomicAdd(&fx[idi], f.x);
  		atomicAdd(&fy[idi], f.y);
  		atomicAdd(&fz[idi], f.z);
  			
  		Vec3d mom_j = {momx[idj], momy[idj], momz[idj]};
  		Vec3d mom_j_res = compute_moments(contact_position, rj, -f, -item.moment);
  		atomicAdd(&momx[idj], mom_j_res.x);
  		atomicAdd(&momy[idj], mom_j_res.y);
  		atomicAdd(&momz[idj], mom_j_res.z);
  		atomicAdd(&fx[idj], -f.x);
  		atomicAdd(&fy[idj], -f.y);
  		atomicAdd(&fz[idj], -f.z);            	
  	}
  	else
  	{
  		item.reset();
  		dn = 0;
  	}
  }
  
  __device__ void kernelContact2(Interaction2& item,
  					double* rx,
  					double* ry,
  					double* rz,
  					double* vrotx,
  					double* vroty,
  					double* vrotz,
  					double* vx,
  					double* vy,
  					double* vz,
  					double* mass,
  					double* momx,
  					double* momy,
  					double* momz,
  					double* fx,
  					double* fy,
  					double* fz,
  					uint32_t* type,
  					VertexArray* vertices,
  					const shape *const shps,
  					const double time,
  					const ContactParams hkp)
  {
  	auto& idi = item.id_i;
  	auto& idj = item.id_j;
  	
  	auto& rx_i = rx[idi];
  	auto& ry_i = ry[idi];
  	auto& rz_i = rz[idi];
  	
  	auto& rx_j = rx[idj];
  	auto& ry_j = ry[idj];
  	auto& rz_j = rz[idj];
  	
  	Vec3d ri = {rx_i, ry_i, rz_i};
  	Vec3d rj = {rx_j, ry_j, rz_j};
  	
  	auto& vrotx_i = vrotx[idi];
  	auto& vroty_i = vroty[idi];
  	auto& vrotz_i = vrotz[idi];
  	
  	auto& vrotx_j = vrotx[idj];
  	auto& vroty_j = vroty[idj];
  	auto& vrotz_j = vrotz[idj];
  	
  	Vec3d vrot_i = {vrotx_i, vroty_i, vrotz_i};
  	Vec3d vrot_j = {vrotx_j, vroty_j, vrotz_j};
  	
  	auto& type_i = type[idi];
  	auto& type_j = type[idj];
  	
  	auto& vertices_i = vertices[idi];
  	auto& vertices_j = vertices[idj];
  	
  	const shape& shp_i = shps[type_i];
  	const shape& shp_j = shps[type_j];
  	
  	auto [contact, dn, n, contact_position] = detection_vertex_edge_precompute(vertices_i, item.sub_i, &shp_i, vertices_j, item.sub_j, &shp_j);
  	
  	//contact = false;
  	
  	//item.reset();
  	
  	Vec3d f = {0, 0, 0};
  	Vec3d fn = {0, 0, 0};
  	if(contact)
  	{
  		//const Vec3d vi = {vx[idi], vy[idi], vz[idi]};
  		//const Vec3d vj = {vx[idj], vy[idj], vz[idj]};
  		//const auto& m_i = mass[idi];
  		//const auto& m_j = mass[idj];
  		
  		//const double meff = compute_effective_mass(m_i, m_j);
  		
  		/*contact_force_core(dn, n, time, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item.friction, contact_position, ri, vi, f, item.moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);
          	
          	fn = f - item.friction;
          	
  		Vec3d mom_i = {momx[idi], momy[idi], momz[idi]};
  		Vec3d mom_i_res = compute_moments(contact_position, ri, f, item.moment);
  		atomicAdd(&momx[idi], mom_i_res.x);
  		atomicAdd(&momy[idi], mom_i_res.y);
  		atomicAdd(&momz[idi], mom_i_res.z);
  		atomicAdd(&fx[idi], f.x);
  		atomicAdd(&fy[idi], f.y);
  		atomicAdd(&fz[idi], f.z);
  			
  		Vec3d mom_j = {momx[idj], momy[idj], momz[idj]};
  		Vec3d mom_j_res = compute_moments(contact_position, rj, -f, -item.moment);
  		atomicAdd(&momx[idj], mom_j_res.x);
  		atomicAdd(&momy[idj], mom_j_res.y);
  		atomicAdd(&momz[idj], mom_j_res.z);
  		atomicAdd(&fx[idj], -f.x);
  		atomicAdd(&fy[idj], -f.y);
  		atomicAdd(&fz[idj], -f.z);*/          	
  	}
  	else
  	{
  		item.reset();
  	//	dn = 0;
  	}
  	
  }
  
  __device__ void kernelContact3(Interaction2& item,
  					double* rx,
  					double* ry,
  					double* rz,
  					double* vrotx,
  					double* vroty,
  					double* vrotz,
  					double* vx,
  					double* vy,
  					double* vz,
  					double* mass,
  					double* momx,
  					double* momy,
  					double* momz,
  					double* fx,
  					double* fy,
  					double* fz,
  					uint32_t* type,
  					VertexArray* vertices,
  					const shape *const shps,
  					const double time,
  					const ContactParams hkp)
  {
  	auto& idi = item.id_i;
  	auto& idj = item.id_j;
  	
  	auto& rx_i = rx[idi];
  	auto& ry_i = ry[idi];
  	auto& rz_i = rz[idi];
  	
  	auto& rx_j = rx[idj];
  	auto& ry_j = ry[idj];
  	auto& rz_j = rz[idj];
  	
  	Vec3d ri = {rx_i, ry_i, rz_i};
  	Vec3d rj = {rx_j, ry_j, rz_j};
  	
  	auto& vrotx_i = vrotx[idi];
  	auto& vroty_i = vroty[idi];
  	auto& vrotz_i = vrotz[idi];
  	
  	auto& vrotx_j = vrotx[idj];
  	auto& vroty_j = vroty[idj];
  	auto& vrotz_j = vrotz[idj];
  	
  	Vec3d vrot_i = {vrotx_i, vroty_i, vrotz_i};
  	Vec3d vrot_j = {vrotx_j, vroty_j, vrotz_j};
  	
  	auto& type_i = type[idi];
  	auto& type_j = type[idj];
  	
  	auto& vertices_i = vertices[idi];
  	auto& vertices_j = vertices[idj];
  	
  	const shape& shp_i = shps[type_i];
  	const shape& shp_j = shps[type_j];
  	
  	auto [contact, dn, n, contact_position] = detection_vertex_face_precompute(vertices_i, item.sub_i, &shp_i, vertices_j, item.sub_j, &shp_j);
  	
  	Vec3d f = {0, 0, 0};
  	Vec3d fn = {0, 0, 0};
  	if(contact)
  	{
  		const Vec3d vi = {vx[idi], vy[idi], vz[idi]};
  		const Vec3d vj = {vx[idj], vy[idj], vz[idj]};
  		const auto& m_i = mass[idi];
  		const auto& m_j = mass[idj];
  		
  		const double meff = compute_effective_mass(m_i, m_j);
  		
  		contact_force_core(dn, n, time, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item.friction, contact_position, ri, vi, f, item.moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);
          	
          	fn = f - item.friction;
          	
  		Vec3d mom_i = {momx[idi], momy[idi], momz[idi]};
  		Vec3d mom_i_res = compute_moments(contact_position, ri, f, item.moment);
  		atomicAdd(&momx[idi], mom_i_res.x);
  		atomicAdd(&momy[idi], mom_i_res.y);
  		atomicAdd(&momz[idi], mom_i_res.z);
  		atomicAdd(&fx[idi], f.x);
  		atomicAdd(&fy[idi], f.y);
  		atomicAdd(&fz[idi], f.z);
  			
  		Vec3d mom_j = {momx[idj], momy[idj], momz[idj]};
  		Vec3d mom_j_res = compute_moments(contact_position, rj, -f, -item.moment);
  		atomicAdd(&momx[idj], mom_j_res.x);
  		atomicAdd(&momy[idj], mom_j_res.y);
  		atomicAdd(&momz[idj], mom_j_res.z);
  		atomicAdd(&fx[idj], -f.x);
  		atomicAdd(&fy[idj], -f.y);
  		atomicAdd(&fz[idj], -f.z);            	
  	}
  	else
  	{
  		item.reset();
  		dn = 0;
  	}
  }
  
  __device__ void kernelContact4(Interaction2& item,
  					double* rx,
  					double* ry,
  					double* rz,
  					double* vrotx,
  					double* vroty,
  					double* vrotz,
  					double* vx,
  					double* vy,
  					double* vz,
  					double* mass,
  					double* momx,
  					double* momy,
  					double* momz,
  					double* fx,
  					double* fy,
  					double* fz,
  					uint32_t* type,
  					VertexArray* vertices,
  					const shape *const shps,
  					const double time,
  					const ContactParams hkp)
  {
  	auto& idi = item.id_i;
  	auto& idj = item.id_j;
  	
  	auto& rx_i = rx[idi];
  	auto& ry_i = ry[idi];
  	auto& rz_i = rz[idi];
  	
  	auto& rx_j = rx[idj];
  	auto& ry_j = ry[idj];
  	auto& rz_j = rz[idj];
  	
  	Vec3d ri = {rx_i, ry_i, rz_i};
  	Vec3d rj = {rx_j, ry_j, rz_j};
  	
  	auto& vrotx_i = vrotx[idi];
  	auto& vroty_i = vroty[idi];
  	auto& vrotz_i = vrotz[idi];
  	
  	auto& vrotx_j = vrotx[idj];
  	auto& vroty_j = vroty[idj];
  	auto& vrotz_j = vrotz[idj];
  	
  	Vec3d vrot_i = {vrotx_i, vroty_i, vrotz_i};
  	Vec3d vrot_j = {vrotx_j, vroty_j, vrotz_j};
  	
  	auto& type_i = type[idi];
  	auto& type_j = type[idj];
  	
  	auto& vertices_i = vertices[idi];
  	auto& vertices_j = vertices[idj];
  	
  	const shape& shp_i = shps[type_i];
  	const shape& shp_j = shps[type_j];
  	
  	auto [contact, dn, n, contact_position] = detection_edge_edge_precompute(vertices_i, item.sub_i, &shp_i, vertices_j, item.sub_j, &shp_j);
  	
  	Vec3d f = {0, 0, 0};
  	Vec3d fn = {0, 0, 0};
  	if(contact)
  	{
  		const Vec3d vi = {vx[idi], vy[idi], vz[idi]};
  		const Vec3d vj = {vx[idj], vy[idj], vz[idj]};
  		const auto& m_i = mass[idi];
  		const auto& m_j = mass[idj];
  		
  		const double meff = compute_effective_mass(m_i, m_j);
  		
  		contact_force_core(dn, n, time, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item.friction, contact_position, ri, vi, f, item.moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);
          	
          	fn = f - item.friction;
          	
  		Vec3d mom_i = {momx[idi], momy[idi], momz[idi]};
  		Vec3d mom_i_res = compute_moments(contact_position, ri, f, item.moment);
  		atomicAdd(&momx[idi], mom_i_res.x);
  		atomicAdd(&momy[idi], mom_i_res.y);
  		atomicAdd(&momz[idi], mom_i_res.z);
  		atomicAdd(&fx[idi], f.x);
  		atomicAdd(&fy[idi], f.y);
  		atomicAdd(&fz[idi], f.z);
  			
  		Vec3d mom_j = {momx[idj], momy[idj], momz[idj]};
  		Vec3d mom_j_res = compute_moments(contact_position, rj, -f, -item.moment);
  		atomicAdd(&momx[idj], mom_j_res.x);
  		atomicAdd(&momy[idj], mom_j_res.y);
  		atomicAdd(&momz[idj], mom_j_res.z);
  		atomicAdd(&fx[idj], -f.x);
  		atomicAdd(&fy[idj], -f.y);
  		atomicAdd(&fz[idj], -f.z);
  		            	
  	}
  	else
  	{
  		item.reset();
  		dn = 0;
  	}
  }
  
  __device__ void kernelContactTotal(Interaction2& item,
  					double* rx,
  					double* ry,
  					double* rz,
  					double* vrotx,
  					double* vroty,
  					double* vrotz,
  					double* vx,
  					double* vy,
  					double* vz,
  					double* mass,
  					double* momx,
  					double* momy,
  					double* momz,
  					double* fx,
  					double* fy,
  					double* fz,
  					uint32_t* type,
  					VertexArray* vertices,
  					const shape *const shps,
  					const double time,
  					const ContactParams hkp,
  					int type2)
  {
  	auto& idi = item.id_i;
  	auto& idj = item.id_j;
  	
  	auto& rx_i = rx[idi];
  	auto& ry_i = ry[idi];
  	auto& rz_i = rz[idi];
  	
  	auto& rx_j = rx[idj];
  	auto& ry_j = ry[idj];
  	auto& rz_j = rz[idj];
  	
  	Vec3d ri = {rx_i, ry_i, rz_i};
  	Vec3d rj = {rx_j, ry_j, rz_j};
  	
  	auto& vrotx_i = vrotx[idi];
  	auto& vroty_i = vroty[idi];
  	auto& vrotz_i = vrotz[idi];
  	
  	auto& vrotx_j = vrotx[idj];
  	auto& vroty_j = vroty[idj];
  	auto& vrotz_j = vrotz[idj];
  	
  	Vec3d vrot_i = {vrotx_i, vroty_i, vrotz_i};
  	Vec3d vrot_j = {vrotx_j, vroty_j, vrotz_j};
  	
  	auto& type_i = type[idi];
  	auto& type_j = type[idj];
  	
  	auto& vertices_i = vertices[idi];
  	auto& vertices_j = vertices[idj];
  	
  	const shape& shp_i = shps[type_i];
  	const shape& shp_j = shps[type_j];

	auto [contact, dn, n, contact_position] = detection_precompute(type2, vertices_i, item.sub_i, &shp_i, vertices_j, item.sub_j, &shp_j);

  	Vec3d f = {0, 0, 0};
  	Vec3d fn = {0, 0, 0};
  	if(contact)
  	{
  		const Vec3d vi = {vx[idi], vy[idi], vz[idi]};
  		const Vec3d vj = {vx[idj], vy[idj], vz[idj]};
  		const auto& m_i = mass[idi];
  		const auto& m_j = mass[idj];
  		
  		const double meff = compute_effective_mass(m_i, m_j);
  		
  		contact_force_core(dn, n, time, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item.friction, contact_position, ri, vi, f, item.moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);
          	
          	fn = f - item.friction;
          	
  		Vec3d mom_i = {momx[idi], momy[idi], momz[idi]};
  		Vec3d mom_i_res = compute_moments(contact_position, ri, f, item.moment);
  		atomicAdd(&momx[idi], mom_i_res.x);
  		atomicAdd(&momy[idi], mom_i_res.y);
  		atomicAdd(&momz[idi], mom_i_res.z);
  		atomicAdd(&fx[idi], f.x);
  		atomicAdd(&fy[idi], f.y);
  		atomicAdd(&fz[idi], f.z);
  			
  		Vec3d mom_j = {momx[idj], momy[idj], momz[idj]};
  		Vec3d mom_j_res = compute_moments(contact_position, rj, -f, -item.moment);
  		atomicAdd(&momx[idj], mom_j_res.x);
  		atomicAdd(&momy[idj], mom_j_res.y);
  		atomicAdd(&momz[idj], mom_j_res.z);
  		atomicAdd(&fx[idj], -f.x);
  		atomicAdd(&fy[idj], -f.y);
  		atomicAdd(&fz[idj], -f.z);
  		            	
  	}
  	else
  	{
  		item.reset();
  		dn = 0;
  	}
  }
  
  __global__ void kernel1(exaDEM::Interaction* interactions,
  				double* rx,
  				double* ry,
  				double* rz,
  				double* vrotx,
  				double* vroty,
  				double* vrotz,
  				double* vx,
  				double* vy,
  				double* vz,
  				double* mass,
  				double* momx,
  				double* momy,
  				double* momz,
  				double* fx,
  				double* fy,
  				double* fz,
  				uint32_t* type,
  				VertexArray* vertices,
  				const shape *const shps,
  				const double time,
  				const ContactParams hkp,
  				int size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		Interaction& item = interactions[idx];
  		
  		Interaction2 item2 = {item.friction, item.moment, item.id_i, item.id_j, item.sub_i, item.sub_j};
  		
  		kernelContact1(item2, rx, ry, rz, vrotx, vroty, vrotz, vx, vy, vz, mass, momx, momy, momz, fx, fy, fz, type, vertices, shps, time, hkp);
  		
  		item.update_friction_and_moment(item2);
  	}
  }	
  
  __global__ void kernel2(exaDEM::Interaction* interactions,
  				double* rx,
  				double* ry,
  				double* rz,
  				double* vrotx,
  				double* vroty,
  				double* vrotz,
  				double* vx,
  				double* vy,
  				double* vz,
  				double* mass,
  				double* momx,
  				double* momy,
  				double* momz,
  				double* fx,
  				double* fy,
  				double* fz,
  				uint32_t* type,
  				VertexArray* vertices,
  				const shape *const shps,
  				const double time,
  				const ContactParams hkp,
  				int size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		Interaction& item = interactions[idx];
  		
  		Interaction2 item2 = {item.friction, item.moment, item.id_i, item.id_j, item.sub_i, item.sub_j};
  		
  		kernelContact2(item2, rx, ry, rz, vrotx, vroty, vrotz, vx, vy, vz, mass, momx, momy, momz, fx, fy, fz, type, vertices, shps, time, hkp);
  		
  		item.update_friction_and_moment(item2);
  	}
  }
  
  __global__ void kernel3(exaDEM::Interaction* interactions,
  				double* rx,
  				double* ry,
  				double* rz,
  				double* vrotx,
  				double* vroty,
  				double* vrotz,
  				double* vx,
  				double* vy,
  				double* vz,
  				double* mass,
  				double* momx,
  				double* momy,
  				double* momz,
  				double* fx,
  				double* fy,
  				double* fz,
  				uint32_t* type,
  				VertexArray* vertices,
  				const shape *const shps,
  				const double time,
  				const ContactParams hkp,
  				int size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		Interaction& item = interactions[idx];
  		
  		Interaction2 item2 = {item.friction, item.moment, item.id_i, item.id_j, item.sub_i, item.sub_j};
  		
  		kernelContact3(item2, rx, ry, rz, vrotx, vroty, vrotz, vx, vy, vz, mass, momx, momy, momz, fx, fy, fz, type, vertices, shps, time, hkp);
  		
  		item.update_friction_and_moment(item2);
  	}
  }
  
  __global__ void kernel4(exaDEM::Interaction* interactions,
  				double* rx,
  				double* ry,
  				double* rz,
  				double* vrotx,
  				double* vroty,
  				double* vrotz,
  				double* vx,
  				double* vy,
  				double* vz,
  				double* mass,
  				double* momx,
  				double* momy,
  				double* momz,
  				double* fx,
  				double* fy,
  				double* fz,
  				uint32_t* type,
  				VertexArray* vertices,
  				const shape *const shps,
  				const double time,
  				const ContactParams hkp,
  				int size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		Interaction& item = interactions[idx];
  		
  		Interaction2 item2 = {item.friction, item.moment, item.id_i, item.id_j, item.sub_i, item.sub_j};
  		
  		kernelContact4(item2, rx, ry, rz, vrotx, vroty, vrotz, vx, vy, vz, mass, momx, momy, momz, fx, fy, fz, type, vertices, shps, time, hkp);
  		
  		item.update_friction_and_moment(item2);
  	}
  }
  
  __global__ void kernelTotal(exaDEM::Interaction* interactions,
  				double* rx,
  				double* ry,
  				double* rz,
  				double* vrotx,
  				double* vroty,
  				double* vrotz,
  				double* vx,
  				double* vy,
  				double* vz,
  				double* mass,
  				double* momx,
  				double* momy,
  				double* momz,
  				double* fx,
  				double* fy,
  				double* fz,
  				uint32_t* type,
  				VertexArray* vertices,
  				const shape *const shps,
  				const double time,
  				const ContactParams hkp,
  				int size,
  				int type2)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		Interaction& item = interactions[idx];
  		
  		Interaction2 item2 = {item.friction, item.moment, item.id_i, item.id_j, item.sub_i, item.sub_j};
  		
  		kernelContactTotal(item2, rx, ry, rz, vrotx, vroty, vrotz, vx, vy, vz, mass, momx, momy, momz, fx, fy, fz, type, vertices, shps, time, hkp, type2);
  		
  		item.update_friction_and_moment(item2);
  	}
  }  			

  template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>> class ComputeContactClassifierPolyhedronGPU : public OperatorNode
  {
    using driver_t = std::variant<exaDEM::Cylinder, exaDEM::Surface, exaDEM::Ball, exaDEM::Stl_mesh, exaDEM::UndefinedDriver>;
    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(ContactParams, config, INPUT, REQUIRED);        // can be re-used for to dump contact network
    ADD_SLOT(ContactParams, config_driver, INPUT, OPTIONAL); // can be re-used for to dump contact network
    ADD_SLOT(double, dt, INPUT, REQUIRED);
    ADD_SLOT(bool, symetric, INPUT_OUTPUT, REQUIRED, DocString{"Activate the use of symetric feature (contact law)"});
    ADD_SLOT(Drivers, drivers, INPUT, DocString{"List of Drivers {Cylinder, Surface, Ball, Mesh}"});
    ADD_SLOT(Classifier<InteractionAOS>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
    ADD_SLOT(shapes, shapes_collection, INPUT_OUTPUT, DocString{"Collection of shapes"});
    // analyses
    ADD_SLOT(long, timestep, INPUT, REQUIRED);
    ADD_SLOT(long, analysis_interaction_dump_frequency, INPUT, REQUIRED, DocString{"Write an interaction dump file"});
    ADD_SLOT(long, analysis_dump_stress_tensor_frequency, INPUT, REQUIRED, DocString{"Compute avg Stress Tensor."});
    ADD_SLOT(long, simulation_log_frequency, INPUT, REQUIRED, DocString{"Log frequency."});
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory name."});
    ADD_SLOT(std::string, interaction_basename, INPUT, REQUIRED, DocString{"Write an Output file containing interactions."});

  public:
    inline std::string documentation() const override final { return R"EOF(This operator computes forces between particles and particles/drivers using the contact law.)EOF"; }

    inline void execute() override final
    {
      if (grid->number_of_cells() == 0)
      {
        return;
      }

      /** Analysis */
      const long frequency_interaction = *analysis_interaction_dump_frequency;
      bool write_interactions = (frequency_interaction > 0 && (*timestep) % frequency_interaction == 0);

      const long frequency_stress_tensor = *analysis_dump_stress_tensor_frequency;
      bool compute_stress_tensor = (frequency_stress_tensor > 0 && (*timestep) % frequency_stress_tensor == 0);

      const long log_frequency = *simulation_log_frequency;
      bool need_interactions_for_log_frequency = (*timestep) % log_frequency;

      bool store_interactions = write_interactions || compute_stress_tensor || need_interactions_for_log_frequency;

      /** Get driver and particles data */
      driver_t *drvs = drivers->data();
      const auto cells = grid->cells();

      /** Get Contact Parameters and Shape */
      const ContactParams hkp = *config;
      ContactParams hkp_drvs{};
      const shape *const shps = shapes_collection->data();

      if (drivers->get_size() > 0 && config_driver.has_value())
      {
        hkp_drvs = *config_driver;
      }

      const double time = *dt;
      auto &classifier = *ic;

      /** Contact fexaDEM/orce kernels */
      contact_law_driver<Cylinder> cyli;
      contact_law_driver<Surface> surf;
      contact_law_driver<Ball> ball;
      contact_law_stl stlm;
      contact_law poly;

      if (*symetric == false)
      {
        lout << "The parameter symetric in contact classifier polyhedron has to be set to true." << std::endl;
        std::abort();
      }

#     define __params__ store_interactions, cells, hkp, shps, time
#     define __params_driver__ store_interactions, cells, drvs, hkp_drvs, shps, time

      int number_of_particles = grid->number_of_particles();

      onika::memory::CudaMMVector<double> rx;
      onika::memory::CudaMMVector<double> ry;
      onika::memory::CudaMMVector<double> rz;

      onika::memory::CudaMMVector<double> vrotx;
      onika::memory::CudaMMVector<double> vroty;
      onika::memory::CudaMMVector<double> vrotz;
      
      onika::memory::CudaMMVector<double> vx;
      onika::memory::CudaMMVector<double> vy;
      onika::memory::CudaMMVector<double> vz;
      
      onika::memory::CudaMMVector<double> mass;
      
      onika::memory::CudaMMVector<double> momx;
      onika::memory::CudaMMVector<double> momy;
      onika::memory::CudaMMVector<double> momz;
      
      onika::memory::CudaMMVector<double> fx;
      onika::memory::CudaMMVector<double> fy;
      onika::memory::CudaMMVector<double> fz;
      
      onika::memory::CudaMMVector<bool> pass;
      
      onika::memory::CudaMMVector<uint32_t> type_particule;
      
      onika::memory::CudaMMVector<VertexArray> vertices;
      
      rx.resize(number_of_particles);
      ry.resize(number_of_particles);
      rz.resize(number_of_particles);
      
      vrotx.resize(number_of_particles);
      vroty.resize(number_of_particles);
      vrotz.resize(number_of_particles);
        
      vx.resize(number_of_particles);
      vy.resize(number_of_particles);
      vz.resize(number_of_particles);
        
      mass.resize(number_of_particles);
        
      momx.resize(number_of_particles);
      momy.resize(number_of_particles);
      momz.resize(number_of_particles);
        
      fx.resize(number_of_particles);
      fy.resize(number_of_particles);
      fz.resize(number_of_particles);
        
      pass.resize(number_of_particles);
      
      type_particule.resize(number_of_particles);
      
      vertices.resize(number_of_particles);
      
      #pragma omp parallel for
      for(int i = 0; i < number_of_particles; i++)
      {
      	pass[i] = false;
      }
      
            

      for (size_t type = 0; type <= 3; type++)
      {
        auto [data, size] = classifier.get_info(type);
        InteractionWrapper<InteractionAOS> wrapper(data);
        
        for(int i = 0; i < size; i++)
        {
        	exaDEM::Interaction item = wrapper(i);
        	
        	auto& cell_i = cells[item.cell_i];
        	auto& cell_j = cells[item.cell_j];
        	
        	auto& p_i = item.p_i;
        	auto& p_j = item.p_j;
        	
        	auto id_i = item.id_i;
        	auto id_j = item.id_j;
        	
        	if(pass[id_i] == false)
        	{
        		rx[id_i] = cell_i[field::rx][p_i];
        		ry[id_i] = cell_i[field::ry][p_i];
        		rz[id_i] = cell_i[field::rz][p_i];

        		vrotx[id_i] = cell_i[field::vrot][p_i].x;
        		vroty[id_i] = cell_i[field::vrot][p_i].y;
        		vrotz[id_i] = cell_i[field::vrot][p_i].z;
        		
        		vx[id_i] = cell_i[field::vx][p_i];
        		vy[id_i] = cell_i[field::vy][p_i];
        		vz[id_i] = cell_i[field::vz][p_i];
        		
        		mass[id_i] = cell_i[field::mass][p_i];
        		
        		momx[id_i] = cell_i[field::mom][p_i].x;
        		momy[id_i] = cell_i[field::mom][p_i].y;
        		momz[id_i] = cell_i[field::mom][p_i].z;
        		
        		fx[id_i] = cell_i[field::fx][p_i];
        		fy[id_i] = cell_i[field::fy][p_i];
        		fz[id_i] = cell_i[field::fz][p_i];
        		
        		type_particule[id_i] = cell_i[field::type][p_i];
        		
        		vertices[id_i] = cell_i[field::vertices][p_i];
        		
        		pass[id_i] = true;
        	}
        	
        	if(pass[id_j] == false)
        	{
        		rx[id_j] = cell_i[field::rx][p_i];
        		ry[id_j] = cell_i[field::ry][p_i];
        		rz[id_j] = cell_i[field::rz][p_i];

        		vrotx[id_j] = cell_i[field::vrot][p_i].x;
        		vroty[id_j] = cell_i[field::vrot][p_i].y;
        		vrotz[id_j] = cell_i[field::vrot][p_i].z;
        		
        		vx[id_j] = cell_i[field::vx][p_i];
        		vy[id_j] = cell_i[field::vy][p_i];
        		vz[id_j] = cell_i[field::vz][p_i];
        		
        		mass[id_j] = cell_i[field::mass][p_i];
        		
        		momx[id_j] = cell_i[field::mom][p_i].x;
        		momy[id_j] = cell_i[field::mom][p_i].y;
        		momz[id_j] = cell_i[field::mom][p_i].z;
        		
        		fx[id_j] = cell_i[field::fx][p_i];
        		fy[id_j] = cell_i[field::fy][p_i];
        		fz[id_j] = cell_i[field::fz][p_i];
        		
        		type_particule[id_j] = cell_i[field::type][p_i];
        		
        		vertices[id_j] = cell_i[field::vertices][p_i];
        		
        		pass[id_j] = true;
        	}
        }
        
        int numBlocks = (size + 255) / 256;
        int threadsPerBlock = 256;
        
        /*if(type == 0)
        {
        	//kernel1<<<numBlocks, threadsPerBlock>>>(wrapper.interactions, rx.data(), ry.data(), rz.data(), vrotx.data(), vroty.data(), vrotz.data(), vx.data(), vy.data(), vz.data(), mass.data(), momx.data(), momy.data(), momz.data(), fx.data(), fy.data(), fz.data(), type_particule.data(), vertices.data(), shps, time, hkp, size);
        }
        else if(type == 1)
        {
        	kernel2<<<numBlocks, threadsPerBlock>>>(wrapper.interactions, rx.data(), ry.data(), rz.data(), vrotx.data(), vroty.data(), vrotz.data(), vx.data(), vy.data(), vz.data(), mass.data(), momx.data(), momy.data(), momz.data(), fx.data(), fy.data(), fz.data(), type_particule.data(), vertices.data(), shps, time, hkp, size);
        }
        else if(type == 2)
        {
        	//kernel3<<<numBlocks, threadsPerBlock>>>(wrapper.interactions, rx.data(), ry.data(), rz.data(), vrotx.data(), vroty.data(), vrotz.data(), vx.data(), vy.data(), vz.data(), mass.data(), momx.data(), momy.data(), momz.data(), fx.data(), fy.data(), fz.data(), type_particule.data(), vertices.data(), shps, time, hkp, size);
        }
        else if(type == 3)
        {
        	//kernel4<<<numBlocks, threadsPerBlock>>>(wrapper.interactions, rx.data(), ry.data(), rz.data(), vrotx.data(), vroty.data(), vrotz.data(), vx.data(), vy.data(), vz.data(), mass.data(), momx.data(), momy.data(), momz.data(), fx.data(), fy.data(), fz.data(), type_particule.data(), vertices.data(), shps, time, hkp, size);
        }*/
        
        printf("TYPE:%d\n", type);
        
        kernelTotal<<<numBlocks, threadsPerBlock>>>(wrapper.interactions, rx.data(), ry.data(), rz.data(), vrotx.data(), vroty.data(), vrotz.data(), vx.data(), vy.data(), vz.data(), mass.data(), momx.data(), momy.data(), momz.data(), fx.data(), fy.data(), fz.data(), type_particule.data(), vertices.data(), shps, time, hkp, size, type);
      
        run_contact_law(parallel_execution_context(), type, classifier, poly, __params__);
      }
      run_contact_law(parallel_execution_context(), 4, classifier, cyli, __params_driver__);
      run_contact_law(parallel_execution_context(), 5, classifier, surf, __params_driver__);
      run_contact_law(parallel_execution_context(), 6, classifier, ball, __params_driver__);
      for (int type = 7; type <= 12; type++)
      {
        run_contact_law(parallel_execution_context(), type, classifier, stlm, __params_driver__);
      }

#undef __params__
#undef __params_driver__

      if (write_interactions)
      {
        auto stream = itools::create_buffer(*grid, classifier);
        std::string ts = std::to_string(*timestep);
        itools::write_file(stream, (*dir_name), (*interaction_basename) + ts);
      }
    }
  };

  template <class GridT> using ComputeContactClassifierPolyGPUTmpl = ComputeContactClassifierPolyhedronGPU<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("contact_polyhedron", make_grid_variant_operator<ComputeContactClassifierPolyGPUTmpl>); }
} // namespace exaDEM
