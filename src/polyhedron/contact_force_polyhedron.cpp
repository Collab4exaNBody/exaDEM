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

  template <typename GridT> __global__ void kernelContact1(double* ft_x,
  								double* ft_y,
  								double* ft_z,
  								double* mom_x,
  								double* mom_y,
  								double* mom_z,
  								uint64_t* id_i,
  								uint64_t* id_j,
  								uint32_t* celli,
  								uint32_t* cellj,
  								uint16_t* p_i,
  								uint16_t* p_j,
  								uint16_t* sub_i,
  								uint16_t* sub_j, 
  								GridT* cells,
  								const shape *const shps,
  								const ContactParams hkp,
  								const double dt,
  								double* dnp,
  								Vec3d* cpp,
  								Vec3d* fnp,
  								Vec3d* ftp,		
  								size_t size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	if(idx < size)
  	{
        	auto &cell_i = cells[celli[idx]];
        	auto &cell_j = cells[cellj[idx]];

        	// === positions
        	const Vec3d ri = {cell_i[field::rx][p_i[idx]], cell_i[field::ry][p_i[idx]], cell_i[field::rz][p_i[idx]]};//get_r(cell_i, item.p_i);
        	const Vec3d rj = {cell_j[field::rx][p_j[idx]], cell_j[field::ry][p_j[idx]], cell_j[field::rz][p_j[idx]]};//get_r(cell_j, item.p_j);

        	// === vrot
        	const Vec3d &vrot_i = cell_i[field::vrot][p_i[idx]];
        	const Vec3d &vrot_j = cell_j[field::vrot][p_j[idx]];

        	// === type
        	const auto &type_i = cell_i[field::type][p_i[idx]];
        	const auto &type_j = cell_j[field::type][p_j[idx]];

        	// === vertex array
        	const auto &vertices_i = cell_i[field::vertices][p_i[idx]];
        	const auto &vertices_j = cell_j[field::vertices][p_j[idx]];

        	// === shapes
        	const shape &shp_i = shps[type_i];
        	const shape &shp_j = shps[type_j];
        	
        	auto [contact, dn, n, contact_position] = exaDEM::detection_vertex_vertex_precompute(vertices_i, sub_i[idx], &shp_i, vertices_j, sub_j[idx], &shp_j);
        	
        	Vec3d f = {0, 0, 0};
        	Vec3d fn = {0, 0, 0};
        	if (contact)
        	{
         	const Vec3d vi = {cell_i[field::vx][p_i[idx]], cell_i[field::vy][p_i[idx]], cell_i[field::vz][p_i[idx]]};//get_v(cell_i, item.p_i);
          	const Vec3d vj = {cell_j[field::vx][p_j[idx]], cell_j[field::vy][p_j[idx]], cell_j[field::vz][p_j[idx]]};//get_v(cell_j, item.p_j);
          	const auto &m_i = cell_i[field::mass][p_i[idx]];
          	const auto &m_j = cell_j[field::mass][p_j[idx]];

          	const double meff = compute_effective_mass(m_i, m_j);
          	
          	Vec3d item_friction = {ft_x[idx], ft_y[idx], ft_z[idx]};
          	Vec3d item_moment = {mom_x[idx], mom_y[idx], mom_z[idx]};
  
          	contact_force_core(dn, n, dt, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item_friction, contact_position, ri, vi, f, item_moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);

          	fn = f - item_friction;
          	
          	Vec3d f2 = {0,0,0};

          	// === update particle informations
          	// ==== Particle i
          	auto &mom_i = cell_i[field::mom][p_i[idx]];
          	
          	auto mom_res_i = compute_moments(contact_position, ri, f, item_moment);
          	
          	//atomicAdd(&cells[celli[idx]][field::mom][p_i[idx]].x, mom_res_i.x);
          	//atomicAdd(&cells[celli[idx]][field::mom][p_i[idx]].y, mom_res_i.y);
          	//atomicAdd(&cells[celli[idx]][field::mom][p_i[idx]].z, mom_res_i.z);
          	
          	//atomicAdd(&(cells[celli[idx]][field::fx][p_i[idx]]), f.x);
          	//atomicAdd(&(cells[celli[idx]][field::fy][p_i[idx]]), f.y);
          	//atomicAdd(&(cells[celli[idx]][field::fz][p_i[idx]]), f.z);

          	// ==== Particle j
          	auto &mom_j = cell_j[field::mom][p_j[idx]];
          	
          	auto mom_res_j = compute_moments(contact_position, rj, -f, -item_moment);
          	
          	//atomicAdd(&cells[cellj[idx]][field::mom][p_j[idx]].x, mom_res_j.x);
          	//atomicAdd(&cells[cellj[idx]][field::mom][p_j[idx]].y, mom_res_j.y);
          	//atomicAdd(&cells[cellj[idx]][field::mom][p_j[idx]].z, mom_res_j.z);
          	
          	//atomicAdd(&cells[cellj[idx]][field::fx][p_j[idx]], -f.x);
          	//atomicAdd(&cells[cellj[idx]][field::fy][p_j[idx]], -f.y);
          	//atomicAdd(&cells[cellj[idx]][field::fz][p_j[idx]], -f.z);
          	
          	ft_x[idx] = item_friction.x;
          	ft_y[idx] = item_friction.y;
          	ft_z[idx] = item_friction.z;
          	
          	mom_x[idx] = item_moment.x;
          	mom_y[idx] = item_moment.y;
          	mom_z[idx] = item_moment.z;

       		}
        	else
        	{
          	//item.reset();
          	
          	ft_x[idx] = 0;
          	ft_y[idx] = 0;
          	ft_z[idx] = 0;
          	
          	mom_x[idx] = 0;
          	mom_y[idx] = 0;
          	mom_z[idx] = 0;
          	
          	dn = 0;
          	
        	}        	

		dnp[idx] = dn;
		cpp[idx] = contact_position;
		fnp[idx] = fn;
		ftp[idx] = {ft_x[idx], ft_y[idx], ft_z[idx]};

  	}
  	
  }
  
  template <typename GridT> __global__ void kernelContact2(double* ft_x,
  								double* ft_y,
  								double* ft_z,
  								double* mom_x,
  								double* mom_y,
  								double* mom_z,
  								uint64_t* id_i,
  								uint64_t* id_j,
  								uint32_t* celli,
  								uint32_t* cellj,
  								uint16_t* p_i,
  								uint16_t* p_j,
  								uint16_t* sub_i,
  								uint16_t* sub_j, 
  								GridT* cells,
  								const shape *const shps,
  								const ContactParams hkp,
  								const double dt,
  								double* dnp,
  								Vec3d* cpp,
  								Vec3d* fnp,
  								Vec3d* ftp,		
  								size_t size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	if(idx < size)
  	{
        	auto &cell_i = cells[celli[idx]];
        	auto &cell_j = cells[cellj[idx]];

        	// === positions
        	const Vec3d ri = {cell_i[field::rx][p_i[idx]], cell_i[field::ry][p_i[idx]], cell_i[field::rz][p_i[idx]]};//get_r(cell_i, item.p_i);
        	const Vec3d rj = {cell_j[field::rx][p_j[idx]], cell_j[field::ry][p_j[idx]], cell_j[field::rz][p_j[idx]]};//get_r(cell_j, item.p_j);

        	// === vrot
        	const Vec3d &vrot_i = cell_i[field::vrot][p_i[idx]];
        	const Vec3d &vrot_j = cell_j[field::vrot][p_j[idx]];

        	// === type
        	const auto &type_i = cell_i[field::type][p_i[idx]];
        	const auto &type_j = cell_j[field::type][p_j[idx]];

        	// === vertex array
        	const auto &vertices_i = cell_i[field::vertices][p_i[idx]];
        	const auto &vertices_j = cell_j[field::vertices][p_j[idx]];

        	// === shapes
        	const shape &shp_i = shps[type_i];
        	const shape &shp_j = shps[type_j];
        	
        	auto [contact, dn, n, contact_position] = exaDEM::detection_vertex_edge_precompute(vertices_i, sub_i[idx], &shp_i, vertices_j, sub_j[idx], &shp_j);
        	
        	Vec3d f = {0, 0, 0};
        	Vec3d fn = {0, 0, 0};
        	if (contact)
        	{
         	const Vec3d vi = {cell_i[field::vx][p_i[idx]], cell_i[field::vy][p_i[idx]], cell_i[field::vz][p_i[idx]]};//get_v(cell_i, item.p_i);
          	const Vec3d vj = {cell_j[field::vx][p_j[idx]], cell_j[field::vy][p_j[idx]], cell_j[field::vz][p_j[idx]]};//get_v(cell_j, item.p_j);
          	const auto &m_i = cell_i[field::mass][p_i[idx]];
          	const auto &m_j = cell_j[field::mass][p_j[idx]];

          	const double meff = compute_effective_mass(m_i, m_j);
          	
          	Vec3d item_friction = {ft_x[idx], ft_y[idx], ft_z[idx]};
          	Vec3d item_moment = {mom_x[idx], mom_y[idx], mom_z[idx]};
  
          	contact_force_core(dn, n, dt, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item_friction, contact_position, ri, vi, f, item_moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);

          	fn = f - item_friction;
          	
          	Vec3d f2 = {0,0,0};

          	// === update particle informations
          	// ==== Particle i
          	auto &mom_i = cell_i[field::mom][p_i[idx]];
          	
          	auto mom_res_i = compute_moments(contact_position, ri, f, item_moment);
          	
          	//atomicAdd(&cells[celli[idx]][field::mom][p_i[idx]].x, mom_res_i.x);
          	//atomicAdd(&cells[celli[idx]][field::mom][p_i[idx]].y, mom_res_i.y);
          	//atomicAdd(&cells[celli[idx]][field::mom][p_i[idx]].z, mom_res_i.z);
          	
          	//atomicAdd(&(cells[celli[idx]][field::fx][p_i[idx]]), f.x);
          	//atomicAdd(&(cells[celli[idx]][field::fy][p_i[idx]]), f.y);
          	//atomicAdd(&(cells[celli[idx]][field::fz][p_i[idx]]), f.z);

          	// ==== Particle j
          	auto &mom_j = cell_j[field::mom][p_j[idx]];
          	
          	auto mom_res_j = compute_moments(contact_position, rj, -f, -item_moment);
          	
          	//atomicAdd(&cells[cellj[idx]][field::mom][p_j[idx]].x, mom_res_j.x);
          	//atomicAdd(&cells[cellj[idx]][field::mom][p_j[idx]].y, mom_res_j.y);
          	//atomicAdd(&cells[cellj[idx]][field::mom][p_j[idx]].z, mom_res_j.z);
          	
          	//atomicAdd(&cells[cellj[idx]][field::fx][p_j[idx]], -f.x);
          	//atomicAdd(&cells[cellj[idx]][field::fy][p_j[idx]], -f.y);
          	//atomicAdd(&cells[cellj[idx]][field::fz][p_j[idx]], -f.z);
          	
          	ft_x[idx] = item_friction.x;
          	ft_y[idx] = item_friction.y;
          	ft_z[idx] = item_friction.z;
          	
          	mom_x[idx] = item_moment.x;
          	mom_y[idx] = item_moment.y;
          	mom_z[idx] = item_moment.z;

       		}
        	else
        	{
          	//item.reset();
          	
          	ft_x[idx] = 0;
          	ft_y[idx] = 0;
          	ft_z[idx] = 0;
          	
          	mom_x[idx] = 0;
          	mom_y[idx] = 0;
          	mom_z[idx] = 0;
          	
          	dn = 0;
          	
        	}        	

		dnp[idx] = dn;
		cpp[idx] = contact_position;
		fnp[idx] = fn;
		ftp[idx] = {ft_x[idx], ft_y[idx], ft_z[idx]};

  	}
  	
  }
  
  template <typename GridT> __global__ void kernelContact3(double* ft_x,
  								double* ft_y,
  								double* ft_z,
  								double* mom_x,
  								double* mom_y,
  								double* mom_z,
  								uint64_t* id_i,
  								uint64_t* id_j,
  								uint32_t* celli,
  								uint32_t* cellj,
  								uint16_t* p_i,
  								uint16_t* p_j,
  								uint16_t* sub_i,
  								uint16_t* sub_j, 
  								GridT* cells,
  								const shape *const shps,
  								const ContactParams hkp,
  								const double dt,
  								double* dnp,
  								Vec3d* cpp,
  								Vec3d* fnp,
  								Vec3d* ftp,		
  								size_t size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	if(idx < size)
  	{
        	auto &cell_i = cells[celli[idx]];
        	auto &cell_j = cells[cellj[idx]];

        	// === positions
        	const Vec3d ri = {cell_i[field::rx][p_i[idx]], cell_i[field::ry][p_i[idx]], cell_i[field::rz][p_i[idx]]};//get_r(cell_i, item.p_i);
        	const Vec3d rj = {cell_j[field::rx][p_j[idx]], cell_j[field::ry][p_j[idx]], cell_j[field::rz][p_j[idx]]};//get_r(cell_j, item.p_j);

        	// === vrot
        	const Vec3d &vrot_i = cell_i[field::vrot][p_i[idx]];
        	const Vec3d &vrot_j = cell_j[field::vrot][p_j[idx]];

        	// === type
        	const auto &type_i = cell_i[field::type][p_i[idx]];
        	const auto &type_j = cell_j[field::type][p_j[idx]];

        	// === vertex array
        	const auto &vertices_i = cell_i[field::vertices][p_i[idx]];
        	const auto &vertices_j = cell_j[field::vertices][p_j[idx]];

        	// === shapes
        	const shape &shp_i = shps[type_i];
        	const shape &shp_j = shps[type_j];
        	
        	auto [contact, dn, n, contact_position] = exaDEM::detection_vertex_face_precompute(vertices_i, sub_i[idx], &shp_i, vertices_j, sub_j[idx], &shp_j);
        	
        	Vec3d f = {0, 0, 0};
        	Vec3d fn = {0, 0, 0};
        	if (contact)
        	{
         	const Vec3d vi = {cell_i[field::vx][p_i[idx]], cell_i[field::vy][p_i[idx]], cell_i[field::vz][p_i[idx]]};//get_v(cell_i, item.p_i);
          	const Vec3d vj = {cell_j[field::vx][p_j[idx]], cell_j[field::vy][p_j[idx]], cell_j[field::vz][p_j[idx]]};//get_v(cell_j, item.p_j);
          	const auto &m_i = cell_i[field::mass][p_i[idx]];
          	const auto &m_j = cell_j[field::mass][p_j[idx]];

          	const double meff = compute_effective_mass(m_i, m_j);
          	
          	Vec3d item_friction = {ft_x[idx], ft_y[idx], ft_z[idx]};
          	Vec3d item_moment = {mom_x[idx], mom_y[idx], mom_z[idx]};
  
          	contact_force_core(dn, n, dt, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item_friction, contact_position, ri, vi, f, item_moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);

          	fn = f - item_friction;
          	
          	Vec3d f2 = {0,0,0};

          	// === update particle informations
          	// ==== Particle i
          	auto &mom_i = cell_i[field::mom][p_i[idx]];
          	
          	auto mom_res_i = compute_moments(contact_position, ri, f, item_moment);
          	
          	//atomicAdd(&cells[celli[idx]][field::mom][p_i[idx]].x, mom_res_i.x);
          	//atomicAdd(&cells[celli[idx]][field::mom][p_i[idx]].y, mom_res_i.y);
          	//atomicAdd(&cells[celli[idx]][field::mom][p_i[idx]].z, mom_res_i.z);
          	
          	//atomicAdd(&(cells[celli[idx]][field::fx][p_i[idx]]), f.x);
          	//atomicAdd(&(cells[celli[idx]][field::fy][p_i[idx]]), f.y);
          	//atomicAdd(&(cells[celli[idx]][field::fz][p_i[idx]]), f.z);

          	// ==== Particle j
          	auto &mom_j = cell_j[field::mom][p_j[idx]];
          	
          	auto mom_res_j = compute_moments(contact_position, rj, -f, -item_moment);
          	
          	//atomicAdd(&cells[cellj[idx]][field::mom][p_j[idx]].x, mom_res_j.x);
          	//atomicAdd(&cells[cellj[idx]][field::mom][p_j[idx]].y, mom_res_j.y);
          	//atomicAdd(&cells[cellj[idx]][field::mom][p_j[idx]].z, mom_res_j.z);
          	
          	//atomicAdd(&cells[cellj[idx]][field::fx][p_j[idx]], -f.x);
          	//atomicAdd(&cells[cellj[idx]][field::fy][p_j[idx]], -f.y);
          	//atomicAdd(&cells[cellj[idx]][field::fz][p_j[idx]], -f.z);
          	
          	ft_x[idx] = item_friction.x;
          	ft_y[idx] = item_friction.y;
          	ft_z[idx] = item_friction.z;
          	
          	mom_x[idx] = item_moment.x;
          	mom_y[idx] = item_moment.y;
          	mom_z[idx] = item_moment.z;

       		}
        	else
        	{
          	//item.reset();
          	
          	ft_x[idx] = 0;
          	ft_y[idx] = 0;
          	ft_z[idx] = 0;
          	
          	mom_x[idx] = 0;
          	mom_y[idx] = 0;
          	mom_z[idx] = 0;
          	
          	dn = 0;
          	
        	}        	

		dnp[idx] = dn;
		cpp[idx] = contact_position;
		fnp[idx] = fn;
		ftp[idx] = {ft_x[idx], ft_y[idx], ft_z[idx]};

  	}
  	
  }
  
  template <typename GridT> __global__ void kernelContact4(double* ft_x,
  								double* ft_y,
  								double* ft_z,
  								double* mom_x,
  								double* mom_y,
  								double* mom_z,
  								uint64_t* id_i,
  								uint64_t* id_j,
  								uint32_t* celli,
  								uint32_t* cellj,
  								uint16_t* p_i,
  								uint16_t* p_j,
  								uint16_t* sub_i,
  								uint16_t* sub_j, 
  								GridT* cells,
  								const shape *const shps,
  								const ContactParams hkp,
  								const double dt,
  								double* dnp,
  								Vec3d* cpp,
  								Vec3d* fnp,
  								Vec3d* ftp,		
  								size_t size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	if(idx < size)
  	{
        	auto &cell_i = cells[celli[idx]];
        	auto &cell_j = cells[cellj[idx]];

        	// === positions
        	const Vec3d ri = {cell_i[field::rx][p_i[idx]], cell_i[field::ry][p_i[idx]], cell_i[field::rz][p_i[idx]]};//get_r(cell_i, item.p_i);
        	const Vec3d rj = {cell_j[field::rx][p_j[idx]], cell_j[field::ry][p_j[idx]], cell_j[field::rz][p_j[idx]]};//get_r(cell_j, item.p_j);

        	// === vrot
        	const Vec3d &vrot_i = cell_i[field::vrot][p_i[idx]];
        	const Vec3d &vrot_j = cell_j[field::vrot][p_j[idx]];

        	// === type
        	const auto &type_i = cell_i[field::type][p_i[idx]];
        	const auto &type_j = cell_j[field::type][p_j[idx]];

        	// === vertex array
        	const auto &vertices_i = cell_i[field::vertices][p_i[idx]];
        	const auto &vertices_j = cell_j[field::vertices][p_j[idx]];

        	// === shapes
        	const shape &shp_i = shps[type_i];
        	const shape &shp_j = shps[type_j];
        	
        	auto [contact, dn, n, contact_position] = exaDEM::detection_edge_edge_precompute(vertices_i, sub_i[idx], &shp_i, vertices_j, sub_j[idx], &shp_j);
        	
        	Vec3d f = {0, 0, 0};
        	Vec3d fn = {0, 0, 0};
        	if (contact)
        	{
         	const Vec3d vi = {cell_i[field::vx][p_i[idx]], cell_i[field::vy][p_i[idx]], cell_i[field::vz][p_i[idx]]};//get_v(cell_i, item.p_i);
          	const Vec3d vj = {cell_j[field::vx][p_j[idx]], cell_j[field::vy][p_j[idx]], cell_j[field::vz][p_j[idx]]};//get_v(cell_j, item.p_j);
          	const auto &m_i = cell_i[field::mass][p_i[idx]];
          	const auto &m_j = cell_j[field::mass][p_j[idx]];

          	const double meff = compute_effective_mass(m_i, m_j);
          	
          	Vec3d item_friction = {ft_x[idx], ft_y[idx], ft_z[idx]};
          	Vec3d item_moment = {mom_x[idx], mom_y[idx], mom_z[idx]};
  
          	contact_force_core(dn, n, dt, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item_friction, contact_position, ri, vi, f, item_moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);

          	fn = f - item_friction;
          	
          	Vec3d f2 = {0,0,0};

          	// === update particle informations
          	// ==== Particle i
          	auto &mom_i = cell_i[field::mom][p_i[idx]];
          	
          	auto mom_res_i = compute_moments(contact_position, ri, f, item_moment);
          	
          	//atomicAdd(&cells[celli[idx]][field::mom][p_i[idx]].x, mom_res_i.x);
          	//atomicAdd(&cells[celli[idx]][field::mom][p_i[idx]].y, mom_res_i.y);
          	//atomicAdd(&cells[celli[idx]][field::mom][p_i[idx]].z, mom_res_i.z);
          	
          	//atomicAdd(&(cells[celli[idx]][field::fx][p_i[idx]]), f.x);
          	//atomicAdd(&(cells[celli[idx]][field::fy][p_i[idx]]), f.y);
          	//atomicAdd(&(cells[celli[idx]][field::fz][p_i[idx]]), f.z);

          	// ==== Particle j
          	auto &mom_j = cell_j[field::mom][p_j[idx]];
          	
          	auto mom_res_j = compute_moments(contact_position, rj, -f, -item_moment);
          	
          	//atomicAdd(&cells[cellj[idx]][field::mom][p_j[idx]].x, mom_res_j.x);
          	//atomicAdd(&cells[cellj[idx]][field::mom][p_j[idx]].y, mom_res_j.y);
          	//atomicAdd(&cells[cellj[idx]][field::mom][p_j[idx]].z, mom_res_j.z);
          	
          	//atomicAdd(&cells[cellj[idx]][field::fx][p_j[idx]], -f.x);
          	//atomicAdd(&cells[cellj[idx]][field::fy][p_j[idx]], -f.y);
          	//atomicAdd(&cells[cellj[idx]][field::fz][p_j[idx]], -f.z);
          	
          	ft_x[idx] = item_friction.x;
          	ft_y[idx] = item_friction.y;
          	ft_z[idx] = item_friction.z;
          	
          	mom_x[idx] = item_moment.x;
          	mom_y[idx] = item_moment.y;
          	mom_z[idx] = item_moment.z;

       		}
        	else
        	{
          	//item.reset();
          	
          	ft_x[idx] = 0;
          	ft_y[idx] = 0;
          	ft_z[idx] = 0;
          	
          	mom_x[idx] = 0;
          	mom_y[idx] = 0;
          	mom_z[idx] = 0;
          	
          	dn = 0;
          	
        	}        	

		dnp[idx] = dn;
		cpp[idx] = contact_position;
		fnp[idx] = fn;
		ftp[idx] = {ft_x[idx], ft_y[idx], ft_z[idx]};

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
    ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
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
      store_interactions = true;

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

      int blockSize = 256;

      auto [data1, size1] = classifier.get_info(0);
      InteractionWrapper<InteractionSOA> interactions1(data1);
      AnalysisDataPacker packer1(classifier, 0);
      int numBlocks1 = (size1 + blockSize - 1) / blockSize;

      auto [data2, size2] = classifier.get_info(1);
      InteractionWrapper<InteractionSOA> interactions2(data2);
      AnalysisDataPacker packer2(classifier, 1);
      int numBlocks2 = (size2 + blockSize - 1) / blockSize;
      
      auto [data3, size3] = classifier.get_info(2);
      InteractionWrapper<InteractionSOA> interactions3(data3);
      AnalysisDataPacker packer3(classifier, 2);
      int numBlocks3 = (size3 + blockSize - 1) / blockSize;
      
      auto [data4, size4] = classifier.get_info(3);
      InteractionWrapper<InteractionSOA> interactions4(data4);
      AnalysisDataPacker packer4(classifier, 3);
      int numBlocks4 = (size4 + blockSize - 1) / blockSize;
      
/*      cudaStream_t stream1, stream2, stream3, stream4;
      
      cudaStreamCreate(&stream1);
      cudaStreamCreate(&stream2);
      cudaStreamCreate(&stream3);
      cudaStreamCreate(&stream4);*/
      
      kernelContact1<<<numBlocks1, blockSize>>>(interactions1.ft_x, interactions1.ft_y, interactions1.ft_z, interactions1.mom_x, interactions1.mom_y, interactions1.mom_z, interactions1.id_i, interactions1.id_j, interactions1.cell_i, interactions1.cell_j, interactions1.p_i, interactions1.p_j, interactions1.sub_i, interactions1.sub_j, cells, shps, hkp, time, packer1.dnp, packer1.cpp, packer1.fnp, packer1.ftp, size1);
      kernelContact2<<<numBlocks2, blockSize>>>(interactions2.ft_x, interactions2.ft_y, interactions2.ft_z, interactions2.mom_x, interactions2.mom_y, interactions2.mom_z, interactions2.id_i, interactions2.id_j, interactions2.cell_i, interactions2.cell_j, interactions2.p_i, interactions2.p_j, interactions2.sub_i, interactions2.sub_j, cells, shps, hkp, time, packer2.dnp, packer2.cpp, packer2.fnp, packer2.ftp, size2);
      kernelContact3<<<numBlocks3, blockSize>>>(interactions3.ft_x, interactions3.ft_y, interactions3.ft_z, interactions3.mom_x, interactions3.mom_y, interactions3.mom_z, interactions3.id_i, interactions3.id_j, interactions3.cell_i, interactions3.cell_j, interactions3.p_i, interactions3.p_j, interactions3.sub_i, interactions3.sub_j, cells, shps, hkp, time, packer3.dnp, packer3.cpp, packer3.fnp, packer3.ftp, size3);
      kernelContact4<<<numBlocks4, blockSize>>>(interactions4.ft_x, interactions4.ft_y, interactions4.ft_z, interactions4.mom_x, interactions4.mom_y, interactions4.mom_z, interactions4.id_i, interactions4.id_j, interactions4.cell_i, interactions4.cell_j, interactions4.p_i, interactions4.p_j, interactions4.sub_i, interactions4.sub_j, cells, shps, hkp, time, packer4.dnp, packer4.cpp, packer4.fnp, packer4.ftp, size4);
      
/*      cudaStreamSynchronize(stream1);
      cudaStreamSynchronize(stream2);
      cudaStreamSynchronize(stream3);
      cudaStreamSynchronize(stream4);
      
      cudaStreamDestroy(stream1);
      cudaStreamDestroy(stream2);
      cudaStreamDestroy(stream3);
      cudaStreamDestroy(stream4);*/

      for (size_t type = 0; type <= 3; type++)
      {
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
