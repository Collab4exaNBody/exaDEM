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
  
  
  
  __device__ double compute_damp_gpu(const double a_damp_rate, const double a_kn, const double a_meff)
  {
    const double ret = a_damp_rate * 2.0 * sqrt(a_kn * a_meff);
    return ret;
  }

  __device__ Vec3d compute_relative_velocity_gpu(const Vec3d &contact_position, const Vec3d &pos_i, const Vec3d &vel_i, const Vec3d &vrot_i, const Vec3d &pos_j, const Vec3d &vel_j, const Vec3d &vrot_j)
  {
    const auto contribution_i = vel_i - exanb::cross(contact_position - pos_i, vrot_i);
    const auto contribution_j = vel_j - exanb::cross(contact_position - pos_j, vrot_j);
    const auto ret = contribution_j - contribution_i;
    return ret;
  }
  
 __device__ Vec3d compute_normal_force_gpu(const double a_kn, const double a_damp, const double a_dn, const double a_vn, const Vec3d &a_n)
  {
    const double fne = -a_kn * a_dn;  // elastic contact
    const double fnv = a_damp * a_vn; // viscous damping
    const double fn = fnv + fne;
    const auto ret = fn * a_n;
    return ret;
  }
  
  __device__ Vec3d compute_tangential_force_gpu(const double a_kt, const double a_dt, const double a_vn, const Vec3d &a_n, const Vec3d &a_vel)
  {
    const Vec3d vt = a_vel - a_vn * a_n;
    const Vec3d ft = a_kt * (a_dt * vt);
    return ft;
  }
  
  __device__ double compute_threshold_ft_gpu(const double a_mu, const double a_kn, const double a_dn)
  {
    // === recompute fne
    const double fne = a_kn * a_dn; //(remove -)
    const double threshold_ft = std::fabs(a_mu * fne);
    return threshold_ft;
  }

  __device__ void fit_tangential_force_gpu(const double threshold_ft, Vec3d &a_ft)
  {
    double ft_square = exanb::dot(a_ft, a_ft);
    if (ft_square > 0.0 && ft_square > threshold_ft * threshold_ft)
      a_ft *= (threshold_ft / sqrt(ft_square));
  }

  
  __device__ void contact_force_core_gpu(const double dn,
                                                        const Vec3d &n, // -normal
                                                        const double dt, const double kn, const double kt, const double kr, const double mu, const double dampRate, const double meff,
                                                        Vec3d &ft, // tangential force between particle i and j
                                                        const Vec3d &contact_position,
                                                        const Vec3d &pos_i,  // positions i
                                                        const Vec3d &vel_i,  // positions i
                                                        Vec3d f_i,          // forces i
                                                        Vec3d &mom_i,        // moments i
                                                        const Vec3d &vrot_i, // angular velocities i
                                                        const Vec3d &pos_j,  // positions j
                                                        const Vec3d &vel_j,  // positions j
                                                        const Vec3d &vrot_j  // angular velocities j
  )
  {
    const double damp = compute_damp_gpu(dampRate, kn, meff);

    // === Relative velocity (j relative to i)
    auto vel = compute_relative_velocity_gpu(contact_position, pos_i, vel_i, vrot_i, pos_j, vel_j, vrot_j);

    // compute relative velocity
    const double vn = exanb::dot(vel, n);

    // === Normal force (elatic contact + viscous damping)
    const Vec3d fn = compute_normal_force_gpu(kn, damp, dn, vn, n); // fc ==> cohesive force

    // === Tangential force (friction)
    ft += compute_tangential_force_gpu(kt, dt, vn, n, vel);
    // ft	 	+= exaDEM::compute_tangential_force(kt, dt, vn, n, vel);

    // fit tangential force
    auto threshold_ft = compute_threshold_ft_gpu(mu, kn, dn);
    fit_tangential_force_gpu(threshold_ft, ft);

    // === sum forces
    f_i = fn + ft;
    
    //printf("FORCE(%f,%f,%f)\n", fn.x,fn.y,fn.z);

    // === update moments
    mom_i += kr * (vrot_j - vrot_i) * dt;

    ///*
    // test
    Vec3d branch = contact_position - pos_i;
    double r = (exanb::dot(branch, vrot_i)) / (exanb::dot(vrot_i, vrot_i));
    branch -= r * vrot_i;

    constexpr double mur = 0;
    double threshold_mom = std::abs(mur * exanb::norm(branch) * exanb::norm(fn)); // even without fabs, the value should
                                                                                  // be positive
    double mom_square = exanb::dot(mom_i, mom_i);
    if (mom_square > 0.0 && mom_square > threshold_mom * threshold_mom)
      mom_i = mom_i * (threshold_mom / sqrt(mom_square));
    //*/
  }  
  

  template <typename GridT> __global__ void kernelContact1(InteractionWrapper<InteractionSOA> soa, 
  								GridT* cells,
  								const shape *const shps,
  								const ContactParams &hkp,
  								const double dt,
  								size_t size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	if(idx < size)
  	{
  		Interaction item = soa(idx);
  		
        	auto &cell_i = cells[item.cell_i];
        	auto &cell_j = cells[item.cell_j];

        	// === positions
        	const Vec3d ri = {cell_i[field::rx][item.p_i], cell_i[field::ry][item.p_i], cell_i[field::rz][item.p_i]};//get_r(cell_i, item.p_i);
        	const Vec3d rj = {cell_j[field::rx][item.p_j], cell_j[field::ry][item.p_j], cell_j[field::rz][item.p_j]};//get_r(cell_j, item.p_j);

        	// === vrot
        	const Vec3d &vrot_i = cell_i[field::vrot][item.p_i];
        	const Vec3d &vrot_j = cell_j[field::vrot][item.p_j];

        	// === type
        	const auto &type_i = cell_i[field::type][item.p_i];
        	const auto &type_j = cell_j[field::type][item.p_j];

        	// === vertex array
        	const auto &vertices_i = cell_i[field::vertices][item.p_i];
        	const auto &vertices_j = cell_j[field::vertices][item.p_j];

        	// === shapes
        	const shape &shp_i = shps[type_i];
        	const shape &shp_j = shps[type_j];
        	
        	auto [contact, dn, n, contact_position] = exaDEM::detection_vertex_vertex_precompute(vertices_i, item.sub_i, &shp_i, vertices_j, item.sub_j, &shp_j);
        	
        	Vec3d f = {0, 0, 0};
        	Vec3d fn = {0, 0, 0};
        	if (contact)
        	{
         	const Vec3d vi = {cell_i[field::vx][item.p_i], cell_i[field::vy][item.p_i], cell_i[field::vz][item.p_i]};//get_v(cell_i, item.p_i);
          	const Vec3d vj = {cell_j[field::vx][item.p_j], cell_j[field::vy][item.p_j], cell_j[field::vz][item.p_j]};//get_v(cell_j, item.p_j);
          	const auto &m_i = cell_i[field::mass][item.p_i];
          	const auto &m_j = cell_j[field::mass][item.p_j];

          	const double meff = compute_effective_mass(m_i, m_j);
		
          	contact_force_core_gpu(dn, n, dt, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item.friction, contact_position, ri, vi, f, item.moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);
		
          	fn = f - item.friction;
          	
          	Vec3d f2 = {0,0,0};

          	// === update particle informations
          	// ==== Particle i
          	auto &mom_i = cell_i[field::mom][item.p_i];
          	
          	auto mom_res_i = compute_moments(contact_position, ri, f, item.moment);
     
          	
          	atomicAdd(&cells[item.cell_i][field::mom][item.p_i].x, 0);//mom_res_i.x);
          	atomicAdd(&cells[item.cell_i][field::mom][item.p_i].y, 0);//mom_res_i.y);
          	atomicAdd(&cells[item.cell_i][field::mom][item.p_i].z, 0);//mom_res_i.z);
          	
          	atomicAdd(&(cells[item.cell_i][field::fx][item.p_i]), 0);//a);
          	atomicAdd(&(cells[item.cell_i][field::fy][item.p_i]), 0);//f.y);
          	atomicAdd(&(cells[item.cell_i][field::fz][item.p_i]), 0);//f.z);

          	// ==== Particle j
          	auto &mom_j = cell_j[field::mom][item.p_j];
          	
          	auto mom_res_j = compute_moments(contact_position, rj, -f, -item.moment);
          	
          	atomicAdd(&cells[item.cell_j][field::mom][item.p_j].x, 0);//mom_res_j.x);
          	atomicAdd(&cells[item.cell_j][field::mom][item.p_j].y, 0);//mom_res_j.y);
          	atomicAdd(&cells[item.cell_j][field::mom][item.p_j].z, 0);//mom_res_j.z);
          	
          	atomicAdd(&cells[item.cell_j][field::fx][item.p_j], 0);//-f.x);
          	atomicAdd(&cells[item.cell_j][field::fy][item.p_j], 0);//-f.y);
          	atomicAdd(&cells[item.cell_j][field::fz][item.p_j], 0);//-f.z);

       		}
        	else
        	{
          	//item.reset();
          	soa.update(idx, item);
          	dn = 0;
          	
        	}        	
  		
  	}
  	
  }
  
    template <typename GridT> __global__ void kernelContact2(InteractionWrapper<InteractionSOA> soa, 
  								GridT* cells,
  								const shape *const shps,
  								const ContactParams &hkp,
  								const double dt,
  								size_t size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	if(idx < size)
  	{
  		Interaction item = soa(idx);
  		
        	auto &cell_i = cells[item.cell_i];
        	auto &cell_j = cells[item.cell_j];

        	// === positions
        	const Vec3d ri = {cell_i[field::rx][item.p_i], cell_i[field::ry][item.p_i], cell_i[field::rz][item.p_i]};//get_r(cell_i, item.p_i);
        	const Vec3d rj = {cell_j[field::rx][item.p_j], cell_j[field::ry][item.p_j], cell_j[field::rz][item.p_j]};//get_r(cell_j, item.p_j);

        	// === vrot
        	const Vec3d &vrot_i = cell_i[field::vrot][item.p_i];
        	const Vec3d &vrot_j = cell_j[field::vrot][item.p_j];

        	// === type
        	const auto &type_i = cell_i[field::type][item.p_i];
        	const auto &type_j = cell_j[field::type][item.p_j];

        	// === vertex array
        	const auto &vertices_i = cell_i[field::vertices][item.p_i];
        	const auto &vertices_j = cell_j[field::vertices][item.p_j];

        	// === shapes
        	const shape &shp_i = shps[type_i];
        	const shape &shp_j = shps[type_j];
        	
        	auto [contact, dn, n, contact_position] = exaDEM::detection_vertex_edge_precompute(vertices_i, item.sub_i, &shp_i, vertices_j, item.sub_j, &shp_j);
        	
        	Vec3d f = {0, 0, 0};
        	Vec3d fn = {0, 0, 0};
        	if (contact)
        	{
         	const Vec3d vi = {cell_i[field::vx][item.p_i], cell_i[field::vy][item.p_i], cell_i[field::vz][item.p_i]};//get_v(cell_i, item.p_i);
          	const Vec3d vj = {cell_j[field::vx][item.p_j], cell_j[field::vy][item.p_j], cell_j[field::vz][item.p_j]};//get_v(cell_j, item.p_j);
          	const auto &m_i = cell_i[field::mass][item.p_i];
          	const auto &m_j = cell_j[field::mass][item.p_j];

          	const double meff = compute_effective_mass(m_i, m_j);
		
          	contact_force_core_gpu(dn, n, dt, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item.friction, contact_position, ri, vi, f, item.moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);
		
          	fn = f - item.friction;

          	// === update particle informations
          	// ==== Particle i
          	auto &mom_i = cell_i[field::mom][item.p_i];
          	
          	auto mom_res_i = compute_moments(contact_position, ri, f, item.moment);
          	
          	atomicAdd(&cells[item.cell_i][field::mom][item.p_i].x, 0);//mom_res_i.x);
          	atomicAdd(&cells[item.cell_i][field::mom][item.p_i].y, 0);//mom_res_i.y);
          	atomicAdd(&cells[item.cell_i][field::mom][item.p_i].z, 0);//mom_res_i.z);
          	
          	atomicAdd(&cells[item.cell_i][field::fx][item.p_i], 0);//f.x);
          	atomicAdd(&cells[item.cell_i][field::fy][item.p_i], 0);//f.y);
          	atomicAdd(&cells[item.cell_i][field::fz][item.p_i], 0);//f.z);

          	// ==== Particle j
          	auto &mom_j = cell_j[field::mom][item.p_j];
          	
          	auto mom_res_j = compute_moments(contact_position, rj, -f, -item.moment);
          	
          	atomicAdd(&cells[item.cell_j][field::mom][item.p_j].x, 0);//mom_res_j.x);
          	atomicAdd(&cells[item.cell_j][field::mom][item.p_j].y, 0);//mom_res_j.y);
          	atomicAdd(&cells[item.cell_j][field::mom][item.p_j].z, 0);//mom_res_j.z);
          	
          	atomicAdd(&cells[item.cell_j][field::fx][item.p_j], 0);//-f.x);
          	atomicAdd(&cells[item.cell_j][field::fy][item.p_j], 0);//-f.y);
          	atomicAdd(&cells[item.cell_j][field::fz][item.p_j], 0);//-f.z);
       		}
        	else
        	{
          	//item.reset();
          	soa.update(idx, item);
          	dn = 0;
        	}     	
  		
  		}
  	
  }
  
  template <typename GridT> __global__ void kernelContact3(InteractionWrapper<InteractionSOA> soa, 
  								GridT* cells,
  								const shape *const shps,
  								const ContactParams &hkp,
  								const double dt,
  								size_t size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	if(idx < size)
  	{
  		Interaction item = soa(idx);
  		
        	auto &cell_i = cells[item.cell_i];
        	auto &cell_j = cells[item.cell_j];

        	// === positions
        	const Vec3d ri = {cell_i[field::rx][item.p_i], cell_i[field::ry][item.p_i], cell_i[field::rz][item.p_i]};//get_r(cell_i, item.p_i);
        	const Vec3d rj = {cell_j[field::rx][item.p_j], cell_j[field::ry][item.p_j], cell_j[field::rz][item.p_j]};//get_r(cell_j, item.p_j);

        	// === vrot
        	const Vec3d &vrot_i = cell_i[field::vrot][item.p_i];
        	const Vec3d &vrot_j = cell_j[field::vrot][item.p_j];

        	// === type
        	const auto &type_i = cell_i[field::type][item.p_i];
        	const auto &type_j = cell_j[field::type][item.p_j];

        	// === vertex array
        	const auto &vertices_i = cell_i[field::vertices][item.p_i];
        	const auto &vertices_j = cell_j[field::vertices][item.p_j];

        	// === shapes
        	const shape &shp_i = shps[type_i];
        	const shape &shp_j = shps[type_j];
        	
        	auto [contact, dn, n, contact_position] = exaDEM::detection_vertex_face_precompute(vertices_i, item.sub_i, &shp_i, vertices_j, item.sub_j, &shp_j);
        	
        	Vec3d f = {0, 0, 0};
        	Vec3d fn = {0, 0, 0};
        	if (contact)
        	{
         	const Vec3d vi = {cell_i[field::vx][item.p_i], cell_i[field::vy][item.p_i], cell_i[field::vz][item.p_i]};//get_v(cell_i, item.p_i);
          	const Vec3d vj = {cell_j[field::vx][item.p_j], cell_j[field::vy][item.p_j], cell_j[field::vz][item.p_j]};//get_v(cell_j, item.p_j);
          	const auto &m_i = cell_i[field::mass][item.p_i];
          	const auto &m_j = cell_j[field::mass][item.p_j];

          	const double meff = compute_effective_mass(m_i, m_j);
		
          	contact_force_core_gpu(dn, n, dt, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item.friction, contact_position, ri, vi, f, item.moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);
		
          	fn = f - item.friction;

          	// === update particle informations
          	// ==== Particle i
          	auto &mom_i = cell_i[field::mom][item.p_i];
          	
          	auto mom_res_i = compute_moments(contact_position, ri, f, item.moment);
          	
          	atomicAdd(&cells[item.cell_i][field::mom][item.p_i].x, 0);//mom_res_i.x);
          	atomicAdd(&cells[item.cell_i][field::mom][item.p_i].y, 0);//mom_res_i.y);
          	atomicAdd(&cells[item.cell_i][field::mom][item.p_i].z, 0);//mom_res_i.z);
          	
          	atomicAdd(&cells[item.cell_i][field::fx][item.p_i], 0);//f.x);
          	atomicAdd(&cells[item.cell_i][field::fy][item.p_i], 0);//f.y);
          	atomicAdd(&cells[item.cell_i][field::fz][item.p_i], 0);//f.z);

          	// ==== Particle j
          	auto &mom_j = cell_j[field::mom][item.p_j];
          	
          	auto mom_res_j = compute_moments(contact_position, rj, -f, -item.moment);
          	
          	atomicAdd(&cells[item.cell_j][field::mom][item.p_j].x, 0);//mom_res_j.x);
          	atomicAdd(&cells[item.cell_j][field::mom][item.p_j].y, 0);//mom_res_j.y);
          	atomicAdd(&cells[item.cell_j][field::mom][item.p_j].z, 0);//mom_res_j.z);
          	
          	atomicAdd(&cells[item.cell_j][field::fx][item.p_j], 0);//-f.x);
          	atomicAdd(&cells[item.cell_j][field::fy][item.p_j], 0);//-f.y);
          	atomicAdd(&cells[item.cell_j][field::fz][item.p_j], 0);//-f.z);
       		}
        	else
        	{
          	//item.reset();
          	soa.update(idx, item);
          	dn = 0;
        	}
  		
  		}
  	
  }
  
  template <typename GridT> __global__ void kernelContact4(InteractionWrapper<InteractionSOA> soa, 
  								GridT* cells,
  								const shape *const shps,
  								const ContactParams &hkp,
  								const double dt,
  								size_t size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	if(idx < size)
  	{
  		Interaction item = soa(idx);
  		
        	auto &cell_i = cells[item.cell_i];
        	auto &cell_j = cells[item.cell_j];

        	// === positions
        	const Vec3d ri = {cell_i[field::rx][item.p_i], cell_i[field::ry][item.p_i], cell_i[field::rz][item.p_i]};//get_r(cell_i, item.p_i);
        	const Vec3d rj = {cell_j[field::rx][item.p_j], cell_j[field::ry][item.p_j], cell_j[field::rz][item.p_j]};//get_r(cell_j, item.p_j);

        	// === vrot
        	const Vec3d &vrot_i = cell_i[field::vrot][item.p_i];
        	const Vec3d &vrot_j = cell_j[field::vrot][item.p_j];

        	// === type
        	const auto &type_i = cell_i[field::type][item.p_i];
        	const auto &type_j = cell_j[field::type][item.p_j];

        	// === vertex array
        	const auto &vertices_i = cell_i[field::vertices][item.p_i];
        	const auto &vertices_j = cell_j[field::vertices][item.p_j];

        	// === shapes
        	const shape &shp_i = shps[type_i];
        	const shape &shp_j = shps[type_j];
        	
        	auto [contact, dn, n, contact_position] = exaDEM::detection_edge_edge_precompute(vertices_i, item.sub_i, &shp_i, vertices_j, item.sub_j, &shp_j);
        	
        	Vec3d f = {0, 0, 0};
        	Vec3d fn = {0, 0, 0};
        	if (contact)
        	{
         	const Vec3d vi = {cell_i[field::vx][item.p_i], cell_i[field::vy][item.p_i], cell_i[field::vz][item.p_i]};//get_v(cell_i, item.p_i);
          	const Vec3d vj = {cell_j[field::vx][item.p_j], cell_j[field::vy][item.p_j], cell_j[field::vz][item.p_j]};//get_v(cell_j, item.p_j);
          	const auto &m_i = cell_i[field::mass][item.p_i];
          	const auto &m_j = cell_j[field::mass][item.p_j];

          	const double meff = compute_effective_mass(m_i, m_j);
		
          	contact_force_core_gpu(dn, n, dt, hkp.m_kn, hkp.m_kt, hkp.m_kr, hkp.m_mu, hkp.m_damp_rate, meff, item.friction, contact_position, ri, vi, f, item.moment, vrot_i, // particle 1
                             rj, vj, vrot_j                                                                                                                             // particle nbh
          	);
		
          	fn = f - item.friction;

          	// === update particle informations
          	// ==== Particle i
          	auto &mom_i = cell_i[field::mom][item.p_i];
          	
          	auto mom_res_i = compute_moments(contact_position, ri, f, item.moment);
          	
          	atomicAdd(&cells[item.cell_i][field::mom][item.p_i].x, 0);//mom_res_i.x);
          	atomicAdd(&cells[item.cell_i][field::mom][item.p_i].y, 0);//mom_res_i.y);
          	atomicAdd(&cells[item.cell_i][field::mom][item.p_i].z, 0);//mom_res_i.z);
          	
          	atomicAdd(&cells[item.cell_i][field::fx][item.p_i], 0);//f.x);
          	atomicAdd(&cells[item.cell_i][field::fy][item.p_i], 0);//f.y);
          	atomicAdd(&cells[item.cell_i][field::fz][item.p_i], 0);//f.z);

          	// ==== Particle j
          	auto &mom_j = cell_j[field::mom][item.p_j];
          	
          	auto mom_res_j = compute_moments(contact_position, rj, -f, -item.moment);
          	
          	atomicAdd(&cells[item.cell_j][field::mom][item.p_j].x, 0);//mom_res_j.x);
          	atomicAdd(&cells[item.cell_j][field::mom][item.p_j].y, 0);//mom_res_j.y);
          	atomicAdd(&cells[item.cell_j][field::mom][item.p_j].z, 0);//mom_res_j.z);
          	
          	atomicAdd(&cells[item.cell_j][field::fx][item.p_j], 0);//-f.x);
          	atomicAdd(&cells[item.cell_j][field::fy][item.p_j], 0);//-f.y);
          	atomicAdd(&cells[item.cell_j][field::fz][item.p_j], 0);//-f.z);
       		}
        	else
        	{
          	//item.reset();
          	soa.update(idx, item);
          	dn = 0;
        	}        	
  		
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

      for (size_t type = 0; type <= 3; type++)
      {
        //run_contact_law(parallel_execution_context(), type, classifier, poly, __params__);
        
        auto [data, size] = classifier.get_info(type);
    	InteractionWrapper<InteractionSOA> interactions(data);
    	
    	int blockSize = 256;
    	int numBlocks = (size + blockSize - 1) / blockSize;
    	
    	if(type == 0)
    	{
    		kernelContact1<<<numBlocks, blockSize>>>(interactions, cells, shps, hkp, time, size);
    	}
    	else if(type == 1)
    	{
    		kernelContact2<<<numBlocks, blockSize>>>(interactions, cells, shps, hkp, time, size);
    	}
    	else if(type == 2)
    	{
    		kernelContact3<<<numBlocks, blockSize>>>(interactions, cells, shps, hkp, time, size);
    	}
    	else if(type == 3)
    	{
    		kernelContact4<<<numBlocks, blockSize>>>(interactions, cells, shps, hkp, time, size);
    	}
    	
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
