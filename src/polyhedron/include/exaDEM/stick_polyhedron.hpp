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

#include <exaDEM/mutexes.h>
#include <exaDEM/shape.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/stick_force.h>

namespace exaDEM
{
  namespace polyhedron
  {
    using namespace exanb;

    /**
     * @brief Atomically adds a value to a double variable.
     *
     * This function atomically adds the specified value to the given double variable.
     *
     * @param val Reference to the double variable to be modified.
     * @param add Value to add atomically to the variable.
     */
    ONIKA_HOST_DEVICE_FUNC
      inline void lockAndAdd(double &val, double add) { ONIKA_CU_ATOMIC_ADD(val, add); }

    /**
     * @brief Atomically adds components of a Vec3d to another Vec3d variable.
     *
     * @param val Reference to the destination Vec3d variable.
     * @param add Rvalue reference to the source Vec3d whose components are to be added.
     */
    ONIKA_HOST_DEVICE_FUNC
      inline void lockAndAdd(Vec3d &val, Vec3d &add)
      {
        ONIKA_CU_ATOMIC_ADD(val.x, add.x);
        ONIKA_CU_ATOMIC_ADD(val.y, add.y);
        ONIKA_CU_ATOMIC_ADD(val.z, add.z);
      }
    ONIKA_HOST_DEVICE_FUNC
      inline void lockAndAdd(Vec3d &val, Vec3d &&add)
      {
        ONIKA_CU_ATOMIC_ADD(val.x, add.x);
        ONIKA_CU_ATOMIC_ADD(val.y, add.y);
        ONIKA_CU_ATOMIC_ADD(val.z, add.z);
      }

    /**
     * @struct stick_law
     * @brief Structure defining contact law interactions for particles (polyhedra).
     */
		template<typename XFormT>
			struct stick_law
			{
				XFormT xform;
				/**
				 * @brief Default constructor for stick_law struct.
				 */
				stick_law() {}

				/**
				 * @brief Retrieves the position vector of a particle from a cell.
				 *
				 * This function retrieves the position vector of a particle identified
				 * by `pi.p` from the given cell using field indices `field::rx`, `field::ry`,
				 * and `field::rz`.
				 *
				 * @tparam TMPLC Type of the cell.
				 * @param cell Reference to the cell containing particle data.
				 * @param pi.p Index of the particle.
				 * @return Vec3d Position vector of the particle.
				 */
				template <typename TMPLC> 
					ONIKA_HOST_DEVICE_FUNC 
					inline const Vec3d get_r(TMPLC &cell, const int p) const
					{
						Vec3d res = {cell[field::rx][p], cell[field::ry][p], cell[field::rz][p]};
						return xform.transformCoord(res);
					};

				/**
				 * @brief Retrieves the velocity vector of a particle from a cell.
				 *
				 * This function retrieves the velocity vector of a particle identified
				 * by `pi.p` from the given cell using field indices `field::vx`, `field::vy`,
				 * and `field::vz`.
				 *
				 * @tparam TMPLC Type of the cell.
				 * @param cell Reference to the cell containing particle data.
				 * @param pi.p Index of the particle.
				 * @return Vec3d Velocity vector of the particle.
				 */
				template <typename TMPLC> ONIKA_HOST_DEVICE_FUNC inline const Vec3d get_v(TMPLC &cell, const int p) const
				{
					const Vec3d res = {cell[field::vx][p], cell[field::vy][p], cell[field::vz][p]};
					return res;
				};

				/**
				 * @brief Operator function for performing interactions between particles (polyhedra).
				 *
				 * @tparam TMPLC Type of the cells or particles container.
				 * @tparam TMPLV Vertex Type container.
				 * @tparam TCFPA Template Contact Force Parameters Accessor.
				 * @param item Reference to the Interaction object representing the interaction details.
				 * @param cells Pointer to the cells or particles container.
				 * @param cpa Reference to the ContactParams object containing interaction parameters.
				 * @param shps Pointer to the shapes array providing shape information for interactions.
				 * @param dt Time increment for the simulation step.
				 */
				template <typename TMPLC, typename TCFPA, typename TMPLV> ONIKA_HOST_DEVICE_FUNC 
					inline std::tuple<double, Vec3d, Vec3d, Vec3d> operator()(
							InnerBondInteraction &item, 
							TMPLC* const __restrict__ cells, 
							TMPLV* const __restrict__ gv, /* grid of vertices */
							TCFPA& cpa, 
							const shape * const shps, 
							const double dt) const
					{
						auto& pi = item.i(); // particle i (id, cell id, particle position, sub vertex)
						auto& pj = item.j(); // particle j (id, cell id, particle position, sub vertex)
																 // === cell
						auto &cell_i = cells[pi.cell];
						auto &cell_j = cells[pj.cell];

						// === positions
						const Vec3d ri = get_r(cell_i, pi.p);
						const Vec3d rj = get_r(cell_j, pj.p);

						// === vrot
						const Vec3d &vrot_i = cell_i[field::vrot][pi.p];
						const Vec3d &vrot_j = cell_j[field::vrot][pj.p];

						// === type
						const auto &type_i = cell_i[field::type][pi.p];
						const auto &type_j = cell_j[field::type][pj.p];

						// === vertex array
						const ParticleVertexView vertices_i = { pi.p, gv[pi.cell] };
						const ParticleVertexView vertices_j = { pj.p, gv[pj.cell] };

						// === shapes
						const shape &shp_i = shps[type_i];
						const shape &shp_j = shps[type_j];

						auto [contact, dn, n, contact_position] = detection_vertex_vertex(vertices_i, pi.sub, &shp_i, vertices_j, pj.sub, &shp_j);
						// temporary vec3d to store forces.
						Vec3d f = {0, 0, 0};
						Vec3d fn = {0, 0, 0};

						// === Contact Force parameters
						const InnerBondParams& cp = cpa(type_i, type_j);

						const Vec3d vi = get_v(cell_i, pi.p);
						const Vec3d vj = get_v(cell_j, pj.p);
						const auto &m_i = cell_i[field::mass][pi.p];
						const auto &m_j = cell_j[field::mass][pj.p];

						const double meff = compute_effective_mass(m_i, m_j);

						stick_force_core(dn, n, item.dn0, dt, cp, meff, item.en, item.et, item.friction, contact_position, 
								ri, vi, f, vrot_i, // particle 1
								rj, vj, vrot_j // particle nbh
								);

						fn = f - item.friction;

						// === update particle informations
						// ==== Particle i
						lockAndAdd(cell_i[field::fx][pi.p], f.x);
						lockAndAdd(cell_i[field::fy][pi.p], f.y);
						lockAndAdd(cell_i[field::fz][pi.p], f.z);

						// ==== Particle j
						lockAndAdd(cell_j[field::fx][pj.p], -f.x);
						lockAndAdd(cell_j[field::fy][pj.p], -f.y);
						lockAndAdd(cell_j[field::fz][pj.p], -f.z);
						return {dn, contact_position, fn, item.friction};
					}
			};

	} // namespace polyhedron
} // namespace exaDEM
