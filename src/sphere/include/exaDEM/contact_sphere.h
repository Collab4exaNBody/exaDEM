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
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/interaction/interaction.hpp>

namespace exaDEM
{
  /**
   * @brief Atomically adds a value to a double variable.
   *
   * This function performs an atomic addition of the given value to the
   * specified double variable, ensuring thread safety in a parallel computing
   * environment. More details in Onika.
   *
   * @param val The double variable to which the value will be added.
   * @param add The value to add to the variable.
   */
  ONIKA_HOST_DEVICE_FUNC
    inline void lockAndAdd(double &val, double add) { ONIKA_CU_ATOMIC_ADD(val, add); }

  /**
   * @brief Atomically adds a value to a double variable.
   *
   * This function performs an atomic addition of the given value to the
   * specified double variable, ensuring thread safety in a parallel computing
   * environment. More details in Onika.
   *
   * @param val The Vec3d variable to which the value will be added.
   * @param add The value to add to the variable.
   */
  ONIKA_HOST_DEVICE_FUNC
    inline void lockAndAdd(Vec3d &val, Vec3d &&add)
    {
      ONIKA_CU_ATOMIC_ADD(val.x, add.x);
      ONIKA_CU_ATOMIC_ADD(val.y, add.y);
      ONIKA_CU_ATOMIC_ADD(val.z, add.z);
    }

  /**
   * @namespace sphere
   * @brief Namespace for sphere-related simulation functors.
   */
  namespace sphere
  {

    using namespace exanb;

    /**
     * @struct contact_law
     * @brief Template structure for contact law force calculations.
     *
     * This structure provides methods for calculating forces according to contact law
     * in a simulation environment. It supports both symmetric and asymmetric cases.
     *
     * @tparam sym Boolean indicating whether the calculations should be symmetric.
     */
    template <bool sym, bool cohesive, typename XFormT> 
      struct contact_law
      {
        XFormT xform;

        /**
         * @brief Retrieves the position vector of a particle.
         *
         * This function returns the position vector of a particle within a cell.
         *
         * @tparam Cell The type representing a cell in the simulation.
         * @param cell Reference to the cell containing the particle.
         * @param p_id The ID of the particle whose position is to be retrieved.
         * @return The position vector of the particle.
         */
        template <typename Cell> ONIKA_HOST_DEVICE_FUNC inline const Vec3d get_r(Cell &cell, const int p_id) const
        {
          Vec3d res = {cell[field::rx][p_id], cell[field::ry][p_id], cell[field::rz][p_id]};
          return xform.transformCoord(res);
        };

        /**
         * @brief Retrieves the velocity vector of a particle.
         *
         * This function returns the velocity vector of a particle within a cell.
         *
         * @tparam Cell The type representing a cell in the simulation.
         * @param cell Reference to the cell containing the particle.
         * @param p_id The ID of the particle whose velocity is to be retrieved.
         * @return The velocity vector of the particle.
         */
        template <typename Cell> ONIKA_HOST_DEVICE_FUNC inline const Vec3d get_v(Cell &cell, const int p_id) const
        {
          const Vec3d res = {cell[field::vx][p_id], cell[field::vy][p_id], cell[field::vz][p_id]};
          return res;
        };

        /**
         * @brief Handles particle interactions using contact law parameters.
         *
         * This function applies contact law to calculate forces between interacting particles
         * within the given cells. It updates the interaction item based on the specified
         * parameters and uses mutexes for thread safety.
         *
         * @tparam TMPLC The type representing the collection of cells in the simulation.
         * @tparam TCFPA Template Contact Force Parameters Accessor.
         * @param item Reference to the interaction item to be updated.
         * @param cells Pointer to the collection of cells containing the particles.
         * @param cpa Reference to the contact law parameters.
         * @param time Increment simulation time.
         */
        template <typename TMPLC, typename TCFPA>
          ONIKA_HOST_DEVICE_FUNC inline std::tuple<double, Vec3d, Vec3d, Vec3d> operator()(
              Interaction &item, 
              TMPLC *cells, 
              const TCFPA &cpa, 
              const double time) const
          {
            using ContactParamsT = ContactParams; // template later
            // === cell
            auto &cell_i = cells[item.cell_i];
            auto &cell_j = cells[item.cell_j];

            // === positions
            const Vec3d ri = get_r(cell_i, item.p_i);
            const Vec3d rj = get_r(cell_j, item.p_j);

            // === positions
            const double rad_i = cell_i[field::radius][item.p_i];
            const double rad_j = cell_j[field::radius][item.p_j];

            // === vrot
            const Vec3d &vrot_i = cell_i[field::vrot][item.p_i];
            const Vec3d &vrot_j = cell_j[field::vrot][item.p_j];

            auto [contact, dn, n, contact_position] = detection_vertex_vertex_core(ri, rad_i, rj, rad_j);
            Vec3d fn = {0, 0, 0};

						// === types
						const int &t_i = cell_i[field::type][item.p_i];
						const int &t_j = cell_j[field::type][item.p_j];

						// === Conctact Parameters 
						const ContactParamsT& cp = cpa(t_i, t_j); 

						/** if cohesive force */
						if constexpr ( cohesive ) contact = ( contact || dn <= cp.dncut );

						if (contact)
						{
							// === velocities
							const Vec3d vi = get_v(cell_i, item.p_i);
							const Vec3d vj = get_v(cell_j, item.p_j);

							// === mass
							const auto &m_i = cell_i[field::mass][item.p_i];
							const auto &m_j = cell_j[field::mass][item.p_j];

							// temporary vec3d to store forces.
							Vec3d f = {0, 0, 0};
							const double meff = compute_effective_mass(m_i, m_j);

							contact_force_core<cohesive>(dn, n, time, 
									cp, 
									meff, item.friction, contact_position, 
									ri, vi, f, item.moment, vrot_i, // particle 1
									rj, vj, vrot_j                  // particle nbh
									);

							// === For analysis
							fn = f - item.friction;

							// === update particle informations
							// ==== Particle i
							auto &mom_i = cell_i[field::mom][item.p_i];
							lockAndAdd(mom_i, compute_moments(contact_position, ri, f, item.moment));
							lockAndAdd(cell_i[field::fx][item.p_i], f.x);
							lockAndAdd(cell_i[field::fy][item.p_i], f.y);
							lockAndAdd(cell_i[field::fz][item.p_i], f.z);

							if constexpr (sym)
							{
								// ==== Particle j
								auto &mom_j = cell_j[field::mom][item.p_j];
								lockAndAdd(mom_j, compute_moments(contact_position, rj, -f, -item.moment));
								lockAndAdd(cell_j[field::fx][item.p_j], -f.x);
								lockAndAdd(cell_j[field::fy][item.p_j], -f.y);
								lockAndAdd(cell_j[field::fz][item.p_j], -f.z);
							}
						}
						else
						{
							item.reset();
							dn = 0;
						}
						return {dn, contact_position, fn, item.friction};
					}
			};

		/**
		 * @struct contact_law_driver
		 * @brief Template structure for contact law force calculations with various drivers.
		 *
		 * This structure provides methods for calculating forces according to contact law
		 * in a simulation environment. The calculations can involve various types of drivers
		 * such as cylinders, spheres, surfaces, or mesh faces (STL).
		 *
		 * @tparam TMPLD Template parameter for specifying the type of driver.
		 */
		template <bool cohesive, typename TMPLD, typename XFormT> 
			struct contact_law_driver
			{
				XFormT xform;
				/**
				 * @brief Handles particle interactions using contact law parameters and various drivers.
				 *
				 * @tparam TMPLC The type representing the collection of cells in the simulation.
				 * @tparam TCFPA Template Contact Force Parameters Accessor.
				 * @param item Reference to the interaction item to be updated.
				 * @param cells Pointer to the collection of cells containing the particles.
				 * @param drvs Pointer to the collection of drivers, which can include cylinders,
				 *             spheres, surfaces, or mesh faces (STL).
				 * @param cp Reference to the contact law parameters.
				 * @param time Increment simulation time.
				 */
				template <typename TMPLC, typename TCFPA> ONIKA_HOST_DEVICE_FUNC 
					inline std::tuple<double, Vec3d, Vec3d, Vec3d> operator()(
							Interaction &item, 
							TMPLC *cells, 
							const DriversGPUAccessor& drvs, 
							const TCFPA& cpa, 
							const double time) const
					{
            using ContactParamsT = ContactParams; // template later
						const int driver_idx = item.id_j; //
																							// TMPLD& driver = std::get<TMPLD>(drvs[driver_idx]); // issue on GPU
						TMPLD &driver = drvs.get_typed_driver<TMPLD>(driver_idx); // (TMPLD &)(drvs[driver_idx]);
						auto &cell = cells[item.cell_i];
						const size_t p = item.p_i;

						// === positions
						Vec3d r = {cell[field::rx][p], cell[field::ry][p], cell[field::rz][p]};
						r = xform.transformCoord(r);
						const double rad = cell[field::radius][p];

						// === vertex array
						constexpr Vec3d null = {0, 0, 0};
						auto [contact, dn, n, contact_position] = exaDEM::detector_vertex_driver(driver, r, rad);
						Vec3d fn = null;

						// === types
						const auto& type = cell[field::type][item.p_i];

						// === Conctact Parameters 
						const ContactParamsT& cp = cpa(type, driver_idx); 

						/** if cohesive force */
						if constexpr ( cohesive ) contact = ( contact || dn <= cp.dncut );

						if (contact)
						{
							// === vrot
							const Vec3d &vrot = cell[field::vrot][p];
							auto &mom = cell[field::mom][p];
							const Vec3d v = {cell[field::vx][p], cell[field::vy][p], cell[field::vz][p]};
							const double meff = cell[field::mass][p];
							Vec3d f = null;
							contact_force_core<cohesive>(dn, n, time, cp, meff, item.friction, contact_position, 
									r, v, f, item.moment, vrot,                   // particle i
									driver.center, driver.get_vel(), driver.vrot  // particle j
									);

							// === For analysis
							fn = f - item.friction;

							// === update informations
							lockAndAdd(mom, compute_moments(contact_position, r, f, item.moment));
							lockAndAdd(cell[field::fx][p], f.x);
							lockAndAdd(cell[field::fy][p], f.y);
							lockAndAdd(cell[field::fz][p], f.z);

							// only forces now
							if( driver.need_forces() ) lockAndAdd( driver.forces, -f);
						}
						else
						{
							item.reset();
							dn = 0;
						}
						return {dn, contact_position, fn, item.friction};
					}
			};

		/**
		 * @struct contact_law_stl
		 * @brief Structure for applying contact law interactions with STL drivers.
		 *
		 * This structure provides methods for applying contact law interactions between
		 * particles and STL drivers (such as cylinders, spheres, surfaces, or mesh faces).
		 */
		template<int interaction_type, bool cohesive, typename XFormT>
			struct contact_law_stl
			{
				XFormT xform;
				detect<interaction_type> detection; ///< STL mesh detector function object.
				/**
				 * @brief Applies contact law interactions with STL drivers.
				 *
				 * @tparam TMPLC The type representing the collection of cells in the simulation.
				 * @tparam TCFPA Template Contact Force Parameters Accessor.
				 * @param item Reference to the interaction item to be updated.
				 * @param cells Pointer to the collection of cells containing the particles.
				 * @param drvs Pointer to the collection of drivers, which can include cylinders,
				 *             spheres, surfaces, or mesh faces (STL).
				 * @param cpa Reference to the contact law parameters.
				 * @param time The simulation time increment.
				 */
				template <typename TMPLC, typename TPCFA> 
					ONIKA_HOST_DEVICE_FUNC inline std::tuple<double, Vec3d, Vec3d, Vec3d> operator()(
							Interaction &item, 
							TMPLC *cells, 
							const DriversGPUAccessor& drvs, 
							const TPCFA& cpa, 
							const double time) const
					{
            using ContactParamsT = ContactParams; // template later
						const int driver_idx = item.id_j; //
						Stl_mesh &driver = drvs.get_typed_driver<Stl_mesh>(driver_idx); // (Stl_mesh &)(drvs[driver_idx]);
						auto &cell = cells[item.cell_i];

						const size_t p_i = item.p_i;
						const size_t sub_j = item.sub_j;

						// === particle i
						Vec3d r_i = {cell[field::rx][p_i], cell[field::ry][p_i], cell[field::rz][p_i]};
						r_i = xform.transformCoord(r_i); /** Def box */
						const Vec3d &vrot_i = cell[field::vrot][p_i];
						const double radius_i = cell[field::radius][p_i];
						// === driver j
						const auto &shp_j = driver.shp;
						const Quaternion orient_j = driver.quat;
						auto [contact, dn, n, contact_position] = detection(r_i, radius_i, driver.center, sub_j, &shp_j, orient_j);
						Vec3d fn = {0, 0, 0};

						// === types
						const auto& type = cell[field::type][item.p_i];

						// === Conctact Parameters 
						const ContactParamsT& cp = cpa(type, driver_idx); 

						/** if cohesive force */
						if constexpr ( cohesive ) contact = ( contact || dn <= cp.dncut );

						if (contact)
						{

							auto &mom = cell[field::mom][p_i];
							const Vec3d v_i = {cell[field::vx][p_i], cell[field::vy][p_i], cell[field::vz][p_i]};
							const double meff = cell[field::mass][p_i];
							Vec3d f = {0, 0, 0};
							contact_force_core<cohesive>(dn, n, time, cp, 
									meff, item.friction, contact_position, r_i, 
									v_i, f, item.moment, vrot_i, // particle i
									driver.center, driver.get_vel(), driver.vrot // particle j
									);

							// === For analysis
							fn = f - item.friction;

							// === update informations
							lockAndAdd(mom, compute_moments(contact_position, r_i, f, item.moment));
							lockAndAdd(cell[field::fx][p_i], f.x);
							lockAndAdd(cell[field::fy][p_i], f.y);
							lockAndAdd(cell[field::fz][p_i], f.z);

							// only forces now
							if( driver.need_forces() ) lockAndAdd( driver.forces, -f);
						}
						else
						{
							item.reset();
							dn = 0;
						}
						return {dn, contact_position, fn, item.friction};
					}
			};
	} // namespace sphere
} // namespace exaDEM
