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
     * @struct contact_law
     * @brief Structure defining contact law interactions for particles (polyhedra).
     */
    template<int interaction_type, ContactLawType ContactLaw, CohesiveLawType CohesiveLaw, typename XFormT>
      struct contact_law
      {

        XFormT xform;
        detect<interaction_type> detection;
        /**
         * @brief Default constructor for contact_law struct.
         */
        contact_law() {}

        /**
         * @brief Retrieves the position vector of a particle from a cell.
         *
         * This function retrieves the position vector of a particle identified
         * by `p_id` from the given cell using field indices `field::rx`, `field::ry`,
         * and `field::rz`.
         *
         * @tparam TMPLC Type of the cell.
         * @param cell Reference to the cell containing particle data.
         * @param p_id Index of the particle.
         * @return Vec3d Position vector of the particle.
         */
        template <typename TMPLC> 
          ONIKA_HOST_DEVICE_FUNC 
          inline const Vec3d get_r(TMPLC &cell, const int p_id) const
          {
            Vec3d res = {cell[field::rx][p_id], cell[field::ry][p_id], cell[field::rz][p_id]};
            return xform.transformCoord(res);
          };

        /**
         * @brief Retrieves the velocity vector of a particle from a cell.
         *
         * This function retrieves the velocity vector of a particle identified
         * by `p_id` from the given cell using field indices `field::vx`, `field::vy`,
         * and `field::vz`.
         *
         * @tparam TMPLC Type of the cell.
         * @param cell Reference to the cell containing particle data.
         * @param p_id Index of the particle.
         * @return Vec3d Velocity vector of the particle.
         */
        template <typename TMPLC> ONIKA_HOST_DEVICE_FUNC inline const Vec3d get_v(TMPLC &cell, const int p_id) const
        {
          const Vec3d res = {cell[field::vx][p_id], cell[field::vy][p_id], cell[field::vz][p_id]};
          return res;
        };

        /**
         * @brief Operator function for performing interactions between particles (polyhedra).
         *
         * @tparam TMPLC Type of the cells or particles container.
         * @tparam TCFPA Template Contact Force Parameters Accessor.
         * @tparam TMPLV Vertex Type container.
         * @param item Reference to the Interaction object representing the interaction details.
         * @param cells Pointer to the cells or particles container.
         * @param cpa Reference to the ContactParams object containing interaction parameters.
         * @param shps Pointer to the shapes array providing shape information for interactions.
         * @param dt Time increment for the simulation step.
         */
        template <typename TMPLC, typename TCFPA, typename TMPLV> ONIKA_HOST_DEVICE_FUNC 
          inline std::tuple<double, Vec3d, Vec3d, Vec3d> operator()(
              Interaction &item, 
              TMPLC* const __restrict__ cells, 
              TMPLV* const __restrict__ gv, /* grid of vertices */
              TCFPA& cpa, 
              const shape * const shps, 
              const double dt) const
          {
            // === cell
            auto &cell_i = cells[item.cell_i];
            auto &cell_j = cells[item.cell_j];

            // === positions
            const Vec3d ri = get_r(cell_i, item.p_i);
            const Vec3d rj = get_r(cell_j, item.p_j);

            // === vrot
            const Vec3d &vrot_i = cell_i[field::vrot][item.p_i];
            const Vec3d &vrot_j = cell_j[field::vrot][item.p_j];

            // === type
            const auto &type_i = cell_i[field::type][item.p_i];
            const auto &type_j = cell_j[field::type][item.p_j];

            // === vertex array
            const ParticleVertexView vertices_i = { item.p_i, gv[item.cell_i] };
            const ParticleVertexView vertices_j = { item.p_j, gv[item.cell_j] };

            // === shapes
            const shape &shp_i = shps[type_i];
            const shape &shp_j = shps[type_j];

            auto [contact, dn, n, contact_position] = detection(vertices_i, item.sub_i, &shp_i, vertices_j, item.sub_j, &shp_j);
            // temporary vec3d to store forces.
            Vec3d f = {0, 0, 0};
            Vec3d fn = {0, 0, 0};

            // === Contact Force parameters
            const ContactParams& cp = cpa(type_i, type_j);
            constexpr auto LawCombo = makeLawCombo(ContactLaw, CohesiveLaw);

            /** if cohesive force */
            if constexpr ( LawComboTraits<LawCombo>::cohesive ) contact = ( contact || dn <= cp.dncut );

            if (contact)
            {
              const Vec3d vi = get_v(cell_i, item.p_i);
              const Vec3d vj = get_v(cell_j, item.p_j);
              const auto &m_i = cell_i[field::mass][item.p_i];
              const auto &m_j = cell_j[field::mass][item.p_j];

              double rad_i = shp_i.m_radius;
              double rad_j = shp_j.m_radius;



              const double meff = compute_effective_mass(m_i, m_j);
              const double reff = compute_effective_mass(rad_i, rad_j);


              contact_force_core<ContactLaw, CohesiveLaw>(dn, n, dt, cp, meff, reff, item.friction, contact_position, 
                  ri, vi, f, item.moment, vrot_i, // particle 1
                  rj, vj, vrot_j // particle nbh
                  );

              fn = f - item.friction;

              // === update particle informations
              // ==== Particle i
              auto &mom_i = cell_i[field::mom][item.p_i];
              lockAndAdd(mom_i, compute_moments(contact_position, ri, f, item.moment));
              lockAndAdd(cell_i[field::fx][item.p_i], f.x);
              lockAndAdd(cell_i[field::fy][item.p_i], f.y);
              lockAndAdd(cell_i[field::fz][item.p_i], f.z);

              // ==== Particle j
              auto &mom_j = cell_j[field::mom][item.p_j];
              lockAndAdd(mom_j, compute_moments(contact_position, rj, -f, -item.moment));
              lockAndAdd(cell_j[field::fx][item.p_j], -f.x);
              lockAndAdd(cell_j[field::fy][item.p_j], -f.y);
              lockAndAdd(cell_j[field::fz][item.p_j], -f.z);
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
     * @brief Struct for applying contact law interactions driven by drivers.
     * @tparam TMPLD Type of the drivers.
     */
    template <ContactLawType ContactLaw, CohesiveLawType CohesiveLaw, typename TMPLD> struct contact_law_driver
    {
      //using driven_t = std::variant<exaDEM::Cylinder, exaDEM::Surface, exaDEM::Ball, exaDEM::Stl_mesh, exaDEM::UndefinedDriver>;
      /**
       * @brief Functor for applying contact law interactions driven by drivers.
       *
       * This functor applies contact law interactions between particles or cells, driven by
       * specified drivers (`drvs`). It uses interaction parameters (`hkp`), precomputed shapes
       * (`shps`), and a time increment (`dt`) for simulation.
       *
       * @tparam TMPLC Type of the cells or particles container.
       * @tparam TCFPA Template Contact Force Parameters Accessor.
       * @tparam TMPLV Vertex Type container.
       * @param item Reference to the Interaction object representing the interaction details.
       * @param cells Pointer to the cells or particles container.
       * @param drvs Pointer to the Drivers object providing driving forces.
       * @param hkp Reference to the ContactParams object containing interaction parameters.
       * @param shps Pointer to the shapes array providing shape information for interactions.
       * @param dt Time increment for the simulation step.
       */
      template <typename TMPLC, typename TCFPA, typename TMPLV> 
        ONIKA_HOST_DEVICE_FUNC inline std::tuple<double, Vec3d, Vec3d, Vec3d> operator()(
            Interaction &item, 
            TMPLC * __restrict__ cells, 
            TMPLV* const __restrict__ gv, /* grid of vertices */
            const DriversGPUAccessor& drvs, 
            TCFPA &cpa,
            const shape *shps, 
            const double dt) const
        {
          //const int driver_idx = item.sub; //
          const int driver_idx = item.id_j; //
                                            // TMPLD& driver        = std::get<TMPLD>(drvs[driver_idx]) ;
          TMPLD &driver = drvs.get_typed_driver<TMPLD>(driver_idx); // (TMPLD &)(drvs[driver_idx]);
          auto &cell = cells[item.cell_i];
          const auto type = cell[field::type][item.p_i];
          auto &shp = shps[type];

          const size_t p = item.p_i;
          const size_t sub = item.sub_i;
          // === positions
          const Vec3d r = {cell[field::rx][p], cell[field::ry][p], cell[field::rz][p]};
          // === vertex array
          ParticleVertexView vertices = { p, gv[item.cell_i] };

          auto [contact, dn, n, contact_position] = exaDEM::detector_vertex_driver(driver, vertices, sub, &shp);
          constexpr Vec3d null = {0, 0, 0};
          Vec3d f = null;
          Vec3d fn = null;

          // === Contact Force Parameters
          const ContactParams& cp = cpa(type, driver_idx);
          constexpr auto LawCombo = makeLawCombo(ContactLaw, CohesiveLaw);
          /** if cohesive force */
          if constexpr ( LawComboTraits<LawCombo>::cohesive ) contact = ( contact || dn <= cp.dncut );

          if (contact)
          {
            // === vrot
            const Vec3d &vrot = cell[field::vrot][p];

            auto &mom = cell[field::mom][p];
            const Vec3d v = {cell[field::vx][p], cell[field::vy][p], cell[field::vz][p]};
            const double meff = cell[field::mass][p];

            const auto &type = cell[field::type][p];
            const shape &shp = shps[type];
            //const double reff = shp->minskowski();
            const double reff = shp.m_radius;


            contact_force_core<ContactLaw, CohesiveLaw>(dn, n, dt, cp, meff, reff, item.friction, contact_position, 
                r, v, f, item.moment, vrot, // particle i
                driver.center, driver.get_vel(), driver.vrot // particle j
                );

            // === for analysis
            fn = f - item.friction;

            // === update informations
            lockAndAdd(mom, compute_moments(contact_position, r, f, item.moment));
            lockAndAdd(cell[field::fx][p], f.x);
            lockAndAdd(cell[field::fy][p], f.y);
            lockAndAdd(cell[field::fz][p], f.z);

            if( driver.need_forces() )
            {
              lockAndAdd( driver.forces, -f);
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
     * @brief Functor for applying contact law interactions with STL mesh objects.
     */
    template<int interaction_type, ContactLawType ContactLaw, CohesiveLawType CohesiveLaw, typename XFormT> /* def xform does nothing*/
      struct contact_law_stl
      {
        //using driver_t = std::variant<exaDEM::Cylinder, exaDEM::Surface, exaDEM::Ball, exaDEM::Stl_mesh, exaDEM::UndefinedDriver>;
        XFormT xform;
        detect<interaction_type> detection;

        /* Default constructor */
        contact_law_stl() {}

        /**
         * @brief Operator function for applying contact law interactions with STL mesh objects.
         *
         * This function applies contact law interactions between particles or cells and STL mesh objects,
         * driven by specified drivers (`drvs`). It uses interaction parameters (`hkp`), precomputed shapes
         * (`shps`), and a time increment (`dt`) for simulation.
         *
         * @tparam TMPLC Type of the cells or particles container.
         * @tparam TCFPA Template Contact Force Parameters Accessor.
         * @tparam TMPLV Vertex Type container.
         * @param item Reference to the Interaction object representing the interaction details.
         * @param cells Pointer to the cells or particles container.
         * @param drvs Pointer to the Drivers object providing driving forces.
         * @param hkp Reference to the ContactParams object containing interaction parameters.
         * @param shps Pointer to the shapes array providing shape information for interactions.
         * @param dt Time increment for the simulation step.
         */
        template <typename TMPLC, typename TCFPA, typename TMPLV> 
          ONIKA_HOST_DEVICE_FUNC inline std::tuple<double, Vec3d, Vec3d, Vec3d> operator()(
              Interaction &item, 
              TMPLC * __restrict__ cells, 
              TMPLV* const __restrict__ gv, /* grid of vertices */
              const DriversGPUAccessor& drvs, 
              TCFPA& cpa, 
              const shape *shps, 
              const double dt) const
          {
            const int driver_idx = item.id_j; //
            Stl_mesh &driver = drvs.get_typed_driver<exaDEM::Stl_mesh>(driver_idx); // (exaDEM::Stl_mesh &)(drvs[driver_idx]);
            auto &cell = cells[item.cell_i];
            // renaming
            const size_t p_i = item.p_i;
            const size_t sub_i = item.sub_i;
            const size_t sub_j = item.sub_j;

            // get shapes
            const auto type = cell[field::type][item.p_i];
            const auto &shp_i = shps[type];
            const auto &shp_j = driver.shp;


            // === positions
            const Vec3d r_i = {cell[field::rx][p_i], cell[field::ry][p_i], cell[field::rz][p_i]};
            const ParticleVertexView vertices_i = { p_i, gv[item.cell_i] };
            // === vrot
            const Vec3d &vrot_i = cell[field::vrot][p_i];

            // STL Vertices
            const Vec3d* const stl_vertices =  onika::cuda::vector_data( driver.vertices ); 
            // === detection
            auto [contact, dn, n, contact_position] = detection(vertices_i, sub_i, &shp_i, stl_vertices, sub_j, &shp_j);
            constexpr Vec3d null = {0, 0, 0};
            Vec3d fn = null;

            // === Contact Force Parameters
            const ContactParams& cp = cpa(type, driver_idx);
            constexpr auto LawCombo = makeLawCombo(ContactLaw, CohesiveLaw);
            /** if cohesive force */
            if constexpr ( LawComboTraits<LawCombo>::cohesive ) contact = ( contact || dn <= cp.dncut );

            if (contact)
            {
              Vec3d f = null;
              auto &mom = cell[field::mom][p_i];
              const Vec3d v_i = {cell[field::vx][p_i], cell[field::vy][p_i], cell[field::vz][p_i]};
              const double meff = cell[field::mass][p_i];
              const auto &type_i = cell[field::type][p_i];
              const shape &shp_i = shps[type_i];
              const double reff = shp_i.m_radius;


              // i to j
              if constexpr (interaction_type <= 10 && interaction_type >= 7 )
              {
                contact_force_core<ContactLaw, CohesiveLaw>(dn, n, dt, cp, meff, reff,
                    item.friction, contact_position, 
                    r_i, v_i, f, item.moment, vrot_i,       // particle i
                    driver.center, driver.vel, driver.vrot  // particle j
                    );

                // === used for analysis
                fn = f - item.friction;
                // === update informations
                lockAndAdd(mom, compute_moments(contact_position, r_i, f, item.moment));
                lockAndAdd(cell[field::fx][p_i], f.x);
                lockAndAdd(cell[field::fy][p_i], f.y);
                lockAndAdd(cell[field::fz][p_i], f.z);
                if( driver.need_forces() ) lockAndAdd( driver.forces, -f);
                if( driver.need_moment() ) lockAndAdd( driver.mom, compute_moments(contact_position, driver.center, -f, -item.moment) );
              }

              //  j to i 
              if constexpr (interaction_type <= 12 && interaction_type >= 11 )
              {
                contact_force_core<ContactLaw, CohesiveLaw>(dn, n, dt, cp, meff, reff,
                    item.friction, contact_position, 
                    driver.center, driver.get_vel(), f, item.moment, driver.vrot,  // particle j
                    r_i, v_i,  vrot_i       // particle i
                    );

                // === used for analysis
                fn = item.friction - f;
                // === update informations
                lockAndAdd(mom, compute_moments(contact_position, r_i, -f, -item.moment));
                lockAndAdd(cell[field::fx][p_i], -f.x);
                lockAndAdd(cell[field::fy][p_i], -f.y);
                lockAndAdd(cell[field::fz][p_i], -f.z);
                item.friction = -item.friction;
                if( driver.need_forces() ) lockAndAdd( driver.forces, f); 
                if( driver.need_moment() ) lockAndAdd( driver.mom, compute_moments(contact_position, driver.center, f, item.moment));
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
  } // namespace polyhedron
} // namespace exaDEM
