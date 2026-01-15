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

#include <exaDEM/forcefield/inner_bond_force.h>
#include <tuple>
#include <exaDEM/shape.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/interaction/interaction.hpp>

namespace exaDEM {
namespace polyhedron {
/**
 * @brief Atomically adds a value to a double variable.
 *
 * This function atomically adds the specified value to the given double variable.
 *
 * @param val Reference to the double variable to be modified.
 * @param add Value to add atomically to the variable.
 */
ONIKA_HOST_DEVICE_FUNC inline void lockAndAdd(double &val, double add) {
  ONIKA_CU_ATOMIC_ADD(val, add);
}

/**
 * @brief Atomically adds components of a Vec3d to another Vec3d variable.
 *
 * @param val Reference to the destination Vec3d variable.
 * @param add Rvalue reference to the source Vec3d whose components are to be added.
 */
ONIKA_HOST_DEVICE_FUNC inline
void lockAndAdd(Vec3d &val, Vec3d &add) {
  ONIKA_CU_ATOMIC_ADD(val.x, add.x);
  ONIKA_CU_ATOMIC_ADD(val.y, add.y);
  ONIKA_CU_ATOMIC_ADD(val.z, add.z);
}

ONIKA_HOST_DEVICE_FUNC inline
void lockAndAdd(Vec3d &val, Vec3d &&add) {
  ONIKA_CU_ATOMIC_ADD(val.x, add.x);
  ONIKA_CU_ATOMIC_ADD(val.y, add.y);
  ONIKA_CU_ATOMIC_ADD(val.z, add.z);
}

/**
 * @struct inner_bond_law
 * @brief Structure defining contact law interactions for particles (polyhedra).
 */
template<typename XFormT>
struct inner_bond_law {
  XFormT xform;
  /**
   * @brief Default constructor for inner_bond_law struct.
   */
  inner_bond_law() {}

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
  ONIKA_HOST_DEVICE_FUNC inline
  const Vec3d get_r(TMPLC &cell, const int p) const {
    Vec3d res = {cell[field::rx][p], cell[field::ry][p], cell[field::rz][p]};
    return xform.transformCoord(res);
  }

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
  template <typename TMPLC>
  ONIKA_HOST_DEVICE_FUNC inline
  const Vec3d get_v(TMPLC &cell, const int p) const {
    const Vec3d res = {cell[field::vx][p], cell[field::vy][p], cell[field::vz][p]};
    return res;
  }

  /**
   * @brief Operator function for performing interactions between particles (polyhedra).
   *
   * @tparam TMPLC Type of the cells or particles container.
   * @tparam TMPLV Vertex Type container.
   * @tparam TIBFPA Template Inner Bond Force Parameters Accessor.
   * @param item Reference to the Interaction object representing the interaction details.
   * @param cells Pointer to the cells or particles container.
   * @param cpa Reference to the ContactParams object containing interaction parameters.
   * @param shps Pointer to the shapes array providing shape information for interactions.
   * @param dt Time increment for the simulation step.
   */
  template <typename TMPLC, typename TIBFPA, typename TMPLV>
  ONIKA_HOST_DEVICE_FUNC inline
  std::tuple<double, Vec3d, Vec3d, Vec3d> operator()(
      InnerBondInteraction &item,
      TMPLC* const __restrict__ cells,
      TMPLV* const __restrict__ gv, /* grid of vertices */
      TIBFPA& ibpa,
      const shape * const shps,
      const double dt) const {
    // particle i (id, cell id, particle position, sub vertex)
    const auto& pi = item.i();
    // particle j (id, cell id, particle position, sub vertex)
    const auto& pj = item.j();
    auto &celli = cells[pi.cell];
    auto &cellj = cells[pj.cell];

    assert(pi.p < celli.size());
    assert(pj.p < cellj.size());

    // === positions
    const Vec3d ri = get_r(celli, pi.p);
    const Vec3d rj = get_r(cellj, pj.p);

    // === vrot
    const Vec3d &vroti = celli[field::vrot][pi.p];
    const Vec3d &vrotj = cellj[field::vrot][pj.p];

    // === type
    const auto &typei = celli[field::type][pi.p];
    const auto &typej = cellj[field::type][pj.p];

    // === vertex array
    const ParticleVertexView verticesi = { pi.p, gv[pi.cell] };
    const ParticleVertexView verticesj = { pj.p, gv[pj.cell] };

    // === homothety
    const double hi = celli[field::homothety][pi.p];
    const double hj = cellj[field::homothety][pj.p];

    // === shapes
    const shape &shpi = shps[typei];
    const shape &shpj = shps[typej];

    auto [contact, dn, n, contact_position] = detection_vertex_vertex(
        verticesi, hi, pi.sub, &shpi,
        verticesj, hj, pj.sub, &shpj);

    // temporary vec3d to store forces.
    Vec3d fi = {0, 0, 0};
    Vec3d fn = {0, 0, 0};

    // === Contact Force parameters
    const InnerBondParams& ibp = ibpa(typei, typej);

    const Vec3d vi = get_v(celli, pi.p);
    const Vec3d vj = get_v(cellj, pj.p);
    const auto &mi = celli[field::mass][pi.p];
    const auto &mj = cellj[field::mass][pj.p];

    const double meff = compute_effective_mass(mi, mj);

    force_law_core(dn, n, item.dn0, dt, ibp, meff,
                   item.en, item.et, item.friction, contact_position,
                   ri, vi, fi, vroti,  // particle 1
                   rj, vj, vrotj);     // particle nbh

    fn = fi - item.friction;
    Vec3d null = {0, 0, 0};

    // === update particle informations
    // ==== Particle i
    auto &momi = celli[field::mom][pi.p];
    lockAndAdd(momi, compute_moments(contact_position, ri, fi, null));
    lockAndAdd(celli[field::fx][pi.p], fi.x);
    lockAndAdd(celli[field::fy][pi.p], fi.y);
    lockAndAdd(celli[field::fz][pi.p], fi.z);

    // ==== Particle j
    auto &momj = cellj[field::mom][pj.p];
    lockAndAdd(momj, compute_moments(contact_position, rj, -fi, null));
    lockAndAdd(cellj[field::fx][pj.p], -fi.x);
    lockAndAdd(cellj[field::fy][pj.p], -fi.y);
    lockAndAdd(cellj[field::fz][pj.p], -fi.z);

    return {dn, contact_position, fn, item.friction};
  }
};
}  // namespace polyhedron
}  // namespace exaDEM
