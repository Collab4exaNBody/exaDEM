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

#include <exaDEM/forcefield/inner_bond_force.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/shape.hpp>
#include <exaDEM/shape_detection.hpp>
#include <tuple>

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
ONIKA_HOST_DEVICE_FUNC inline void lockAndAdd(double& val, double add) { ONIKA_CU_ATOMIC_ADD(val, add); }

/**
 * @brief Atomically adds components of a Vec3d to another Vec3d variable.
 *
 * @param val Reference to the destination Vec3d variable.
 * @param add Rvalue reference to the source Vec3d whose components are to be added.
 */
ONIKA_HOST_DEVICE_FUNC inline void lockAndAdd(Vec3d& val, Vec3d& add) {
  ONIKA_CU_ATOMIC_ADD(val.x, add.x);
  ONIKA_CU_ATOMIC_ADD(val.y, add.y);
  ONIKA_CU_ATOMIC_ADD(val.z, add.z);
}

ONIKA_HOST_DEVICE_FUNC inline void lockAndAdd(Vec3d& val, Vec3d&& add) {
  ONIKA_CU_ATOMIC_ADD(val.x, add.x);
  ONIKA_CU_ATOMIC_ADD(val.y, add.y);
  ONIKA_CU_ATOMIC_ADD(val.z, add.z);
}

/**
 * @struct inner_bond_law
 * @brief Structure defining contact law interactions for particles (polyhedra).
 */
template <typename XFormT>
struct inner_bond_law {
  XFormT xform;
  /**
   * @brief Default constructor for inner_bond_law struct.
   */
  inner_bond_law() {}

  /**
   * @brief Retrieves the position vector of a particle from a cell.
   *
   * @tparam TMPLC Type of the cell.
   * @param cell Reference to the cell containing particle data.
   * @param p Index of the particle.
   * @return Vec3d Position vector of the particle (transformed).
   */
  template <typename TMPLC>
  ONIKA_HOST_DEVICE_FUNC inline Vec3d get_r(TMPLC& cell, const int p) const {
    Vec3d res = {cell[field::rx][p], cell[field::ry][p], cell[field::rz][p]};
    return xform.transformCoord(res);
  }

  /**
   * @brief Retrieves the velocity vector of a particle from a cell.
   *
   * @tparam TMPLC Type of the cell.
   * @param cell Reference to the cell containing particle data.
   * @param p Index of the particle.
   * @return Vec3d Velocity vector of the particle.
   */
  template <typename TMPLC>
  ONIKA_HOST_DEVICE_FUNC inline Vec3d get_v(TMPLC& cell, const int p) const {
    return {cell[field::vx][p], cell[field::vy][p], cell[field::vz][p]};
  }

  /**
   * @brief Operator function for performing interactions between particles (polyhedra).
   *
   * @tparam TMPLC Type of the cells or particles container.
   * @tparam TMPLV Vertex Type container.
   * @tparam TIBFPA Template Inner Bond Force Parameters Accessor.
   * @param item Reference to the InnerBondInteraction object representing the interaction details.
   * @param cells Pointer to the cells or particles container.
   * @param ibpa Reference to the InnerBondParams accessor containing interaction parameters.
   * @param shps Pointer to the shapes array providing shape information for interactions.
   * @param dt Time increment for the simulation step.
   */
  template <typename TMPLC, typename TIBFPA, typename TMPLV>
  ONIKA_HOST_DEVICE_FUNC inline std::tuple<double, Vec3d, Vec3d, Vec3d> operator()(
      InnerBondInteraction& item, TMPLC* const __restrict__ cells, TMPLV* const __restrict__ gv, /* grid of vertices */
      TIBFPA& ibpa, const shape* const shps, const double dt) const {
    // particle i (id, cell id, particle position, sub vertex)
    const auto& pi = item.i();
    // particle j (id, cell id, particle position, sub vertex)
    const auto& pj = item.j();
    auto& celli = cells[pi.cell_];
    auto& cellj = cells[pj.cell_];

    assert(pi.p_ < celli.size());
    assert(pj.p_ < cellj.size());

    // === positions
    const Vec3d ri = get_r(celli, pi.p_);
    const Vec3d rj = get_r(cellj, pj.p_);

    // === vrot
    const Vec3d& vroti = celli[field::vrot][pi.p_];
    const Vec3d& vrotj = cellj[field::vrot][pj.p_];

    // === type (for shape lookup) and group (for contact parameters)
    const auto& typei = celli[field::type][pi.p_];
    const auto& typej = cellj[field::type][pj.p_];
    const auto& groupi = celli[field::group][pi.p_];
    const auto& groupj = cellj[field::group][pj.p_];

    // === vertex array
    const ParticleVertexView verticesi = {pi.p_, gv[pi.cell_]};
    const ParticleVertexView verticesj = {pj.p_, gv[pj.cell_]};

    // === homothety
    const double hi = celli[field::homothety][pi.p_];
    const double hj = cellj[field::homothety][pj.p_];

    // === shapes
    const shape& shpi = shps[typei];
    const shape& shpj = shps[typej];

    auto [contact, dn, n, contact_position] =
        detection_vertex_vertex(verticesi, hi, pi.sub_, &shpi, verticesj, hj, pj.sub_, &shpj);

    // === Contact Force parameters
    const InnerBondParams& ibp = ibpa(groupi, groupj);

    const Vec3d vi = get_v(celli, pi.p_);
    const Vec3d vj = get_v(cellj, pj.p_);
    const auto& mi = celli[field::mass][pi.p_];
    const auto& mj = cellj[field::mass][pj.p_];

    const double meff = compute_effective_mass(mi, mj);

    Vec3d fi;  // set by force_law_core
    force_law_core(dn, n, item.dn0_, item.weight_, dt, ibp, meff, item.en_, item.tds_, item.et_, item.friction_,
                   contact_position, ri, vi, fi, vroti,  // particle 1
                   rj, vj, vrotj);                       // particle nbh

    const Vec3d fn = fi - item.friction_;

    // === update particle informations
    // ==== Particle i
    auto& momi = celli[field::mom][pi.p_];
    lockAndAdd(momi, compute_moments(contact_position, ri, fi, Vec3d{}));
    lockAndAdd(celli[field::fx][pi.p_], fi.x);
    lockAndAdd(celli[field::fy][pi.p_], fi.y);
    lockAndAdd(celli[field::fz][pi.p_], fi.z);

    // ==== Particle j
    auto& momj = cellj[field::mom][pj.p_];
    lockAndAdd(momj, compute_moments(contact_position, rj, -fi, Vec3d{}));
    lockAndAdd(cellj[field::fx][pj.p_], -fi.x);
    lockAndAdd(cellj[field::fy][pj.p_], -fi.y);
    lockAndAdd(cellj[field::fz][pj.p_], -fi.z);

    return {dn, contact_position, fn, item.friction_};
  }
};
}  // namespace polyhedron
}  // namespace exaDEM
