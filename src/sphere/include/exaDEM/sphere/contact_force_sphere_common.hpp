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

#include <cstdint>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/classifier/classifier_for_all.hpp>
#include <exaDEM/color_log.hpp>
#include <exaDEM/drivers.hpp>
#include <exaDEM/forcefield/contact_force.hpp>
#include <exaDEM/itools/itools.hpp>
#include <exaDEM/sphere/contact_sphere.hpp>
#include <string>

/**
 * Code shared by the contact force operators of spheres (contact_force_sphere_hooke.cu.in and
 * contact_force_sphere_hertz.cu.in). The parameters of the contact law (ContactParams or HertzParams)
 * are given by the accessors cp and cp_drvs.
 */
namespace exaDEM::sphere {

/**
 * Runs the contact law FuncT on the interactions of the types start, start + 1, ..., end.
 * exec_ctx() returns a new parallel execution context (OperatorNode::parallel_execution_context).
 */
template <int start, int end, template <int, ContactLawType, CohesiveLawType, typename> typename FuncT,
          ContactLawType ContactLaw, CohesiveLawType CohesiveLaw, typename XFormT, typename ExecCtxT, typename... Args>
inline void loop_contact_force(ExecCtxT&& exec_ctx, Classifier& classifier, XFormT& cp_xform, Args&&... args) {
  FuncT<start, ContactLaw, CohesiveLaw, XFormT> contact_law;
  contact_law.xform = cp_xform;
  run_contact_law<start>(exec_ctx(), classifier, contact_law, args...);
  if constexpr (start + 1 <= end) {
    loop_contact_force<start + 1, end, FuncT, ContactLaw, CohesiveLaw, XFormT>(exec_ctx, classifier, cp_xform,
                                                                               std::forward<Args>(args)...);
  }
}

/**
 * Runs the contact law on every interaction of spheres: sphere / sphere (cp) and sphere / drivers (cp_drvs).
 */
template <bool is_sym, ContactLawType ContactLaw, CohesiveLawType CohesiveLaw, typename XFormT, typename ExecCtxT,
          typename CellsT, typename ParticleAccessorT, typename DriverAccessorT>
inline void run_sphere_contact_laws(ExecCtxT&& exec_ctx, Classifier& classifier, XFormT& xform, CellsT* cells,
                                    const DriversGPUAccessor& drvs, const ParticleAccessorT& cp,
                                    const DriverAccessorT& cp_drvs, const double time) {
  ContactLawFunc<is_sym, ContactLaw, CohesiveLaw, XFormT> sph = {xform};
  ContactLawDriverFunc<ContactLaw, CohesiveLaw, Cylinder, XFormT> cyl = {xform};
  ContactLawDriverFunc<ContactLaw, CohesiveLaw, Surface, XFormT> surf = {xform};
  ContactLawDriverFunc<ContactLaw, CohesiveLaw, Ball, XFormT> ball = {xform};

  run_contact_law<InteractionTypeId::VertexVertex>(exec_ctx(), classifier, sph, cells, cp, time);
  run_contact_law<InteractionTypeId::VertexCylinder>(exec_ctx(), classifier, cyl, cells, drvs, cp_drvs, time);
  run_contact_law<InteractionTypeId::VertexSurface>(exec_ctx(), classifier, surf, cells, drvs, cp_drvs, time);
  run_contact_law<InteractionTypeId::VertexBall>(exec_ctx(), classifier, ball, cells, drvs, cp_drvs, time);

  constexpr int rshape_driver_type_start = 7;
  constexpr int rshape_driver_type_end = 9;
  loop_contact_force<rshape_driver_type_start, rshape_driver_type_end, ContactLawRShapeDriverFunc, ContactLaw,
                     CohesiveLaw, XFormT>(exec_ctx, classifier, xform, cells, drvs, cp_drvs, time);
}

/** Analysis: writes the interactions every `frequency` time steps. */
template <typename GridT>
inline void save_interactions(GridT& grid, Classifier& classifier, long timestep, long frequency,
                              const std::string& dir_name, const std::string& interaction_basename) {
  const uint64_t frequency_interaction = frequency;
  bool write_interactions = (frequency_interaction > 0 && timestep % frequency_interaction == 0);
  if (write_interactions) {
    auto stream = itools::create_buffer(grid, classifier);
    std::string ts = std::to_string(timestep);
    itools::write_file(stream, dir_name, interaction_basename + ts);
  }
}

/** Stops the simulation if polyhedron interactions are defined (the operator is the one of spheres). */
inline void check_no_polyhedron_interactions(Classifier& classifier, const std::string& operator_name) {
  for (int i = 1; i <= 3; i++) {
    size_t size = classifier.get_size(i);
    if (size > 0) {
      std::string msg = "The contact operator for spheres is being used, but polyhedron interactions are defined.\n";
      msg += "                        Please, use contact_polyhedron operators.";
      color_log::error(operator_name, msg);
    }
  }
}
}  // namespace exaDEM::sphere
