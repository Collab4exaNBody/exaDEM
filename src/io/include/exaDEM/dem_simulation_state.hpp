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

#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>

#include <cstdlib>

namespace exaDEM {
class SimulationState {
  using Mat3d = exanb::Mat3d;
  using Vec3d = exanb::Vec3d;

 public:
  inline void set_kinetic_energy(const Vec3d& x) { kinetic_energy_ = x; }

  inline const Vec3d& kinetic_energy() const { return kinetic_energy_; }

  inline double kinetic_energy_scal() const { return kinetic_energy_.x + kinetic_energy_.y + kinetic_energy_.z; }

  inline void set_rotation_energy(const Vec3d& x) { rotation_energy_ = x; }

  inline const Vec3d& rotation_energy() const { return rotation_energy_; }

  inline double rotation_energy_scal() const { return rotation_energy_.x + rotation_energy_.y + rotation_energy_.z; }

  inline void set_mass(double x) { mass_ = x; }

  inline double mass() const { return mass_; }

  inline void set_volume(double x) { volume_ = x; }

  inline double volume() const { return volume_; }

  inline void set_particle_count(uint64_t x) { particle_count_ = x; }

  inline uint64_t particle_count() const { return particle_count_; }

  inline void set_active_interaction_count(uint64_t x) { active_interactions_ = x; }

  inline uint64_t active_interaction_count() const { return active_interactions_; }

  inline void set_interaction_count(uint64_t x) { interaction_count_ = x; }

  inline uint64_t interaction_count() const { return interaction_count_; }

  inline void set_dn(double x) { dn_ = x; }

  inline double dn() const { return dn_; }

  inline void set_interface_count(uint64_t n) { interface_count_ = n; }

  inline uint64_t interface_count() const { return interface_count_; }

  inline size_t compute_particles_throughput(std::chrono::time_point<std::chrono::steady_clock> new_timepoint,
                                             int new_timestep) const {
    if (last_timestep_ == -1) {
      return 0;
    }
    return particle_count_ * (new_timestep - last_timestep_) /
           (std::chrono::duration<double>(new_timepoint - last_timepoint_).count());
  }
  inline void update_timestep_timepoint(std::chrono::time_point<std::chrono::steady_clock> new_timepoint,
                                        int new_timestep) {
    last_timestep_ = new_timestep;
    last_timepoint_ = new_timepoint;
  }

 private:
  Vec3d kinetic_energy_;
  Vec3d rotation_energy_;
  Vec3d temperature_;
  double mass_ = 0.;
  double volume_ = 0.;
  uint64_t particle_count_ = 0;
  uint64_t active_interactions_ = 0;
  uint64_t interaction_count_ = 0;
  double dn_ = 0;
  uint64_t interface_count_ = 0;
  int last_timestep_ = -1;
  std::chrono::time_point<std::chrono::steady_clock> last_timepoint_;
};
}  // namespace exaDEM
