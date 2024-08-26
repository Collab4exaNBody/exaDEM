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

#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <cstdlib>

namespace exaDEM
{
	class SimulationState
	{
		using Mat3d = exanb::Mat3d;
		using Vec3d = exanb::Vec3d;

		public:
		inline void set_kinetic_energy(const Vec3d& x) { m_kinetic_energy = x; }
		inline const Vec3d& kinetic_energy() const { return m_kinetic_energy; }
		inline double kinetic_energy_scal() const { return m_kinetic_energy.x + m_kinetic_energy.y + m_kinetic_energy.z; }

		inline void set_rotation_energy(const Vec3d& x) { m_rotation_energy = x; }
		inline const Vec3d& rotation_energy() const { return m_rotation_energy; }
		inline double rotation_energy_scal() const { return m_rotation_energy.x + m_rotation_energy.y + m_rotation_energy.z; }

		inline void set_mass(double x) { m_mass = x; }
		inline double mass() const { return m_mass; }

		inline void set_volume(double x) { m_volume = x; }
		inline double volume() const { return m_volume; }

		inline void set_particle_count(uint64_t x) { m_particle_count = x; }
		inline uint64_t particle_count() const { return m_particle_count; }

		inline void set_active_interaction_count(uint64_t x) { m_active_interactions = x; }
		inline uint64_t active_interaction_count() const { return m_active_interactions; }

		inline void set_interaction_count(uint64_t x) { m_interaction_count = x; }
		inline uint64_t interaction_count() const { return m_interaction_count; }

		inline void set_dn(double x) { m_dn = x; }
		inline double dn() const { return m_dn; }

		inline size_t compute_particles_throughput(std::chrono::time_point<std::chrono::steady_clock> new_timepoint, int new_timestep) const 
		{
			if(m_last_timestep == -1) return 0; 
			return m_particle_count * (new_timestep - m_last_timestep) / (std::chrono::duration<double>(new_timepoint - m_last_timepoint).count()); 
		}
		inline void update_timestep_timepoint(std::chrono::time_point<std::chrono::steady_clock> new_timepoint, int new_timestep) { m_last_timestep = new_timestep; m_last_timepoint = new_timepoint; }

		private:
		Vec3d m_kinetic_energy;
		Vec3d m_rotation_energy;
		Vec3d m_temperature;
		double m_mass = 0.;
		double m_volume = 0.;
		uint64_t m_particle_count = 0;
    uint64_t m_active_interactions = 0;
    uint64_t m_interaction_count = 0;
    double m_dn = 0;
		int m_last_timestep = -1;
		std::chrono::time_point<std::chrono::steady_clock> m_last_timepoint;
    
	};
}
