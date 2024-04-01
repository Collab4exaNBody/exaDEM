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

		inline void set_temperature(const Vec3d& x) { m_temperature = x; }
		inline const Vec3d& temperature() const { return m_temperature; }
		inline double temperature_scal() const { return ( m_temperature.x + m_temperature.y + m_temperature.z ) / 3. ; }

		inline void set_mass(double x) { m_mass = x; }
		inline double mass() const { return m_mass; }

		inline void set_volume(double x) { m_volume = x; }
		inline double volume() const { return m_volume; }

		inline void set_particle_count(size_t x) { m_particle_count = x; }
		inline size_t particle_count() const { return m_particle_count; }

		inline size_t compute_particles_throughput(std::chrono::time_point<std::chrono::steady_clock> new_timepoint, int new_timestep) const 
		{
			if(m_last_timestep == -1) return 0; 
			return m_particle_count * (new_timestep - m_last_timestep) / (std::chrono::duration<double>(new_timepoint - m_last_timepoint).count()); 
		}
		inline void update_timestep_timepoint(std::chrono::time_point<std::chrono::steady_clock> new_timepoint, int new_timestep) { m_last_timestep = new_timestep; m_last_timepoint = new_timepoint; }

		private:
		Vec3d m_kinetic_energy;
		Vec3d m_temperature;
		double m_mass = 0.;
		double m_volume = 0.;
		size_t m_particle_count = 0;
		int m_last_timestep = -1;
		std::chrono::time_point<std::chrono::steady_clock> m_last_timepoint;
	};

}
