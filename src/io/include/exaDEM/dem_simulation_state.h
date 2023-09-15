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
		inline void set_virial(const Mat3d& x) { m_virial = x; }
		inline const Mat3d& virial() const { return m_virial; }

		inline void set_pressure(const Vec3d& x) { m_pressure = x; }
		inline const Vec3d& pressure() const { return m_pressure; }
		inline double pressure_scal() const
		{
			using namespace exanb;
			double kinetic_pressure = temperature_scal() / volume();
			Vec3d virdiag = { m_virial.m11 , m_virial.m22, m_virial.m33 };
			double potential_pressure = ( virdiag.x + virdiag.y + virdiag.z ) / ( 3. * volume() );
			return kinetic_pressure + potential_pressure;
		}

		inline void set_vonmises(const Vec3d& x) { m_vonmises = x; }
		inline const Vec3d& vonmises() const { return m_vonmises; }
		inline double vonmises_scal() const
		{
			Vec3d virdiag = { m_virial.m11 , m_virial.m22, m_virial.m33 };
			Vec3d virdeviatoric = { m_virial.m12 , m_virial.m13, m_virial.m23 };
			double vonmises = sqrt( 0.5 * (( virdiag.x - virdiag.y ) * ( virdiag.x - virdiag.y ) + ( virdiag.y - virdiag.z ) * ( virdiag.y - virdiag.z ) + ( virdiag.z - virdiag.x ) * ( virdiag.z - virdiag.x ) + 6.0 * ( virdeviatoric.x * virdeviatoric.x + virdeviatoric.y * virdeviatoric.y + virdeviatoric.z * virdeviatoric.z)) ) / volume();
			return vonmises;
		}

		inline Mat3d stress_tensor() const
		{
			Mat3d S = (1./volume()) * m_virial;
			return S;
		}

		inline void set_kinetic_energy(const Vec3d& x) { m_kinetic_energy = x; }
		inline const Vec3d& kinetic_energy() const { return m_kinetic_energy; }
		inline double kinetic_energy_scal() const { return m_kinetic_energy.x + m_kinetic_energy.y + m_kinetic_energy.z; }

		inline void set_rotational_energy(const Vec3d& x) { m_rotational_energy = x; }
		inline const Vec3d& rotational_energy() const { return m_rotational_energy; }
		inline double rotational_energy_scal() const { return m_rotational_energy.x + m_rotational_energy.y + m_rotational_energy.z; }

		inline void set_ndof(const Vec3d& x) { m_ndof = x; }
		inline const Vec3d& ndof() const { return m_ndof; }
		inline double ndof_scal() const { return m_ndof.x + m_ndof.y + m_ndof.z; }

		inline void set_temperature(const Vec3d& x) { m_temperature = x; }
		inline const Vec3d& temperature() const { return m_temperature; }
		inline double temperature_scal() const { return ( m_temperature.x + m_temperature.y + m_temperature.z ) / 3. ; }
		inline double temperature_rigidmol_scal() const { return ( m_temperature.x + m_temperature.y + m_temperature.z ) / (3. + ndof_scal() / particle_count()) ; }

		inline void set_kinetic_temperature(const Vec3d& x) { m_kinetic_temperature = x; }
		inline const Vec3d& kinetic_temperature() const { return m_kinetic_temperature; }
		inline double kinetic_temperature_x() const { return m_kinetic_temperature.x ; }
		inline double kinetic_temperature_y() const { return m_kinetic_temperature.y ; }
		inline double kinetic_temperature_z() const { return m_kinetic_temperature.z ; }

		inline void set_rotational_temperature(const Vec3d& x) { m_rotational_temperature = x; }
		inline const Vec3d& rotational_temperature() const { return m_rotational_temperature; }
		inline double rotational_temperature_x() const { return m_rotational_temperature.x ; }
		inline double rotational_temperature_y() const { return m_rotational_temperature.y ; }
		inline double rotational_temperature_z() const { return m_rotational_temperature.z ; }

		inline void set_kinetic_momentum(const Vec3d& x) { m_kinetic_momentum = x; }
		inline const Vec3d& kinetic_momentum() const { return m_kinetic_momentum; }

		inline void set_potential_energy(double x) { m_potential_energy = x; }
		inline double potential_energy() const { return m_potential_energy; }

		inline void set_internal_energy(double x) { m_internal_energy = x; }
		inline double internal_energy() const { return m_internal_energy; }

		inline void set_chemical_energy(double x) { m_chemical_energy = x; }
		inline double chemical_energy() const { return m_chemical_energy; }

		inline double total_energy() const { return kinetic_energy_scal() + potential_energy() + internal_energy() + chemical_energy(); }
		inline double total_energy_rigidmol() const { return kinetic_energy_scal() + rotational_energy_scal() + potential_energy() + internal_energy() + chemical_energy(); }

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
		Mat3d m_virial;
		Vec3d m_pressure;
		Vec3d m_vonmises;
		Vec3d m_kinetic_energy;
		Vec3d m_rotational_energy;
		Vec3d m_temperature;
		Vec3d m_kinetic_temperature;
		Vec3d m_rotational_temperature;
		Vec3d m_kinetic_momentum;
		Vec3d m_ndof;
		double m_potential_energy = 0.;
		double m_internal_energy = 0.;
		double m_chemical_energy = 0.;
		double m_mass = 0.;
		double m_volume = 0.;
		size_t m_particle_count = 0;
		int m_last_timestep = -1;
		std::chrono::time_point<std::chrono::steady_clock> m_last_timepoint;
	};

}
