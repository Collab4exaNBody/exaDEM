//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE


#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/log.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <exanb/core/basic_types.h>
#include <exanb/compute/reduce_cell_particles.h>

#include <mpi.h>
#include <cstring>

#include <exaDEM/simulation_state.h>
#include <exaDEM/dem_simulation_state.h>


namespace exaDEM
{
  using namespace exanb;

  // =================== utility functions ==========================

  // get particle virial tensor. assume the virial is null if particle hasn't virial field
  template<bool> static Mat3d get_particle_virial(const Mat3d* __restrict__, size_t);
  template<> inline Mat3d get_particle_virial<false>(const Mat3d* __restrict__ virials, size_t p_i) { return Mat3d(); }
  template<> inline Mat3d get_particle_virial<true>(const Mat3d* __restrict__ virials, size_t p_i) { return virials[p_i]; }
};
  // ================== Thermodynamic state compute operator ======================
namespace exaDEM
{
  template<
    class GridT ,
	  class = AssertGridHasFields< GridT, field::_vx, field::_vy, field::_vz, field::_mass >
	    >
	    struct SimulationStateNode : public OperatorNode
	    {
	      // compile time constant indicating if grid has type field

	      ADD_SLOT( MPI_Comm           , mpi                 , INPUT , MPI_COMM_WORLD);
	      ADD_SLOT( GridT              , grid                , INPUT , REQUIRED);
	      ADD_SLOT( Domain             , domain              , INPUT , REQUIRED);
	      ADD_SLOT( double             , potential_energy_shift , INPUT , 0.0 );
	      ADD_SLOT( SimulationState    , simulation_state , OUTPUT );

	      static constexpr FieldSet<field::_vx ,field::_vy ,field::_vz, field::_mass> reduce_field_set {};
	      static constexpr FieldSet<field::_vx> reduce_vx_field_set {};
	      static constexpr FieldSet<field::_vy> reduce_vy_field_set {};
	      static constexpr FieldSet<field::_vz> reduce_vz_field_set {};
	      static constexpr FieldSet<field::_mass> reduce_mass_field_set {};
	      inline void execute () override final
	      {
		MPI_Comm comm = *mpi;
		SimulationState& sim_info = *simulation_state;

		Mat3d virial; // constructs itself with 0s
		Vec3d momentum;  // constructs itself with 0s
		Vec3d kinetic_energy;  // constructs itself with 0s
		double potential_energy = 0.;
		double mass = 0.;
		unsigned long long int total_particles = 0;
		/*
		   double init_value = 0;
		   ReduceDoubleFunctor func = {};
		   kinetic_energy.x = reduce_cell_particles( *grid , false , func , init_value, reduce_vx_field_set , gpu_execution_context() , gpu_time_account_func() );
		   kinetic_energy.y = reduce_cell_particles( *grid , false , func , init_value, reduce_vy_field_set , gpu_execution_context() , gpu_time_account_func() );
		   kinetic_energy.z = reduce_cell_particles( *grid , false , func , init_value, reduce_vz_field_set , gpu_execution_context() , gpu_time_account_func() );
		   mass = reduce_cell_particles( *grid , false , func , init_value, reduce_mass_field_set , gpu_execution_context() , gpu_time_account_func() );
		   exaDEM::simulation_state_variables sim {kinetic_energy, momentum, mass, potential_energy, total_particles};
		   */

		exaDEM::simulation_state_variables sim_init {kinetic_energy, momentum, mass, potential_energy, total_particles};
		ReduceSimulationStateFunctor func = {};
		simulation_state_variables sim = reduce_cell_particles( *grid , false , func , sim_init, reduce_field_set , gpu_execution_context() , gpu_time_account_func() );


		auto prof_section = profile_begin_section("mpi");

		// reduce partial sums and share the result
		{
		  // tmp size = 3*3 + 3 + 3 + 1 + 1 + 1 = 18
		  double tmp[18] = {
		    virial.m11, virial.m12, virial.m13, virial.m21, virial.m22, virial.m23,  virial.m31, virial.m32, virial.m33, 
		    sim.momentum.x, sim.momentum.y, sim.momentum.z,
		    sim.kinetic_energy.x, sim.kinetic_energy.y, sim.kinetic_energy.z,
		    sim.potential_energy,
		    sim.mass,
		    static_cast<double>(sim.n_particles) };
		  assert( tmp[17] == total_particles );
		  MPI_Allreduce(MPI_IN_PLACE,tmp,18,MPI_DOUBLE,MPI_SUM,comm);
		  virial.m11 = tmp[0];
		  virial.m12 = tmp[1];
		  virial.m13 = tmp[2];
		  virial.m21 = tmp[3];
		  virial.m22 = tmp[4];
		  virial.m23 = tmp[5];
		  virial.m31 = tmp[6];
		  virial.m32 = tmp[7];
		  virial.m33 = tmp[8];
		  momentum.x = tmp[9];
		  momentum.y = tmp[10];
		  momentum.z = tmp[11];
		  kinetic_energy.x = tmp[12];
		  kinetic_energy.y = tmp[13];
		  kinetic_energy.z = tmp[14];
		  potential_energy = tmp[15];
		  mass = tmp[16];
		  total_particles = tmp[17];
		}

		profile_end_section(prof_section);
		// temperature
		Vec3d temperature = 2. * ( kinetic_energy - 0.5 * momentum * momentum / mass );

		Vec3d virdiag = { virial.m11 , virial.m22, virial.m33 };

		// Volume
		double volume = 1.0;
		if( ! domain->xform_is_identity() )
		{
		  Mat3d mat = domain->xform();
		  Vec3d a { mat.m11, mat.m21, mat.m31 };
		  Vec3d b { mat.m12, mat.m22, mat.m32 };
		  Vec3d c { mat.m13, mat.m23, mat.m33 };
		  volume = dot( cross(a,b) , c );
		}
		volume *= bounds_volume( domain->bounds() );

		Vec3d pressure = ( temperature + virdiag ) / volume;

		// write results to output
		sim_info.set_virial( virial );
		sim_info.set_pressure( pressure );
		sim_info.set_kinetic_energy( kinetic_energy );
		sim_info.set_temperature( temperature );
		sim_info.set_kinetic_momentum( momentum );
		sim_info.set_potential_energy( potential_energy + (*potential_energy_shift) );
		sim_info.set_internal_energy( 0. );
		sim_info.set_chemical_energy( 0. );
		sim_info.set_mass( mass );
		sim_info.set_volume( volume );
		sim_info.set_particle_count( total_particles );
	      }
	    };

  template<class GridT> using SimulationStateNodeTmpl = SimulationStateNode<GridT>;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "simulation_state", make_grid_variant_operator< SimulationStateNodeTmpl > );
  }

}

