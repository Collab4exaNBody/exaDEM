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
//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/log.h>
#include <exanb/core/cpp_utils.h>

#include <yaml-cpp/yaml.h>
#include <exanb/core/quantity_yaml.h>

#include <exanb/core/config.h> // for MAX_PARTICLE_NEIGHBORS
#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/compute/compute_cell_particle_pairs.h>

#include <onika/memory/allocator.h> // for DEFAULT_ALIGNMENT

#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/hooke_force_parameters.h>
#include <exaDEM/neighbor_friction.h>


namespace exaDEM
{
	using namespace exanb;

	// Reaction Field Compute functor
	struct alignas(onika::memory::DEFAULT_ALIGNMENT) HookeForceOp 
	{
		// poetential parameters
		const HookeParams m_params;
		const double m_dt;

		template<class CellParticlesT>
			ONIKA_HOST_DEVICE_FUNC inline void operator ()
			(      
			 const Vec3d& dr,   // imposes par la forme sans compute_buffer
			 double d2,         //
			 double radius, // les fiels requis
			 double vx,
			 double vy,
			 double vz,
			 double mass,
			 const Vec3d& vrot,
			 double& fx,
			 double& fy,
			 double& fz,
			 Vec3d& mom,
			 double& homothety,
			 CellParticlesT* cells, // fournis par l'appelant, pour accéder aux attributs du voisins
			 size_t cell_b,
			 size_t p_b,
			 ParticlePairFriction& friction
			) const
			{
				const double drx = dr.x;
				const double dry = dr.y;
				const double drz = dr.z;

				const double vx_nbh = cells[cell_b][field::vx][p_b];
				const double vy_nbh = cells[cell_b][field::vy][p_b];
				const double vz_nbh = cells[cell_b][field::vz][p_b];
				const double mass_nbh = cells[cell_b][field::mass][p_b];
				const double radius_nbh = cells[cell_b][field::radius][p_b];
				const Vec3d vrot_nbh = cells[cell_b][field::vrot][p_b];

				Vec3d& ft = friction.m_friction;
				double fx_nbh(0.0), fy_nbh(0.0), fz_nbh(0.0);
				Vec3d mom_nbh = {0,0,0};
				exaDEM::compute_hooke_force( 
						m_params.dncut, m_dt, m_params.m_kn, m_params.m_kt, m_params.m_kr,   
						m_params.m_fc, m_params.m_mu, m_params.m_damp_rate, 
						ft,
						0., 0., 0.,         
						vx, vy, vz,           
						mass, radius,     
						fx, fy, fz,     
						mom, vrot, 
						drx, dry, drz, 
						vx_nbh, vy_nbh, vz_nbh, 
						mass_nbh, radius_nbh, 
						fx_nbh, fy_nbh, fz_nbh, 
						mom_nbh, vrot_nbh 
						);

				homothety += ft.x * ft.x + ft.y * ft.y + ft.z * ft.z;
			}
	};

}


namespace exanb
{
	// specialize functor traits to allow Cuda execution space
	template<>
		struct ComputePairTraits< exaDEM::HookeForceOp >
		{
			static inline constexpr bool RequiresBlockSynchronousCall = false;
			static inline constexpr bool ComputeBufferCompatible = false;
			static inline constexpr bool BufferLessCompatible = true;
			static inline constexpr bool CudaCompatible = true;
		};
}

namespace exaDEM
{
	using namespace exanb;

	template<
		class GridT,
					class = AssertGridHasFields< GridT, field::_radius, field::_vx,field::_vy,field::_vz, field::_mass, field::_fx ,field::_fy,field::_fz, field::_vrot, field::_mom, field::_homothety >
						>
						class HookeForce : public OperatorNode
						{
							// ========= I/O slots =======================
							ADD_SLOT( HookeParams           , config            , INPUT , REQUIRED );
							ADD_SLOT( double                , rcut_max          , INPUT_OUTPUT , 0.0 );
							ADD_SLOT( exanb::GridChunkNeighbors    , chunk_neighbors   , INPUT , exanb::GridChunkNeighbors{} , DocString{"neighbor list"} );
							ADD_SLOT( GridCellParticleNeigborFriction , nbh_friction , INPUT , GridCellParticleNeigborFriction{}, DocString{"Neighbor particle friction term"} );
							ADD_SLOT( bool                  , ghost             , INPUT , false );
							ADD_SLOT( GridT                 , grid              , INPUT_OUTPUT );
							ADD_SLOT( Domain                , domain            , INPUT , REQUIRED );
							ADD_SLOT( double                , dt                , INPUT , REQUIRED );

							// shortcut to the Compute buffer used (and passed to functor) by compute_pair_singlemat
							static constexpr bool UseNbhId = true;

							// cell particles array type
							using CellParticles = typename GridT::CellParticles;

							// attributes processed during computation
							using ComputeFields = FieldSet< field::_radius, field::_vx,field::_vy,field::_vz, field::_mass, field::_vrot, field::_fx ,field::_fy ,field::_fz, field::_mom, field::_homothety>;
							static constexpr ComputeFields compute_fields {};

							public:

							inline std::string documentation() const override final
							{
								return R"EOF(
        This operator computes forces between spheric particles using the Hooke law.
        )EOF";
							}
							// Operator execution
							inline void execute () override final
							{
								assert( chunk_neighbors->number_of_cells() == grid->number_of_cells() );

								const double rcut = config->rcut;
								*rcut_max = std::max( *rcut_max , rcut );
								if( grid->number_of_cells() == 0 ) { return; }

								ldbg<<"Hooke: rcut="<<rcut<<std::endl;

								ComputePairOptionalLocks<false> cp_locks {};
								exanb::GridChunkNeighborsLightWeightIt<false> nbh_it{ *chunk_neighbors };

								HookeForceOp force_op { *config, *dt };

								auto force_buf = make_default_pair_buffer();
								ParticleNeighborFrictionIterator cp_friction{ nbh_friction->m_cell_friction.data() };
								if( domain->xform_is_identity() )
								{
									auto optional = make_compute_pair_optional_args( nbh_it, cp_friction, NullXForm{}, cp_locks );
									compute_cell_particle_pairs( *grid, rcut, *ghost, optional, force_buf, force_op, compute_fields, DefaultPositionFields{}, parallel_execution_context() );
								}
								else
								{
									auto optional = make_compute_pair_optional_args( nbh_it, cp_friction , LinearXForm{ domain->xform() }, cp_locks );
									compute_cell_particle_pairs( *grid, rcut, *ghost, optional, force_buf, force_op, compute_fields, DefaultPositionFields{}, parallel_execution_context() );
								}
							}

						};

	template<class GridT> using HookeForceTmpl = HookeForce<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{  
		OperatorNodeFactory::instance()->register_factory( "hooke_force" , make_grid_variant_operator< HookeForceTmpl > );
	}

}


