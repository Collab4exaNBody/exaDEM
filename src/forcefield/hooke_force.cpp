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

#include <exaDEM/interactions_PP.h>

namespace exaDEM
{
	using namespace exanb;

	// Reaction Field Compute functor
	struct alignas(onika::memory::DEFAULT_ALIGNMENT) HookeForceOp 
	{
		// poetential parameters
		const HookeParams m_params;
		const double m_dt;
		//Domain domain;
		Mat3d xform;

		template<class CellParticlesT>
			ONIKA_HOST_DEVICE_FUNC inline void operator ()
			(      
			 const Vec3d& dr,   // imposes par la forme sans compute_buffer
			 double d2,         //
			 double rx, 
			 double ry, 
			 double rz,
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
				
				const double rx_nbh = cells[cell_b][field::rx][p_b];
				const double ry_nbh = cells[cell_b][field::ry][p_b];
				const double rz_nbh = cells[cell_b][field::rz][p_b];
				
				//Vec3d p1 = {rx, ry, rz};
				//Vec3d p1_transform = xform * p1;
				
				//Vec3d p2 = {rx_nbh, ry_nbh, rz_nbh};
				//Vec3d p2_transform = xform * p2;
				
				const double dr2x = rx_nbh - rx;
				const double dr2y = ry_nbh - ry;
				const double dr2z = rz_nbh - rz;
				Vec3d dr2 = {dr2x, dr2y, dr2z};
				Vec3d dr_transform = xform * dr2;
				//Vec3d dr_transform = p2_transform - p1_transform;
				//double drx = dr_transform.x;
				//double dry = dr_transform.y;
				//double drz = dr_transform.z;
				
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
						//rx, ry, rz,
						//rx_nbh, ry_nbh, rz_nbh,         
						vx, vy, vz,           
						mass, radius,     
						fx, fy, fz,     
						mom, vrot, 
						drx, dry, drz,
						//rx_nbh, ry_nbh, rz_nbh,
						//rx, ry, rz, 
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
	
	
	template< class GridT > __global__ void HookeForceGPU(GridT* cells, HookeParams hp, double dt, Mat3d xform, int size,
								int* pa_GPU,
								int* cella_GPU,
								int* pb_GPU,
								int* cellb_GPU,
								double* ftx_GPU,
								double* fty_GPU,
								double* ftz_GPU
								)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(idx < size)
		{
			
			int pa = pa_GPU[idx];
			int cella = cella_GPU[idx];
			int pb = pb_GPU[idx];
			int cellb = cellb_GPU[idx];
			double ftx = ftx_GPU[idx];
			double fty = fty_GPU[idx];
			double ftz = ftz_GPU[idx];
			Vec3d ft = {ftx, fty, ftz}; 
	
			
			double rx = cells[cella][field::rx][pa];
			double ry = cells[cella][field::ry][pa];
			double rz = cells[cella][field::rz][pa];
			double vx = cells[cella][field::vx][pa];
			double vy = cells[cella][field::vy][pa];
			double vz = cells[cella][field::vz][pa];
			double mass = cells[cella][field::mass][pa];
			double radius = cells[cella][field::radius][pa];
			double fx = 0;
			double fy = 0;
			double fz = 0;
			Vec3d mom = {0., 0., 0.};
			Vec3d vrot = cells[cella][field::vrot][pa];

					
			double rx_nbh = cells[cellb][field::rx][pb];
			double ry_nbh = cells[cellb][field::ry][pb];
			double rz_nbh = cells[cellb][field::rz][pb];
			double vx_nbh = cells[cellb][field::vx][pb];
			double vy_nbh = cells[cellb][field::vy][pb];
			double vz_nbh = cells[cellb][field::vz][pb];
			double mass_nbh = cells[cellb][field::mass][pb];
			double radius_nbh = cells[cellb][field::radius][pb];
			Vec3d vrot_nbh = cells[cellb][field::vrot][pb];
					
			double fx_nbh(0.0), fy_nbh(0.0), fz_nbh(0.0);
			Vec3d mom_nbh = {0,0,0};
			Vec3d dr = xform * Vec3d{rx_nbh - rx, ry_nbh - ry, rz_nbh - rz};

			exaDEM::compute_hooke_force(
				hp.dncut, dt, hp.m_kn, hp.m_kt, hp.m_kr,
				hp.m_fc, hp.m_mu, hp.m_damp_rate,
				ft,
				0., 0., 0.,
				vx, vy, vz,
				mass, radius,
				fx, fy, fz,
				mom, vrot,
				dr.x, dr.y, dr.z,
				vx_nbh, vy_nbh, vz_nbh,
				mass_nbh, radius_nbh,
				fx_nbh, fy_nbh, fz_nbh,
				mom_nbh, vrot_nbh
			);
			
			ftx_GPU[idx] = ft.x;
			fty_GPU[idx] = ft.y;
			ftz_GPU[idx] = ft.z;

			atomicAdd(&cells[cella][field::fx][pa], fx);
			atomicAdd(&cells[cella][field::fy][pa], fy);
			atomicAdd(&cells[cella][field::fz][pa], fz);
			atomicAdd(&cells[cella][field::mom][pa].x, mom.x);
			atomicAdd(&cells[cella][field::mom][pa].y, mom.y);
			atomicAdd(&cells[cella][field::mom][pa].z, mom.z);
			
			atomicAdd(&cells[cellb][field::fx][pb], -fx);
			atomicAdd(&cells[cellb][field::fy][pb], -fy);
			atomicAdd(&cells[cellb][field::fz][pb], -fz);
			atomicAdd(&cells[cellb][field::mom][pb].x, mom.x);
			atomicAdd(&cells[cellb][field::mom][pb].y, mom.y);
			atomicAdd(&cells[cellb][field::mom][pb].z, mom.z);
		}
	}
	
	template< class GridT > __global__ void HookeForceGPU2(GridT* cells, HookeParams hp, double dt, Mat3d xform,
								int* bloc_nb_warps,
								int* warp1_p,
								int* warp1_cell,
								int* warp1_size,
								int* warp1_start,
								int* warp2_p,
								int* warp2_cell,
								int* warp2_size,
								int* warp2_start,
								int* warp3_p,
								int* warp3_cell,
								int* warp3_size,
								int* warp3_start,
								int* warp4_p,
								int* warp4_cell,
								int* warp4_size,
								int* warp4_start,
								int* warp5_p,
								int* warp5_cell,
								int* warp5_size,
								int* warp5_start,
								int* warp6_p,
								int* warp6_cell,
								int* warp6_size,
								int* warp6_start,
								int* warp7_p,
								int* warp7_cell,
								int* warp7_size,
								int* warp7_start,
								int* warp8_p,
								int* warp8_cell,
								int* warp8_size,
								int* warp8_start,
								int* pb2,
								int* cellb2,
								double* ftx2,
								double* fty2,
								double* ftz2)
	{
		int nb_warps = bloc_nb_warps[blockIdx.x];
		
		int warp = (int)(threadIdx.x / 32);
		
		int particule = threadIdx.x - (warp * 32);
		
		int cella;
		int pa;
		
		int cellb;
		int pb;
		
		int size;
		
		int start;
		
		if(warp < nb_warps)
		{
		
		if(warp == 0)
		{
			cella = warp1_cell[blockIdx.x];
			pa = warp1_p[blockIdx.x];
			size = warp1_size[blockIdx.x];
			
			start = warp1_start[blockIdx.x];
			
			//pb = pb2[start + particule];
			//cb = cellb2[start + particule];
		}
		else if(warp == 1)
		{
			cella = warp2_cell[blockIdx.x];
			pa = warp2_p[blockIdx.x];
			size = warp2_size[blockIdx.x];
			
			start = warp3_start[blockIdx.x];
			
			//pb = pb2[start + particule];
			//cb = cellb2[start + particule];
		}
		else if(warp == 2)
		{
			cella = warp3_cell[blockIdx.x];
			pa = warp3_p[blockIdx.x];
			size = warp3_size[blockIdx.x];
			
			start = warp1_start[blockIdx.x];
			
			//pb = pb2[start + particule];
			//cb = cellb2[start + particule];
		}
		else if(warp == 3)
		{
			cella = warp4_cell[blockIdx.x];
			pa = warp4_p[blockIdx.x];
			size = warp4_size[blockIdx.x];
			
			start = warp4_start[blockIdx.x];
			
			//pb = pb2[start + particule];
			//cb = cellb2[start + particule];
		}
		else if(warp == 4)
		{
			cella = warp5_cell[blockIdx.x];
			pa = warp5_p[blockIdx.x];
			size = warp5_size[blockIdx.x];
			
			start = warp5_start[blockIdx.x];
			
			//pb = pb2[start + particule];
			//cb = cellb2[start + particule];
		}
		else if(warp == 5)
		{
			cella = warp6_cell[blockIdx.x];
			pa = warp6_p[blockIdx.x];
			size = warp6_size[blockIdx.x];
			
			start = warp6_start[blockIdx.x];
			
			//pb = pb2[start + particule];
			//cb = cellb2[start + particule];
		}
		else if(warp == 6)
		{
			cella = warp7_cell[blockIdx.x];
			pa = warp7_p[blockIdx.x];
			size = warp7_size[blockIdx.x];
			
			start = warp7_start[blockIdx.x];
			
			//pb = pb2[start + particule];
			//cb = cellb2[start + particule];
		}
		else if(warp == 7)
		{
			cella = warp8_cell[blockIdx.x];
			pa = warp8_p[blockIdx.x];
			size = warp8_size[blockIdx.x];
			
			start = warp8_start[blockIdx.x];
			
			//pb = pb2[start + particule];
			//cb = cellb2[start + particule];
		}
		
		if(particule < size)
		{
			int pb = pb2[start+particule];
			int cellb = cellb2[start+particule];
			double ftx = ftx2[start+particule];
			double fty = fty2[start+particule];
			double ftz = ftz2[start+particule];
			
			Vec3d ft = {ftx, fty, ftz}; 
	
			
			double rx = cells[cella][field::rx][pa];
			double ry = cells[cella][field::ry][pa];
			double rz = cells[cella][field::rz][pa];
			double vx = cells[cella][field::vx][pa];
			double vy = cells[cella][field::vy][pa];
			double vz = cells[cella][field::vz][pa];
			double mass = cells[cella][field::mass][pa];
			double radius = cells[cella][field::radius][pa];
			double fx = 0;
			double fy = 0;
			double fz = 0;
			Vec3d mom = {0., 0., 0.};
			Vec3d vrot = cells[cella][field::vrot][pa];

					
			double rx_nbh = cells[cellb][field::rx][pb];
			double ry_nbh = cells[cellb][field::ry][pb];
			double rz_nbh = cells[cellb][field::rz][pb];
			double vx_nbh = cells[cellb][field::vx][pb];
			double vy_nbh = cells[cellb][field::vy][pb];
			double vz_nbh = cells[cellb][field::vz][pb];
			double mass_nbh = cells[cellb][field::mass][pb];
			double radius_nbh = cells[cellb][field::radius][pb];
			Vec3d vrot_nbh = cells[cellb][field::vrot][pb];
					
			double fx_nbh(0.0), fy_nbh(0.0), fz_nbh(0.0);
			Vec3d mom_nbh = {0,0,0};
			Vec3d dr = xform * Vec3d{rx_nbh - rx, ry_nbh - ry, rz_nbh - rz};

			exaDEM::compute_hooke_force(
				hp.dncut, dt, hp.m_kn, hp.m_kt, hp.m_kr,
				hp.m_fc, hp.m_mu, hp.m_damp_rate,
				ft,
				0., 0., 0.,
				vx, vy, vz,
				mass, radius,
				fx, fy, fz,
				mom, vrot,
				dr.x, dr.y, dr.z,
				vx_nbh, vy_nbh, vz_nbh,
				mass_nbh, radius_nbh,
				fx_nbh, fy_nbh, fz_nbh,
				mom_nbh, vrot_nbh
			);
			
			ftx2[start+particule] = ft.x;
			fty2[start+particule] = ft.y;
			ftz2[start+particule] = ft.z;

			atomicAdd(&cells[cella][field::fx][pa], fx);
			atomicAdd(&cells[cella][field::fy][pa], fy);
			atomicAdd(&cells[cella][field::fz][pa], fz);
			atomicAdd(&cells[cella][field::mom][pa].x, mom.x);
			atomicAdd(&cells[cella][field::mom][pa].y, mom.y);
			atomicAdd(&cells[cella][field::mom][pa].z, mom.z);
			
			atomicAdd(&cells[cellb][field::fx][pb], -fx);
			atomicAdd(&cells[cellb][field::fy][pb], -fy);
			atomicAdd(&cells[cellb][field::fz][pb], -fz);
			atomicAdd(&cells[cellb][field::mom][pb].x, mom.x);
			atomicAdd(&cells[cellb][field::mom][pb].y, mom.y);
			atomicAdd(&cells[cellb][field::mom][pb].z, mom.z);
			
		}
		}
		
		 
		
	}							
								
								
								
	template<
		class GridT,
					class = AssertGridHasFields< GridT, field::_rx, field::_ry, field::_rz, field::_radius, field::_vx,field::_vy,field::_vz, field::_mass, field::_fx ,field::_fy,field::_fz, field::_vrot, field::_mom, field::_homothety >
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
							
							ADD_SLOT( Interactions_PP       , interactions_PP   , INPUT_OUTPUT );//HOOKE_FORCE_GPU
							

							// shortcut to the Compute buffer used (and passed to functor) by compute_pair_singlemat
							static constexpr bool UseNbhId = true;

							// cell particles array type
							using CellParticles = typename GridT::CellParticles;

							// attributes processed during computation
							using ComputeFields = FieldSet< field::_rx, field::_ry, field::_rz, field::_radius, field::_vx,field::_vy,field::_vz, field::_mass, field::_vrot, field::_fx ,field::_fy ,field::_fz, field::_mom, field::_homothety>;
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
								printf("HOOKE_FORCE\n");
								
								assert( chunk_neighbors->number_of_cells() == grid->number_of_cells() );
								
								Interactions_PP& ints= *interactions_PP;//HOOKE_FORCE_GPU
								
								
								//HOOKE_FORCE_GPU
								auto& g = *grid;
								const auto cells = g.cells();
								const auto cells2 = g.cells();
								//HOOKE_FORCE_GPU

								const double rcut = config->rcut;
								*rcut_max = std::max( *rcut_max , rcut );
								if( grid->number_of_cells() == 0 ) { return; }

								ldbg<<"Hooke: rcut="<<rcut<<std::endl;

								ComputePairOptionalLocks<false> cp_locks {};
								exanb::GridChunkNeighborsLightWeightIt<false> nbh_it{ *chunk_neighbors };
								
								
								HookeForceOp force_op { *config, *dt, domain->xform() };

								auto force_buf = make_default_pair_buffer();
								ParticleNeighborFrictionIterator cp_friction{ nbh_friction->m_cell_friction.data() };
								
								//HOOKE_FORCE_GPU
								int size = ints.nb_interactions;
								int blockSize = 256;
								int numBlocks;// = ints.nb_blocs;
								if(size % blockSize == 0){ numBlocks = size/blockSize;}
								else if(size / blockSize < 1){ numBlocks=1; blockSize = size;}
								else { numBlocks = int(size/blockSize)+1; }
								
								onika::memory::CudaMMVector<double> fx;
								fx.resize(1);
								
								HookeForceGPU<<<numBlocks, blockSize>>>(cells, *config, *dt, domain->xform(), size, ints.pa_GPU.data(), ints.cella_GPU.data(), ints.pb_GPU.data(), ints.cellb_GPU.data(), ints.ftx_GPU.data(), ints.fty_GPU.data(), ints.ftz_GPU2.data());			
								
								//printf("HOOKE END\n");
								
								/*HookeForceGPU2<<<numBlocks, blockSize>>>(cells, *config, *dt, domain->xform(),
													 ints.bloc_nb_warps.data(),
								ints.warp1_p.data(),
								ints.warp1_cell.data(),
								ints.warp1_size.data(),
								ints.warp1_start.data(),
								ints.warp2_p.data(),
								ints.warp2_cell.data(),
								ints.warp2_size.data(),
								ints.warp2_start.data(),
								ints.warp3_p.data(),
								ints.warp3_cell.data(),
								ints.warp3_size.data(),
								ints.warp3_start.data(),
								ints.warp4_p.data(),
								ints.warp4_cell.data(),
								ints.warp4_size.data(),
								ints.warp4_start.data(),
								ints.warp5_p.data(),
								ints.warp5_cell.data(),
								ints.warp5_size.data(),
								ints.warp5_start.data(),
								ints.warp6_p.data(),
								ints.warp6_cell.data(),
								ints.warp6_size.data(),
								ints.warp6_start.data(),
								ints.warp7_p.data(),
								ints.warp7_cell.data(),
								ints.warp7_size.data(),
								ints.warp7_start.data(),
								ints.warp8_p.data(),
								ints.warp8_cell.data(),
								ints.warp8_size.data(),
								ints.warp8_start.data(),
								ints.pb_GPU3.data(),
								ints.cellb_GPU3.data(),
								ints.ftx3.data(),
								ints.fty3.data(),
								ints.ftz3.data());	*/		

								//sgetchar();
								
								//HOOKE_FORCE_GPU
								
								/*if( domain->xform_is_identity() )
								{
									//printf("NULLXFORM\n");
									auto optional = make_compute_pair_optional_args( nbh_it, cp_friction, NullXForm{}, cp_locks );
									compute_cell_particle_pairs( *grid, rcut, *ghost, optional, force_buf, force_op, compute_fields, DefaultPositionFields{}, parallel_execution_context() );
								}
								else
								{
									//printf("LINEARXFORM\n");
									auto optional = make_compute_pair_optional_args( nbh_it, cp_friction , LinearXForm{ domain->xform() }, cp_locks );
									compute_cell_particle_pairs( *grid, rcut, *ghost, optional, force_buf, force_op, compute_fields, DefaultPositionFields{}, parallel_execution_context() );
								}*/
								
								printf("HOOKE FORCE END\n");
								
							}

						};

	template<class GridT> using HookeForceTmpl = HookeForce<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{  
		OperatorNodeFactory::instance()->register_factory( "hooke_force" , make_grid_variant_operator< HookeForceTmpl > );
	}

}


