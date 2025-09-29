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
#include <memory>
#include <cub/cub.cuh>  

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>

#include <exaDEM/vertices.hpp>

#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction_manager.hpp>
#include <exaDEM/interaction/migration_test.hpp>
#include <exaDEM/drivers.h>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/shape_detection_driver.hpp>
#include <exaDEM/traversal.h>
#include <exaDEM/nbh_polyhedron.h>
#include <exaDEM/nbh_polyhedron_block_pair.h>
#include <exaDEM/nbh_polyhedron_block.h>
#include <exaDEM/nbh_polyhedron_particle.h>
#include <exaDEM/nbh_polyhedron_pair.h>

#include <cassert>

namespace exaDEM
{
	using namespace exanb;
	//using VertexArray = ::onika::oarray_t<::exanb::Vec3d, EXADEM_MAX_VERTICES>;
	
	struct Interaction_history
	{
		//onika::memory::CudaMMVector<int> interaction_id;
		int* interaction_id;
		
		//onika::memory::CudaMMVector<double> ft_x;
		double* ft_x;
		//onika::memory::CudaMMVector<double> ft_y;
		double* ft_y;
		//onika::memory::CudaMMVector<double> ft_z;
		double* ft_z;
		
		//onika::memory::CudaMMVector<double> mom_x;
		double* mom_x;
		//onika::memory::CudaMMVector<double> mom_y;
		double* mom_y;
		//onika::memory::CudaMMVector<double> mom_z;
		double* mom_z;
		
		//onika::memory::CudaMMVector<int> indices;
		int* indices;
		
		int size;
	};
	
	struct Classifier_history
	{
		onika::memory::CudaMMVector<Interaction_history> waves;
	};
	
	/*__device__*/ONIKA_HOST_DEVICE_FUNC int encode (int a, int b, int c, int d,
				int max_b, int max_c, int max_d, /*particle_info p_a, particle_info p_b,*/ int type )
	{
		//int max_c;
		//int max_d;
		
		/*if(type == 0)
		{
			max_c = p_a.shp->get_number_of_vertices();
			max_d = p_b.shp->get_number_of_vertices();
		}
		else if(type == 1)
		{
			max_c = p_a.shp->get_number_of_vertices();
			max_d = p_b.shp->get_number_of_edges();
		}
		else if(type == 2)
		{
			max_c = p_a.shp->get_number_of_vertices();
			max_d = p_b.shp->get_number_of_faces();
		}
		else
		{
			max_c = p_a.shp->get_number_of_edges();
			max_d = p_b.shp->get_number_of_edges();
		}*/
		
		return ((a * max_b + b) * max_c + c) * max_d + d;
	}
	
	template < class GridT > __global__ void encodeClassifier( GridT* cells,
				uint32_t* cell_i,
				uint32_t* cell_j,
				uint16_t* p_i,
				uint16_t* p_j,
				uint64_t* id_i,
				uint64_t* id_j,
				uint16_t* sub_i,
				uint16_t* sub_j,
				shapes shps,
				int max_b,
				int max_c,
				int max_d,
				int* res,
				int* indices,
				int type,
				int size)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(idx < size)
		{
			/*int cell_a = cell_i[idx];
			cell_accessors cellA(cells[cell_a]);
			
			int cell_b = cell_j[idx];
			cell_accessors cellB(cells[cell_b]);
			
			int p_a = p_i[idx];
			particle_info p(shps, p_a, cellA);
			
			int p_b = p_j[idx];
			particle_info p_nbh(shps, p_b, cellB);*/
			
			res[idx] = encode(id_i[idx], id_j[idx], sub_i[idx], sub_j[idx], max_b, max_c, max_d, /*p, p_nbh,*/ type);
			indices[idx] = idx;
		}
	}
	
	__global__ void search_active_interactions(
			double* ft_x,
			double* ft_y,
			double* ft_z,
			double* mom_x,
			double* mom_y,
			double* mom_z,
			int* nb,
			int size)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(idx < size)
		{
			if( ft_x[idx] != 0 || ft_y[idx] != 0 || ft_z[idx] != 0 || mom_x[idx] != 0 || mom_y[idx] != 0 || mom_z[idx] != 0)
			{
				atomicAdd(&nb[blockIdx.x], 1);
			}
		}
	}
	
	__global__ void search_active_interactions2(
			double* ft_x,
			double* ft_y,
			double* ft_z,
			double* mom_x,
			double* mom_y,
			double* mom_z,
			int* nb,
			int size)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(idx < size)
		{
			if( ft_x[idx] != 0 || ft_y[idx] != 0 || ft_z[idx] != 0 || mom_x[idx] != 0 || mom_y[idx] != 0 || mom_z[idx] != 0)
			{
				atomicAdd(&nb[0], 1);
			}
		}
	}
	
	template< class GridT > __global__ void fill_active_interactions( GridT* cells,
	                uint32_t* cell_i,
	                uint32_t* cell_j,
	                uint16_t* p_i,
	                uint16_t* p_j,
			double* ft_x,
			double* ft_y,
			double* ft_z,
			double* mom_x,
			double* mom_y,
			double* mom_z,
			uint64_t* id_i,
			uint64_t* id_j,
			uint16_t* sub_i,
			uint16_t* sub_j,
			shapes shps,
			int max_b,
			int max_c,
			int max_d,
			int* nb_incr,
			int* interaction_id,
			double* ftx,
			double* fty,
			double* ftz,
			double* momx,
			double* momy,
			double* momz,
			int* indices,
			int type,
			int size)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(idx < size)
		{
			__shared__ int s[256];
			s[threadIdx.x] = 0;
			__syncthreads();
			
			bool active = false;
			
			if( ft_x[idx] != 0 || ft_y[idx] != 0 || ft_z[idx] != 0 || mom_x[idx] != 0 || mom_y[idx] != 0 || mom_z[idx] != 0)
			{
				s[threadIdx.x] = 1;
				active = true;
			}
			__syncthreads();
			
			int prefix = 0;
			for(int i = 0; i < threadIdx.x; i++)
			{
				prefix+= s[i];
			}
			prefix+= nb_incr[blockIdx.x];
			
			/*int cell_a = cell_i[idx];
			cell_accessors cellA(cells[cell_a]);
			
			int cell_b = cell_j[idx];
			cell_accessors cellB(cells[cell_b]);
			
			int p_a = p_i[idx];
			particle_info p(shps, p_a, cellA);
			
			int p_b = p_j[idx];
			particle_info p_nbh(shps, p_b, cellB);*/
			
			if(active)
			{
				interaction_id[prefix] = encode(id_i[idx], id_j[idx], sub_i[idx], sub_j[idx], max_b, max_c, max_d, /*p, p_nbh,*/ type);
				ftx[prefix] = ft_x[idx];
				fty[prefix] = ft_y[idx];
				ftz[prefix] = ft_z[idx];
				momx[prefix] = mom_x[idx];
				momy[prefix] = mom_y[idx];
				momz[prefix] = mom_z[idx];
				indices[prefix] = prefix;
			}
		}
	}
	
	__global__ void find_common_elements(
			int* keys_new,
			int* keys_old,
			int* indices_new,
			int* indices_old,
			double* ftx,
			double* fty,
			double* ftz,
			double* momx,
			double* momy,
			double* momz,
			double* ftx_old,
			double* fty_old,
			double* ftz_old,
			double* momx_old,
			double* momy_old,
			double* momz_old,
			int size_new,
			int size_old)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(idx < size_new)
		{
			int key = keys_new[idx];
			int low = 0, high = size_old - 1;
			while(low<=high)
			{
				int mid = low + (high - low) / 2;
				if( keys_old[mid] == key )
				{
					int index = indices_new[idx];
					int index_old = indices_old[mid];
					
					ftx[index] = ftx_old[index_old];
					fty[index] = fty_old[index_old];
					ftz[index] = ftz_old[index_old];
					
					momx[index] = momx_old[index_old];
					momy[index] = momy_old[index_old];
					momz[index] = momz_old[index_old];
					
					return;
				}
				else if( keys_old[mid] < key )
				{
					low = mid + 1;
				}
				else
				{
					high = mid - 1;
				}
			}
		}
	}

	template <typename GridT, class = AssertGridHasFields<GridT>> class UpdateGridCellInteractionGPU : public OperatorNode
	{
		using ComputeFields = FieldSet<>;
		static constexpr ComputeFields compute_field_set{};

		ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
		ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
		ADD_SLOT(Domain , domain, INPUT , REQUIRED );
		ADD_SLOT(exanb::GridChunkNeighbors, chunk_neighbors, INPUT, OPTIONAL, DocString{"Neighbor list"});
		ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
		ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
		ADD_SLOT(shapes, shapes_collection, INPUT, DocString{"Collection of shapes"});
		ADD_SLOT(double, rcut_inc, INPUT_OUTPUT, DocString{"value added to the search distance to update neighbor list less frequently. in physical space"});
		ADD_SLOT(Drivers, drivers, INPUT, DocString{"List of Drivers"});
		ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});
		
		    ADD_SLOT(InteractionSOA, interactions_intermediaire, INPUT_OUTPUT);

    using VectorTypes = onika::memory::CudaMMVector<NumberOfPolyhedronInteractionPerTypes>;

    ADD_SLOT(bool, block_pair_version, false, PRIVATE);
    ADD_SLOT(bool, block_version, false, PRIVATE);
    ADD_SLOT(bool, pair_version, false, PRIVATE);
    ADD_SLOT(bool, particle_version, false, PRIVATE);
    ADD_SLOT(VectorTypes, nbOfInteractionsCell, PRIVATE);
    ADD_SLOT(VectorTypes, prefixInteractionsCell, PRIVATE);
    
	ADD_SLOT(Interactions_intermediaire, interactions_inter, INPUT_OUTPUT);
	//ADD_SLOT(InteractionSOA, interactions_inter, INPUT_OUTPUT);	
	
	ADD_SLOT(InteractionSOA2, interactions_type0, INPUT_OUTPUT);
	ADD_SLOT(InteractionSOA2, interactions_type1, INPUT_OUTPUT);
	ADD_SLOT(InteractionSOA2, interactions_type2, INPUT_OUTPUT);
	ADD_SLOT(InteractionSOA2, interactions_type3, INPUT_OUTPUT);
	
	ADD_SLOT(Classifier2, ic2, INPUT_OUTPUT);	

		public:
		inline std::string documentation() const override final
		{
			return R"EOF(
                      This function builds the list of interactions per particle (polyhedron). Interactions are between two particles or a particle and a driver. In this function, frictions and moments are updated if the interactions are still actived. Note that, a list of non-empty cells is built during this function.
                )EOF";
		}

    template <typename Func> 
      void add_driver_interaction(
          Stl_mesh &mesh, 
          size_t cell_a, 
          Func &add_contact, 
          Interaction &item, 
          const size_t n_particles, 
          const double rVerlet, 
          const ParticleTypeInt *__restrict__ type, 
          const uint64_t *__restrict__ id, 
          const double *__restrict__ rx, 
          const double *__restrict__ ry, 
          const double *__restrict__ rz, 
          VertexField& vertices,
          const exanb::Quaternion *__restrict__ orient, 
          shapes &shps)
      {
#define __particle__ vertices_i, i, shpi
#define __driver__ mesh.vertices.data(), idx, &mesh.shp
        assert(cell_a < mesh.grid_indexes.size());
        auto &list = mesh.grid_indexes[cell_a];
        const size_t stl_nv = list.vertices.size();
        const size_t stl_ne = list.edges.size();
        const size_t stl_nf = list.faces.size();

        if (stl_nv == 0 && stl_ne == 0 && stl_nf == 0)
          return;

        for (size_t p = 0; p < n_particles; p++)
        {
          Vec3d r = {rx[p], ry[p], rz[p]}; // position
          ParticleVertexView vertices_i = {p, vertices};
          const Quaternion& orient_i = orient[p];
          item.p_i = p;
          item.id_i = id[p];
          auto ti = type[p];
          const shape *shpi = shps[ti];
          const size_t nv = shpi->get_number_of_vertices();
          const size_t ne = shpi->get_number_of_edges();
          const size_t nf = shpi->get_number_of_faces();

          // Get OBB from stl mesh
          auto &stl_shp = mesh.shp;
          OBB *__restrict__ stl_obb_vertices = onika::cuda::vector_data(stl_shp.m_obb_vertices);
          [[maybe_unused]] OBB *__restrict__ stl_obb_edges = onika::cuda::vector_data(stl_shp.m_obb_edges);
          [[maybe_unused]] OBB *__restrict__ stl_obb_faces = onika::cuda::vector_data(stl_shp.m_obb_faces);

          // compute OBB from particle p
          OBB obb_i = shpi->obb;
          quat conv_orient_i = quat{vec3r{orient_i.x, orient_i.y, orient_i.z}, orient_i.w};
          obb_i.rotate(conv_orient_i);
          obb_i.translate(vec3r{r.x, r.y, r.z});
          obb_i.enlarge(rVerlet);

          // Note:
          // loop i = particle p
          // loop j = stl mesh
          for (size_t i = 0; i < nv; i++)
          {
            vec3r v = conv_to_vec3r(vertices_i[i]);
            OBB obb_v_i;
            obb_v_i.center = v; 
            obb_v_i.enlarge(rVerlet + shpi->m_radius);

            // vertex - vertex
            item.type = 7;
            item.sub_i = i;
            for (size_t j = 0; j < stl_nv; j++)
            {
              size_t idx = list.vertices[j];
              if(filter_vertex_vertex_v2(rVerlet, __particle__, __driver__))
                //if(filter_vertex_vertex(rVerlet, __particle__, __driver__))
              {
                add_contact(p, item, i, idx);
              } 
            }
            // vertex - edge
            item.type = 8;
            for (size_t j = 0; j < stl_ne; j++)
            {
              size_t idx = list.edges[j];
              if(filter_vertex_edge(rVerlet, __particle__, __driver__))
              {
                add_contact(p, item, i, idx);
              }
            }
            // vertex - face
            item.type = 9;
            for (size_t j = 0; j < stl_nf; j++)
            {
              size_t idx = list.faces[j];
              const OBB& obb_f_stl_j = stl_obb_faces[idx];
              if( obb_f_stl_j.intersect(obb_v_i) )
              {
                if(filter_vertex_face(rVerlet, __particle__, __driver__))
                {
                  add_contact(p, item, i, idx);
                }
              }
            }
          }

          for (size_t i = 0; i < ne; i++)
          {
            item.type = 10;
            item.sub_i = i;
            // edge - edge
            for (size_t j = 0; j < stl_ne; j++)
            {
              const size_t idx = list.edges[j];
              if(filter_edge_edge(rVerlet, __particle__, __driver__))
              {
                add_contact(p, item, i, idx);
              }
            }
          }

          for (size_t j = 0; j < stl_nv; j++)
          {
            const size_t idx = list.vertices[j];

            // rejects vertices that are too far from the stl mesh.
            const OBB& obb_v_stl_j = stl_obb_vertices[idx];
            if( !obb_v_stl_j.intersect(obb_i)) continue;

            item.type = 11;
            // edge - vertex
            for (size_t i = 0; i < ne; i++)
            {
              if(filter_vertex_edge(rVerlet, __driver__, __particle__)) 
              {
                add_contact(p, item, i, idx);
              }
            }
            // face vertex
            item.type = 12;
            for (size_t i = 0; i < nf; i++)
            {
              if(filter_vertex_face(rVerlet, __driver__, __particle__))
              {
                add_contact(p, item, i, idx);
              }
            }
          }
        } // end loop p
#undef __particle__
#undef __driver__
      } // end funcion

    template <typename D, typename Func> 
      void add_driver_interaction(
          D &driver, 
          Func &add_contact, 
          Interaction &item, 
          const size_t n_particles, 
          const double rVerlet, 
          const ParticleTypeInt *__restrict__ type, 
          const uint64_t *__restrict__ id, 
          VertexField& vertices, 
          shapes &shps)
      {
        for (size_t p = 0; p < n_particles; p++)
        {
          ParticleVertexView va = {p, vertices};
          const shape *shp = shps[type[p]];
          int nv = shp->get_number_of_vertices();
          for (int sub = 0; sub < nv; sub++)
          {
            bool contact = exaDEM::filter_vertex_driver(driver, rVerlet, va, sub, shp);
            if (contact)
            {
              item.p_i = p;
              item.id_i = id[p];
              add_contact(p, item, sub, -1);
            }
          }
        }
      }

		inline void execute() override final
		{
			GridT& g = *grid;
			auto* vertex_fields = cvf->data();
			const auto cells = g.cells();
			const size_t n_cells = g.number_of_cells(); // nbh.size();
			const IJK dims = g.dimension();
			auto &interactions = ges->m_data;
			auto &shps = *shapes_collection;
			auto &classifier = *ic; 
			double rVerlet = *rcut_inc;
			Mat3d xform = domain->xform();
			bool is_xform = !domain->xform_is_identity();
			if (drivers.has_value() && is_xform)
			{
				if(drivers->get_size() > 0)
				{
//					lout<< "Error: Contact detection with drivers is deactivated when the simulation box is deformed." << std::endl;
//					std::exit(0);
				}
			}

			//lout << "start nbh_polyhedron_gpu" << std::endl;

			// if grid structure (dimensions) changed, we invalidate thie whole data
			if (interactions.size() != n_cells)
			{
				ldbg << "number of cells has changed, reset friction data" << std::endl;
				interactions.clear();
				interactions.resize(n_cells);
			}

			assert(interactions.size() == n_cells);

			if (!chunk_neighbors.has_value())
			{
#       pragma omp parallel for schedule(static)
				for (size_t i = 0; i < n_cells; i++)
					interactions[i].initialize(0);
				return;
			}

			auto [cell_ptr, cell_size] = traversal_real->info();
			
	auto& interactions2 = *interactions_inter;
	//auto size_interactions = interactions2.size();
	auto size_interactions = interactions2.size;
	
	
      // declare n int and prefix
      auto& number_of_interactions_cell = *nbOfInteractionsCell;
      auto& prefix_interactions_cell = *prefixInteractionsCell;

      if(*block_pair_version)
      {
      			  number_of_interactions_cell.resize(size_interactions);
      			  prefix_interactions_cell.resize(size_interactions);
      }
       else if(number_of_interactions_cell.size() != cell_size)
      {
			  number_of_interactions_cell.resize(cell_size);
			  prefix_interactions_cell.resize(cell_size);
			}
			
			if( *block_pair_version )
			{
	//lout << "NBH polyhedron Block pair version" << std::endl;
				//lout << "Start get_number_of_interactions_block_pair ..." << std::endl;
	/** Define cuda block and grid size*/
				constexpr int block_x = 8;
				constexpr int block_y = 8;
				dim3 BlockSize(block_x, block_y, 1);
				dim3 GridSize(size_interactions, 1, 1);
				
				auto& classifier2 = *ic2;
				
				if(!classifier2.use)
				{
					classifier2.initialize();
				}
				
				Classifier_history update;
				
				if(classifier2.use)
				{
				update.waves.resize(4);			
				
				for(int i = 0; i < 4; i++)
				{
					//if( i != 2)
					//{
					onika::memory::CudaMMVector<int> nb_history;
					
					auto& data = classifier2.waves[i];
					
					int nbBlocks = ( data.size() + 256 - 1) / 256;
					
					nb_history.resize(nbBlocks);
					
					search_active_interactions<<<nbBlocks, 256>>>( data.ft_x, data.ft_y, data.ft_z, data.mom_x, data.mom_y, data.mom_z, nb_history.data(), data.size() );
					
					void* d_temp_storage = nullptr;
					size_t temp_storage_bytes = 0;
	
					onika::memory::CudaMMVector<int> nb_history_incr;
					nb_history_incr.resize( nbBlocks);
	
					cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_history.data(), nb_history_incr.data(), nbBlocks );
	
					cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
					cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_history.data(), nb_history_incr.data(), nbBlocks );
	
					cudaFree(d_temp_storage);
					
					int total = nb_history[nbBlocks - 1] + nb_history_incr[nbBlocks - 1];
					
					//printf("TOTAL POUR LE TYPE %d : %d / %d\n", i, total, data.size());
					
					auto& interaction_history = update.waves[i];
					
					//interaction_history.interaction_id.resize(total);
					cudaMalloc(&interaction_history.interaction_id, total * sizeof(int) );
					//interaction_history.ft_x.resize(total);
					cudaMalloc(&interaction_history.ft_x, total * sizeof(double) );
					//interaction_history.ft_y.resize(total);
					cudaMalloc(&interaction_history.ft_y, total * sizeof(double) );
					//interaction_history.ft_z.resize(total);
					cudaMalloc(&interaction_history.ft_z, total * sizeof(double) );
					//interaction_history.mom_x.resize(total);
					cudaMalloc(&interaction_history.mom_x, total * sizeof(double) );
					//interaction_history.mom_y.resize(total);
					cudaMalloc(&interaction_history.mom_y, total * sizeof(double) );
					//interaction_history.mom_z.resize(total);
					cudaMalloc(&interaction_history.mom_z, total * sizeof(double) );
					//interaction_history.indices.resize(total);
					cudaMalloc(&interaction_history.indices, total * sizeof(int));
					interaction_history.size = total;
					
					//onika::memory::CudaMMVector<int> interaction_id;
					int* interaction_id;
					//interaction_id.resize(total);
					cudaMalloc(&interaction_id, total * sizeof(int));
					//onika::memory::CudaMMVector<int> indices;
					int* indices;
					//indices.resize(total);
					cudaMalloc(&indices, total * sizeof(int));
					
					//printf("MALLOC\n");
					
					int max_b = g.number_of_particles();
					int max_c;
					int max_d;
					
					if(i == 0)
					{
						max_c = 20;
						max_d = 20;
						fill_active_interactions<<<nbBlocks, 256>>>( grid->cells(), data.cell_i, data.cell_j, data.p_i, data.p_j, data.ft_x, data.ft_y, data.ft_z, data.mom_x, data.mom_y, data.mom_z, data.id_i, data.id_j, data.sub_i, data.sub_j, shps, max_b, max_c, max_d, nb_history_incr.data(), /*interaction_history.interaction_id*/interaction_id/*.data()*/, interaction_history.ft_x/*.data()*/, interaction_history.ft_y/*.data()*/, interaction_history.ft_z/*.data()*/, interaction_history.mom_x/*.data()*/, interaction_history.mom_y/*.data()*/, interaction_history.mom_z/*.data()*/, indices/*.data()*/, i, data.size());
						
					/*uint32_t* cell_i_cpu = (uint32_t*)malloc(data.size() * sizeof(uint32_t));
					uint32_t* cell_j_cpu = (uint32_t*)malloc(data.size() * sizeof(uint32_t));
				
					uint16_t* p_i_cpu = (uint16_t*)malloc(data.size() * sizeof(uint16_t));
					uint16_t* p_j_cpu = (uint16_t*)malloc(data.size() * sizeof(uint16_t));
					
					double* ftx_cpu = (double*)malloc(data.size() * sizeof(double));
					double* fty_cpu = (double*)malloc(data.size() * sizeof(double));
					double* ftz_cpu = (double*)malloc(data.size() * sizeof(double));
					
					double* momx_cpu = (double*)malloc(data.size() * sizeof(double));
					double* momy_cpu = (double*)malloc(data.size() * sizeof(double));
					double* momz_cpu = (double*)malloc(data.size() * sizeof(double));
					
					uint64_t* id_i_cpu = (uint64_t*)malloc(data.size() * sizeof(uint64_t));
					uint64_t* id_j_cpu = (uint64_t*)malloc(data.size() * sizeof(uint64_t));
					
					uint16_t* sub_i_cpu = (uint16_t*)malloc(data.size() * sizeof(uint16_t));
					uint16_t* sub_j_cpu = (uint16_t*)malloc(data.size() * sizeof(uint16_t));
				
					cudaMemcpy(cell_i_cpu, data.cell_i, data.size() * sizeof(uint32_t)  , cudaMemcpyDeviceToHost);
					cudaMemcpy(cell_j_cpu, data.cell_j, data.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
				
					cudaMemcpy(p_i_cpu, data.p_i, data.size()  * sizeof(uint16_t), cudaMemcpyDeviceToHost);
					cudaMemcpy(p_j_cpu, data.p_j, data.size()  * sizeof(uint16_t), cudaMemcpyDeviceToHost);
					
					cudaMemcpy(ftx_cpu, data.ft_x, data.size() * sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(fty_cpu, data.ft_y, data.size() * sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(ftz_cpu, data.ft_z, data.size() * sizeof(double), cudaMemcpyDeviceToHost);
					
					cudaMemcpy(momx_cpu, data.mom_x, data.size() * sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(momy_cpu, data.mom_y, data.size() * sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(momz_cpu, data.mom_z, data.size() * sizeof(double), cudaMemcpyDeviceToHost);
					
					cudaMemcpy(id_i_cpu, data.id_i, data.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
					cudaMemcpy(id_j_cpu, data.id_j, data.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
					
					cudaMemcpy(sub_i_cpu, data.sub_i, data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
					cudaMemcpy(sub_j_cpu, data.sub_j, data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
					
					int num = 0;
					
					for(int j = 0; j < data.size(); j++)
					{
						//if(i==2) printf("POUR TYPE%d : INTERACTION_ID %d\n", i, j);
						if(j==9482) printf("LE TYPE C'EST: %d \n", i);
						if(ftx_cpu[j]!=0 || fty_cpu[j]!=0 || ftz_cpu[j]!=0 || momx_cpu[j]!=0 || momy_cpu[j]!=0 || momz_cpu[j]!=0)
						{
							if(i==2 && j==9481 ) printf("TOTAL: %d NUM: %d\n", total, num);
							int cell_a = cell_i_cpu[j];
							cell_accessors cellA(cells[cell_a]);
							
							int cell_b = cell_j_cpu[j];
							cell_accessors cellB(cells[cell_b]);
							
							int p_a = p_i_cpu[j];
							particle_info p(shps, p_a, cellA);
							
							int p_b = p_j_cpu[j];
							particle_info p_nbh(shps, p_b, cellB);
							
							//if(i==2 && j==9481 ) printf("ICI2?\n");
							
							///*interaction_history.*//*nteraction_id[num] = encode(id_i_cpu[j], id_j_cpu[j], sub_i_cpu[j], sub_j_cpu[j], max_b, p, p_nbh, i);
							if(i==2 && j==9481)
							{
								printf("A NV : %d\n", p.shp->get_number_of_vertices());
								printf("B NV : %d\n", p_nbh.shp->get_number_of_vertices());  
								
								printf("A NF : %d\n", p.shp->get_number_of_faces());
								printf("B NF : %d\n", p_nbh.shp->get_number_of_faces());
								
								printf("A NE : %d\n", p.shp->get_number_of_edges());
								printf("B NE : %d\n", p_nbh.shp->get_number_of_edges());    
								
								interaction_id[num] = num;
							}
							interaction_history.ft_x[num] = ftx_cpu[j];
							interaction_history.ft_y[num] = fty_cpu[j];
							interaction_history.ft_z[num] = ftz_cpu[j];
							interaction_history.mom_x[num] = momx_cpu[j];
							interaction_history.mom_y[num] = momy_cpu[j];
							interaction_history.mom_z[num] = momz_cpu[j];
							indices[num] = num;
							num++;
							
							//if(i==2 && j==9481 ) printf("ICI3?\n");
						}
						//if(i==2 && j==9481) printf("E POI?\n");
					}
						
					cudaDeviceSynchronize();*/
					
					//printf("FILL\n");
					}
					else if(i == 1)
					{
						max_c = 20;
						max_d = 30;
						fill_active_interactions<<<nbBlocks, 256>>>( grid->cells(), data.cell_i, data.cell_j, data.p_i, data.p_j, data.ft_x, data.ft_y, data.ft_z, data.mom_x, data.mom_y, data.mom_z, data.id_i, data.id_j, data.sub_i, data.sub_j, shps, max_b, max_c, max_d, nb_history_incr.data(), /*interaction_history.interaction_id*/interaction_id/*.data()*/, interaction_history.ft_x/*.data()*/, interaction_history.ft_y/*.data()*/, interaction_history.ft_z/*.data()*/, interaction_history.mom_x/*.data()*/, interaction_history.mom_y/*.data()*/, interaction_history.mom_z/*.data()*/, indices/*.data()*/, i, data.size());
					}
					else if(i == 2)
					{
						max_c = 20;
						max_d = 12;
						fill_active_interactions<<<nbBlocks, 256>>>( grid->cells(), data.cell_i, data.cell_j, data.p_i, data.p_j, data.ft_x, data.ft_y, data.ft_z, data.mom_x, data.mom_y, data.mom_z, data.id_i, data.id_j, data.sub_i, data.sub_j, shps, max_b, max_c, max_d, nb_history_incr.data(), /*interaction_history.interaction_id*/interaction_id/*.data()*/, interaction_history.ft_x/*.data()*/, interaction_history.ft_y/*.data()*/, interaction_history.ft_z/*.data()*/, interaction_history.mom_x/*.data()*/, interaction_history.mom_y/*.data()*/, interaction_history.mom_z/*.data()*/, indices/*.data()*/, i, data.size());
					}
					else
					{
						max_c = 30;
						max_d = 30;
						fill_active_interactions<<<nbBlocks, 256>>>( grid->cells(), data.cell_i, data.cell_j, data.p_i, data.p_j, data.ft_x, data.ft_y, data.ft_z, data.mom_x, data.mom_y, data.mom_z, data.id_i, data.id_j, data.sub_i, data.sub_j, shps, max_b, max_c, max_d, nb_history_incr.data(), /*interaction_history.interaction_id*/interaction_id/*.data()*/, interaction_history.ft_x/*.data()*/, interaction_history.ft_y/*.data()*/, interaction_history.ft_z/*.data()*/, interaction_history.mom_x/*.data()*/, interaction_history.mom_y/*.data()*/, interaction_history.mom_z/*.data()*/, indices/*.data()*/, i, data.size());
					}
					
					d_temp_storage = nullptr;
					temp_storage_bytes = 0;
					
					cub::DeviceRadixSort::SortPairs(
						d_temp_storage, temp_storage_bytes,
						interaction_id/*.data()*/, interaction_history.interaction_id/*.data()*/,
						indices/*.data()*/, interaction_history.indices/*.data()*/,
						total);
						
					cudaMalloc(&d_temp_storage, temp_storage_bytes);
					
					cub::DeviceRadixSort::SortPairs(
						d_temp_storage, temp_storage_bytes,
						interaction_id/*.data()*/, interaction_history.interaction_id/*.data()*/,
						indices/*.data()*/, interaction_history.indices/*.data()*/,
						total);
						
					cudaFree(d_temp_storage);
					
					cudaFree(interaction_id);
					cudaFree(indices);
					
					//printf("NEXT: %d\n", i);
					//}
				}
				   }
	/*uint32_t* cell_i_cpu = (uint32_t*)malloc(size_interactions * sizeof(uint32_t));
	uint32_t* cell_j_cpu = (uint32_t*)malloc(size_interactions * sizeof(uint32_t));
				
	uint16_t* p_i_cpu = (uint16_t*)malloc(size_interactions * sizeof(uint16_t));
	uint16_t* p_j_cpu = (uint16_t*)malloc(size_interactions * sizeof(uint16_t));
				
	cudaMemcpy(cell_i_cpu, interactions2.cell_i, size_interactions * sizeof(uint32_t)  , cudaMemcpyDeviceToHost);
	cudaMemcpy(cell_j_cpu, interactions2.cell_j, size_interactions * sizeof(uint32_t), cudaMemcpyDeviceToHost);
				
	cudaMemcpy(p_i_cpu, interactions2.p_i, size_interactions  * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(p_j_cpu, interactions2.p_j, size_interactions  * sizeof(uint16_t), cudaMemcpyDeviceToHost);
				
	/*for(int i = 0; i < size_interactions; i++)
	{
		printf("INDEX_OBB: %d CELL_I: %d P_I: %d CELL_J: %d P_J: %d\n", i, cell_i_cpu[i], p_i_cpu[i], cell_j_cpu[i], p_j_cpu[i]);
	}*/
	
	/*int count[4];
	
	for(int i = 0; i < 4; i++)
	{
		count[i] = 0;
	}
	
	for(int i = /*230000*///0; i < size_interactions; i++)
	//{
		
		//printf("INDEX_INTERACTION: %d CELL_I: %d P_I: %d CELL_J: %d P_J: %d\n", i, interactions2.cell_i[i], interactions2.p_i[i], interactions2.cell_j[i], interactions2.p_j[i]);
		//printf("INDEX_INTERACTION: %d CELL_I: %d P_I: %d CELL_J: %d P_J: %d\n", i, cell_i_cpu[i], p_i_cpu[i], cell_j_cpu[i], p_j_cpu[i]);
		
		//auto cell_a = /*interactions2.*/cell_i_cpu[i];
		//cell_accessors cellA(cells[cell_a]);
		
		//printf("CELL_A: %d\n", cell_a);
		
		//auto p_a = /*interactions2.*/p_i_cpu[i];
		//printf("PA FIRST\n");
		//particle_info p(shps, p_a, cellA);
		
		//printf("P_A\n");
		
		//auto cell_b = /*interactions2.*/cell_j_cpu[i];
		//cell_accessors cellB(cells[cell_b]);
		
		//printf("CELL_B\n");
		
		//auto p_b = /*interactions2.*/p_j_cpu[i];
		//particle_info p_nbh(shps, p_b, cellB);
		
		//printf("P_B\n");
		
		//printf("CELLA: %d PA: %d CELLB: %d PB: %d\n", cell_a, p_a, cell_b, p_b);
		
		/*const ParticleVertexView vertices_a = { p_a, vertex_fields[cell_a] };
		const ParticleVertexView vertices_b = { p_b, vertex_fields[cell_b] };
		
		//printf("VERTEX\n");
		
		//count_interaction_block_pair( rVerlet, count, p, vertices_a, p_nbh, vertices_b );
		
		auto& shp = p.shp;
		auto& shp_nbh = p_nbh.shp;
		
		const int nv = shp->get_number_of_vertices();
		const int ne = shp->get_number_of_edges();
		const int nf = shp->get_number_of_faces();
		const int nv_nbh = shp_nbh->get_number_of_vertices();
		const int ne_nbh = shp_nbh->get_number_of_edges();
		const int nf_nbh = shp_nbh->get_number_of_faces();
		
		for(int a = 0; a < nv; a++)
		{
			vec3r vi = conv_to_vec3r(vertices_a[a]);
			
			for(int b = 0; b < nv_nbh; b++)
			{
				if (exaDEM::filter_vertex_vertex(rVerlet, vertices_a, a, shp, vertices_b, b, shp_nbh))
				{
					count[0]++;
				}
			}
			
			for(int b = 0; b < ne_nbh; b++)
			{
				if(exaDEM::filter_vertex_edge(rVerlet, vertices_a, a, shp, vertices_b, b, shp_nbh))
				{
					count[1]++;
				}
			}
			
			for(int b = 0; b < nf_nbh; b++)
			{
				if(exaDEM::filter_vertex_face(rVerlet, vertices_a, a, shp, vertices_b, b, shp_nbh))
				{
					count[2]++;
				}
			}
		}
		
		//printf("NV\n");
		
		for(int a = 0; a < ne; a++)
		{
			for(int b = 0; b < ne_nbh; b++)
			{
				if(exaDEM::filter_edge_edge(rVerlet, vertices_a, a, shp, vertices_b, b, shp_nbh))
				{
					count[3]++;
				}
			}
		}
		
		//printf("NE\n");
		
		for(int a = 0; a < nv_nbh; a++)
		{
			vec3r vj = conv_to_vec3r(vertices_b[a]);
			
			for(int b = 0; b < ne; b++)
			{
				if(exaDEM::filter_vertex_edge(rVerlet, vertices_b, a, shp_nbh, vertices_a, b, shp))
				{
					count[1]++;
				}
			}
			
			for(int b = 0; b < nf; b++)
			{
				if(exaDEM::filter_vertex_face(rVerlet, vertices_b, a, shp_nbh, vertices_a, b, shp))
				{
					count[2]++;
				}
			}
		}
		
		//printf("COUNT_UN: %d COUNT_DEUX: %d COUNT_TROIS: %d COUNT_QUATRE: %d\n", count[0], count[1], count[2], count[3]);
		printf("END_INTERACTION\n");
	}
	
	
	for(int i = 0; i < 4; i++)
	{
		printf("TYPE: %d COUNT: %d\n", i, count[i]);
	}
	
	free(cell_i_cpu);
	free(cell_j_cpu);
	free(p_i_cpu);
	free(p_j_cpu);			
				

	/** first count the number of interactions per cell */
				get_number_of_interactions_block_pair<block_x, block_y><<<GridSize, BlockSize>>>(
						grid->cells(),
						vertex_fields,
						grid->dimension(),
						shps,
						rVerlet,
						onika::cuda::vector_data(number_of_interactions_cell),
						interactions2.cell_i/*.data()*/,
						interactions2.cell_j/*.data()*/,
						interactions2.p_i/*.data()*/,
						interactions2.p_j/*.data()*/);
				cudaDeviceSynchronize();
				//lout << "End get_number_of_interactions_block_pair" << std::endl;
				
				void* d_temp_storage = nullptr;
				size_t temp_storage_bytes = 0;

				// First call to get size
				scan_per_type_with_cub(size_interactions,
    					onika::cuda::vector_data(number_of_interactions_cell),
    					onika::cuda::vector_data(prefix_interactions_cell),
    					nullptr,
    					temp_storage_bytes);

				cudaMalloc(&d_temp_storage, temp_storage_bytes);

				// Actual call
				scan_per_type_with_cub(size_interactions,
    					onika::cuda::vector_data(number_of_interactions_cell),
    					onika::cuda::vector_data(prefix_interactions_cell),
    					d_temp_storage,
    					temp_storage_bytes);

				cudaFree(d_temp_storage);
				
       /** Get the total number of interaction per type */
				NumberOfPolyhedronInteractionPerTypes total_nb_int;
				cudaDeviceSynchronize();
				for(int type = 0 ; type < NumberOfPolyhedronInteractionTypes ; type++)
				{
					total_nb_int[type] = prefix_interactions_cell[size_interactions-1][type] + number_of_interactions_cell[size_interactions-1][type];
					//lout << "size " << type << " =  " << total_nb_int[type] << std::endl;
				}
				
				auto& type0 = classifier2.waves[0];
				auto& type1 = classifier2.waves[1];
				auto& type2 = classifier2.waves[2];
				auto& type3 = classifier2.waves[3];
				
				cudaFree(type0.ft_x);
				cudaFree(type1.ft_x);
				cudaFree(type2.ft_x);
				cudaFree(type3.ft_x);
				
				cudaFree(type0.ft_y);
				cudaFree(type1.ft_y);
				cudaFree(type2.ft_y);
				cudaFree(type3.ft_y);
				
				cudaFree(type0.ft_z);
				cudaFree(type1.ft_z);
				cudaFree(type2.ft_z);
				cudaFree(type3.ft_z);
				
				cudaFree(type0.mom_x);
				cudaFree(type1.mom_x);
				cudaFree(type2.mom_x);
				cudaFree(type3.mom_x);

				cudaFree(type0.mom_y);
				cudaFree(type1.mom_y);
				cudaFree(type2.mom_y);
				cudaFree(type3.mom_y);
				
				cudaFree(type0.mom_z);
				cudaFree(type1.mom_z);
				cudaFree(type2.mom_z);
				cudaFree(type3.mom_z);
				
				cudaFree(type0.id_i);
				cudaFree(type1.id_i);
				cudaFree(type2.id_i);
				cudaFree(type3.id_i);
				
				cudaFree(type0.id_j);
				cudaFree(type1.id_j);
				cudaFree(type2.id_j);
				cudaFree(type3.id_j);
				
				cudaFree(type0.cell_i);
				cudaFree(type1.cell_i);
				cudaFree(type2.cell_i);
				cudaFree(type3.cell_i);
				
				cudaFree(type0.cell_j);
				cudaFree(type1.cell_j);
				cudaFree(type2.cell_j);
				cudaFree(type3.cell_j);
				
				cudaFree(type0.p_i);
				cudaFree(type1.p_i);
				cudaFree(type2.p_i);
				cudaFree(type3.p_i);
				
				cudaFree(type0.p_j);
				cudaFree(type1.p_j);
				cudaFree(type2.p_j);
				cudaFree(type3.p_j);
				
				cudaFree(type0.sub_i);
				cudaFree(type1.sub_i);
				cudaFree(type2.sub_i);
				cudaFree(type3.sub_i);
				
				cudaFree(type0.sub_j);
				cudaFree(type1.sub_j);
				cudaFree(type2.sub_j);
				cudaFree(type3.sub_j);	
				
				cudaMalloc(&type0.ft_x, total_nb_int[0] * sizeof(double) );
				cudaMalloc(&type1.ft_x, total_nb_int[1] * sizeof(double) );
				cudaMalloc(&type2.ft_x, total_nb_int[2] * sizeof(double) );
				cudaMalloc(&type3.ft_x, total_nb_int[3] * sizeof(double) );
				
				cudaMalloc(&type0.ft_y, total_nb_int[0] * sizeof(double) );
				cudaMalloc(&type1.ft_y, total_nb_int[1] * sizeof(double) );
				cudaMalloc(&type2.ft_y, total_nb_int[2] * sizeof(double) );
				cudaMalloc(&type3.ft_y, total_nb_int[3] * sizeof(double) );
				
				cudaMalloc(&type0.ft_z, total_nb_int[0] * sizeof(double) );
				cudaMalloc(&type1.ft_z, total_nb_int[1] * sizeof(double) );
				cudaMalloc(&type2.ft_z, total_nb_int[2] * sizeof(double) );
				cudaMalloc(&type3.ft_z, total_nb_int[3] * sizeof(double) );
				
				cudaMalloc(&type0.mom_x, total_nb_int[0] * sizeof(double) );
				cudaMalloc(&type1.mom_x, total_nb_int[1] * sizeof(double) );
				cudaMalloc(&type2.mom_x, total_nb_int[2] * sizeof(double) );
				cudaMalloc(&type3.mom_x, total_nb_int[3] * sizeof(double) );

				cudaMalloc(&type0.mom_y, total_nb_int[0] * sizeof(double) );
				cudaMalloc(&type1.mom_y, total_nb_int[1] * sizeof(double) );
				cudaMalloc(&type2.mom_y, total_nb_int[2] * sizeof(double) );
				cudaMalloc(&type3.mom_y, total_nb_int[3] * sizeof(double) );
				
				cudaMalloc(&type0.mom_z, total_nb_int[0] * sizeof(double) );
				cudaMalloc(&type1.mom_z, total_nb_int[1] * sizeof(double) );
				cudaMalloc(&type2.mom_z, total_nb_int[2] * sizeof(double) );
				cudaMalloc(&type3.mom_z, total_nb_int[3] * sizeof(double) );
				
				cudaMalloc(&type0.id_i, total_nb_int[0] * sizeof(uint64_t) );
				cudaMalloc(&type1.id_i, total_nb_int[1] * sizeof(uint64_t) );
				cudaMalloc(&type2.id_i, total_nb_int[2] * sizeof(uint64_t) );
				cudaMalloc(&type3.id_i, total_nb_int[3] * sizeof(uint64_t) );
				
				cudaMalloc(&type0.id_j, total_nb_int[0] * sizeof(uint64_t) );
				cudaMalloc(&type1.id_j, total_nb_int[1] * sizeof(uint64_t) );
				cudaMalloc(&type2.id_j, total_nb_int[2] * sizeof(uint64_t) );
				cudaMalloc(&type3.id_j, total_nb_int[3] * sizeof(uint64_t) );
				
				cudaMalloc(&type0.cell_i, total_nb_int[0] * sizeof(uint32_t) );
				cudaMalloc(&type1.cell_i, total_nb_int[1] * sizeof(uint32_t) );
				cudaMalloc(&type2.cell_i, total_nb_int[2] * sizeof(uint32_t) );
				cudaMalloc(&type3.cell_i, total_nb_int[3] * sizeof(uint32_t) );
				
				cudaMalloc(&type0.cell_j, total_nb_int[0] * sizeof(uint32_t) );
				cudaMalloc(&type1.cell_j, total_nb_int[1] * sizeof(uint32_t) );
				cudaMalloc(&type2.cell_j, total_nb_int[2] * sizeof(uint32_t) );
				cudaMalloc(&type3.cell_j, total_nb_int[3] * sizeof(uint32_t) );
				
				cudaMalloc(&type0.p_i, total_nb_int[0] * sizeof(uint16_t) );
				cudaMalloc(&type1.p_i, total_nb_int[1] * sizeof(uint16_t) );
				cudaMalloc(&type2.p_i, total_nb_int[2] * sizeof(uint16_t) );
				cudaMalloc(&type3.p_i, total_nb_int[3] * sizeof(uint16_t) );
				
				cudaMalloc(&type0.p_j, total_nb_int[0] * sizeof(uint16_t) );
				cudaMalloc(&type1.p_j, total_nb_int[1] * sizeof(uint16_t) );
				cudaMalloc(&type2.p_j, total_nb_int[2] * sizeof(uint16_t) );
				cudaMalloc(&type3.p_j, total_nb_int[3] * sizeof(uint16_t) );
				
				cudaMalloc(&type0.sub_i, total_nb_int[0] * sizeof(uint16_t) );
				cudaMalloc(&type1.sub_i, total_nb_int[1] * sizeof(uint16_t) );
				cudaMalloc(&type2.sub_i, total_nb_int[2] * sizeof(uint16_t) );
				cudaMalloc(&type3.sub_i, total_nb_int[3] * sizeof(uint16_t) );
				
				cudaMalloc(&type0.sub_j, total_nb_int[0] * sizeof(uint16_t) );
				cudaMalloc(&type1.sub_j, total_nb_int[1] * sizeof(uint16_t) );
				cudaMalloc(&type2.sub_j, total_nb_int[2] * sizeof(uint16_t) );
				cudaMalloc(&type3.sub_j, total_nb_int[3] * sizeof(uint16_t) );
				
				type0.m_type = 0;
				type1.m_type = 1;
				type2.m_type = 2;
				type3.m_type = 3;
				
				type0.size2 = total_nb_int[0];
				type1.size2 = total_nb_int[1];
				type2.size2 = total_nb_int[2];
				type3.size2 = total_nb_int[3];
				
				//classifier.resize(total_nb_int, ResizeClassifier::POLYHEDRON);
				//cudaDeviceSynchronize();
	
				/** Now, we fill the classifier */
  			//lout << "Run fill_classifier_gpu ... " << std::endl;
				fill_classifier_block_pair<block_x, block_y><<<GridSize, BlockSize>>>(
						onika::cuda::vector_data(classifier2.waves),
						//onika::cuda::vector_data(classifier.waves), 
						grid->cells(),
						vertex_fields,
						grid->dimension(),
						shps,
						rVerlet,
						onika::cuda::vector_data(prefix_interactions_cell),
						interactions2.cell_i/*.data()*/,
						interactions2.cell_j/*.data()*/,
						interactions2.p_i/*.data()*/,
						interactions2.p_j/*.data()*/);
				//lout << "End fill_classifier_gpu" << std::endl;
				
				onika::memory::CudaMMVector<int> actives;
				actives.resize(4);
				
				if(!classifier2.use) 
				{
					classifier2.use = true;
				}
				else
				{
					for(int i = 0; i < 4; i++)
					{
						//if( i != 2 )
						//{
						auto& interaction_classifier = classifier2.waves[i];
						
						int nbBlocks = ( interaction_classifier.size() + 256 - 1) / 256;
						
						//onika::memory::CudaMMVector<int> interaction_ids_in;
						int* interaction_ids_in;
						//interaction_ids_in.resize(interaction_classifier.size());
						cudaMalloc(&interaction_ids_in, interaction_classifier.size() * sizeof(int));
						
						int max_b = g.number_of_particles();
						int max_c;
						int max_d;
						
						if(i == 0)
						{
							max_c = 20;
							max_d = 20;
						}
						else if(i == 1)
						{
							max_c = 20;
							max_d = 30;
						}
						else if(i == 2)
						{
							max_c = 20;
							max_d = 12;
						}
						else
						{
							max_c = 30;
							max_d = 30;
						}
						
						//onika::memory::CudaMMVector<int> indices_in;
						int* indices_in;
						//indices_in.resize(interaction_classifier.size());
						cudaMalloc(&indices_in, interaction_classifier.size() * sizeof(int));
						
						encodeClassifier<<<nbBlocks, 256>>>( grid->cells(), interaction_classifier.cell_i, interaction_classifier.cell_j, interaction_classifier.p_i, interaction_classifier.p_j, interaction_classifier.id_i, interaction_classifier.id_j, interaction_classifier.sub_i, interaction_classifier.sub_j, shps, max_b, max_c, max_d, interaction_ids_in/*.data()*/, indices_in/*.data()*/, i, interaction_classifier.size() );
						
						/*( GridT* cells,
				uint32_t* cell_i,
				uint32_t* cell_j,
				uint16_t* p_i,
				uint16_t* p_j,
				uint64_t* id_i,
				uint64_t* id_j,
				uint16_t* sub_i,
				uint16_t* sub_j,
				shapes shps,
				int max_b,
				int max_c,
				int max_d,
				int* res,
				int* indices,
				int type,
				int size)*/
						
						cudaDeviceSynchronize();
						
						//onika::memory::CudaMMVector<int> interaction_ids_out;
						int* interaction_ids_out;
						//onika::memory::CudaMMVector<int> indices_out;
						int* indices_out;
						
						//interaction_ids_out.resize(interaction_classifier.size());
						cudaMalloc(&interaction_ids_out, interaction_classifier.size() * sizeof(int) );
						//indices_out.resize(interaction_classifier.size());
						cudaMalloc(&indices_out, interaction_classifier.size() * sizeof(int) );
						
						void* d_temp_storage = nullptr;
						size_t temp_storage_bytes = 0;
						
						cub::DeviceRadixSort::SortPairs(
							d_temp_storage, temp_storage_bytes,
							interaction_ids_in/*.data()*/, interaction_ids_out/*.data()*/,
							indices_in/*.data()*/, indices_out/*.data()*/,
							interaction_classifier.size());
						
						cudaMalloc(&d_temp_storage, temp_storage_bytes);
					
						cub::DeviceRadixSort::SortPairs(
							d_temp_storage, temp_storage_bytes,
							interaction_ids_in/*.data()*/, interaction_ids_out/*.data()*/,
							indices_in/*.data()*/, indices_out/*.data()*/,
							interaction_classifier.size());
							
						cudaFree(d_temp_storage);
						
						auto& interaction_history = update.waves[i];
						
						find_common_elements<<<nbBlocks, 256>>>( interaction_ids_out/*.data()*/, interaction_history.interaction_id/*.data()*/, indices_out/*.data()*/, interaction_history.indices/*.data()*/, interaction_classifier.ft_x, interaction_classifier.ft_y, interaction_classifier.ft_z, interaction_classifier.mom_x, interaction_classifier.mom_y, interaction_classifier.mom_z, interaction_history.ft_x/*.data()*/, interaction_history.ft_y/*.data()*/, interaction_history.ft_z/*.data()*/, interaction_history.mom_x/*.data()*/, interaction_history.mom_y/*.data()*/, interaction_history.mom_z/*.data()*/, interaction_classifier.size(), interaction_history.size);

						cudaFree(interaction_history.interaction_id);
						cudaFree(interaction_history.ft_x);
						cudaFree(interaction_history.ft_y);
						cudaFree(interaction_history.ft_z);
						cudaFree(interaction_history.mom_x);
						cudaFree(interaction_history.mom_y);
						cudaFree(interaction_history.mom_z);
						cudaFree(interaction_history.indices);
						
						cudaFree(interaction_ids_in);
						cudaFree(interaction_ids_out);
						cudaFree(indices_in);
						cudaFree(indices_out);
						//}
						
						onika::memory::CudaMMVector<int> nb_active;
						nb_active.resize(1);
						
						search_active_interactions2<<<nbBlocks, 256>>>( interaction_classifier.ft_x, interaction_classifier.ft_y, interaction_classifier.ft_z, interaction_classifier.mom_x, interaction_classifier.mom_y, interaction_classifier.mom_z, nb_active.data(), interaction_classifier.size() );
						cudaDeviceSynchronize();
						
						//printf("TYPE%d  %d/%d\n", i, nb_active[0], interaction_classifier.size());
						
						actives[i] = nb_active[0];
					}
				}
				
				printf("GPU Version :\n");
				printf("    Vertex - Vertex : %d / %d\n", actives[0], total_nb_int[0]);
				printf("    Vertex - Edge   : %d / %d\n", actives[1], total_nb_int[1]);
				printf("    Vertex - Face   : %d / %d\n", actives[2], total_nb_int[2]);
				printf("    Edge - Edge     : %d / %d\n", actives[3], total_nb_int[3]);
			}

			if( *block_version )
			{
        lout << "NBH polyhedron Block version" << std::endl;
				lout << "Start get_number_of_interations_block ..." << std::endl;
        /** Define cuda block and grid size */
/*
				constexpr int block_x = 32;
				constexpr int block_y = 4;
				
*/
				constexpr int block_x = 8;
				constexpr int block_y = 8;
				dim3 BlockSize(block_x, block_y, 1);
				dim3 GridSize(cell_size,1,1);

        /** first count the number of interactions per cell */
				get_number_of_interations_block<block_x, block_y><<<GridSize, BlockSize>>>(
						grid->cells(),
						vertex_fields,
						grid->dimension(), 
						chunk_neighbors->data(), 
						shps, 
						rVerlet,
						onika::cuda::vector_data(number_of_interactions_cell), 
						cell_ptr);
				cudaDeviceSynchronize();
				lout << "End get_number_of_interations_block" << std::endl;

				// compute prefix sum using the most stupid way
				stupid_prefix_sum<<<1,32>>>(
						cell_size,
						onika::cuda::vector_data(number_of_interactions_cell),
						onika::cuda::vector_data(prefix_interactions_cell)
						);

        /** Get the total number of interaction per type */
				NumberOfPolyhedronInteractionPerTypes total_nb_int;
				cudaDeviceSynchronize();
				for(int type = 0 ; type < NumberOfPolyhedronInteractionTypes ; type++)
				{
					total_nb_int[type] = prefix_interactions_cell[cell_size-1][type] + number_of_interactions_cell[cell_size-1][type];
					lout << "size " << type << " =  " << total_nb_int[type] << std::endl;
				}

				//resize classifier
				//classifier.resize(total_nb_int, ResizeClassifier::POLYHEDRON);
				cudaDeviceSynchronize();

				
				/** Now, we fill the classifier */
  			lout << "Run fill_classifier_gpu ... " << std::endl;
				fill_classifier_block<block_x, block_y><<<GridSize, BlockSize>>>(
						onika::cuda::vector_data(classifier.waves), 
						grid->cells(),
						vertex_fields,
						grid->dimension(),
						chunk_neighbors->data(),
						shps,
						rVerlet,
						onika::cuda::vector_data(prefix_interactions_cell),
						cell_ptr);	
				lout << "End fill_classifier_gpu" << std::endl;
			}

			if( *pair_version )
			{
        /*lout << "NBH polyhedron Pair version" << std::endl;
				lout << "Start get_number_of_interations_pair ..." << std::endl;
        /** Define cuda block and grid size */
				/*constexpr int block_x = 8;
				constexpr int block_y = 8;
				dim3 BlockSize(block_x, block_y, 8);
				dim3 GridSize(cell_size,1,1);

        /** first count the number of interactions per cell */
				/*get_number_of_interations_pair<block_x, block_y><<<GridSize, BlockSize>>>(
						grid->cells(),
						vetex_fields,
						grid->dimension(), 
						chunk_neighbors->data(), 
						shps, 
						rVerlet,
						onika::cuda::vector_data(number_of_interactions_cell), 
						cell_ptr); 
				cudaDeviceSynchronize();
				lout << "End get_number_of_interations_pair" << std::endl;

				// compute prefix sum using the most stupid way
				stupid_prefix_sum<<<1,32>>>(
						cell_size, 
						onika::cuda::vector_data(number_of_interactions_cell),
						onika::cuda::vector_data(prefix_interactions_cell)
						); 

        /** Get the total number of interaction per type */
				/*NumberOfPolyhedronInteractionPerTypes total_nb_int;
				cudaDeviceSynchronize();
				for(int type = 0 ; type < NumberOfPolyhedronInteractionTypes ; type++)
				{
					total_nb_int[type] = prefix_interactions_cell[cell_size-1][type] + number_of_interactions_cell[cell_size-1][type];
					lout << "size " << type << " =  " << total_nb_int[type] << std::endl;
				}

				// resize classifier
				classifier.resize(total_nb_int, ResizeClassifier::POLYHEDRON);
				cudaDeviceSynchronize();

				/** Now, we fill the classifier */
  			/*lout << "Run fill_classifier_gpu ... " << std::endl;
				fill_classifier_pair<block_x, block_y><<<GridSize, BlockSize>>>(
						onika::cuda::vector_data(classifier.waves),
						grid->cells(),
						vertex_fields,
						grid->dimension(),
						chunk_neighbors->data(),
						shps,
						rVerlet,
						onika::cuda::vector_data(prefix_interactions_cell),
						cell_ptr);
				lout << "End fill_classifier_gpu" << std::endl;*/
			}

      if( *particle_version ) //particle version
      {  
        /*lout << "NBH polyhedron Particle version" << std::endl;
				lout << "Start get_number_of_interations_gpu ..." << std::endl;
        /** Define cuda block and grid size */
				/*constexpr int block_x = 128;
				dim3 BlockSize(block_x, 1, 1);
				dim3 GridSize(cell_size,1,1);

        /** first count the number of interactions per cell */
				/*get_number_of_interations<block_x><<<GridSize, BlockSize>>>(
						grid->cells(),
						grid->dimension(), 
						chunk_neighbors->data(), 
						shps, 
						rVerlet,
						onika::cuda::vector_data(number_of_interactions_cell), 
						cell_ptr); 
				cudaDeviceSynchronize();
				lout << "End get_number_of_interations" << std::endl;

				// compute prefix sum using the most stupid way
				stupid_prefix_sum<<<1,32>>>(
						cell_size, 
						onika::cuda::vector_data(number_of_interactions_cell),
						onika::cuda::vector_data(prefix_interactions_cell)
						); 

        /** Get the total number of interaction per type */
				/*NumberOfPolyhedronInteractionPerTypes total_nb_int;
				cudaDeviceSynchronize();
				for(int type = 0 ; type < NumberOfPolyhedronInteractionTypes ; type++)
				{
					total_nb_int[type] = prefix_interactions_cell[cell_size-1][type] + number_of_interactions_cell[cell_size-1][type];
					lout << "size " << type << " =  " << total_nb_int[type] << std::endl;
				}

				// resize classifier
				classifier.resize(total_nb_int, ResizeClassifier::POLYHEDRON);
				cudaDeviceSynchronize();

				/** Now, we fill the classifier */
  			/*lout << "Run fill_classifier ... " << std::endl;
				fill_classifier<block_x><<<GridSize, BlockSize>>>(
						onika::cuda::vector_data(classifier.waves),
						grid->cells(),
						grid->dimension(),
						chunk_neighbors->data(),
						shps,
						rVerlet,
						onika::cuda::vector_data(prefix_interactions_cell),
						cell_ptr);
				lout << "End fill_classifier" << std::endl;
				
				//classifier.waves[0].clear();
				//classifier.waves[1].clear();
				//classifier.waves[2].clear();
				//classifier.waves[3].clear();*/
      }

#     pragma omp parallel
      {
        // local storage per thread
        Interaction item;
        interaction_manager manager;
#       pragma omp for schedule(dynamic)
        for (size_t ci = 0; ci < cell_size; ci++)
        {
          size_t cell_a = cell_ptr[ci];
          auto& vertex_cell_a = vertex_fields[cell_a];
          IJK loc_a = grid_index_to_ijk(dims, cell_a);

          const unsigned int n_particles = cells[cell_a].size();
          CellExtraDynamicDataStorageT<Interaction> &storage = interactions[cell_a];

          assert(interaction_test::check_extra_interaction_storage_consistency(storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));

          if (n_particles == 0)
          {
            storage.initialize(0);
            continue;
          }

          // Extract history before reset it
          const size_t data_size = storage.m_data.size();
          Interaction *__restrict__ data_ptr = storage.m_data.data();
          extract_history(manager.hist, data_ptr, data_size);
          std::sort(manager.hist.begin(), manager.hist.end());
          manager.reset(n_particles);

          // Reset storage, interaction history was stored in the manager
          storage.initialize(n_particles);
          auto &info_particles = storage.m_info;

          // Get data pointers
          const uint64_t *__restrict__ id_a = cells[cell_a][field::id];
          ONIKA_ASSUME_ALIGNED(id_a);
          const auto *__restrict__ rx_a = cells[cell_a][field::rx];
          ONIKA_ASSUME_ALIGNED(rx_a);
          const auto *__restrict__ ry_a = cells[cell_a][field::ry];
          ONIKA_ASSUME_ALIGNED(ry_a);
          const auto *__restrict__ rz_a = cells[cell_a][field::rz];
          ONIKA_ASSUME_ALIGNED(rz_a);
          const auto *__restrict__ t_a = cells[cell_a][field::type];
          ONIKA_ASSUME_ALIGNED(t_a);
          const auto *__restrict__ orient_a = cells[cell_a][field::orient];
          ONIKA_ASSUME_ALIGNED(orient_a);

          // Define a function to add a new interaction if a contact is possible.
          auto add_contact = [&manager](size_t p, Interaction &item, int sub_i, int sub_j) -> void
          {
            item.sub_i = sub_i;
            item.sub_j = sub_j;
            manager.add_item(p, item);
          };

          // Fill particle ids in the interaction storage
          for (size_t it = 0; it < n_particles; it++)
          {
            info_particles[it].pid = id_a[it];
          }

          // First, interaction between a polyhedron and a driver
          if (drivers.has_value())
          {
            auto &drvs = *drivers;
            item.cell_i = cell_a;
            // By default, if the interaction is between a particle and a driver
            // Data about the particle j is set to -1
            // Except for id_j that contains the driver id
            item.id_j = -1;
            item.cell_j = -1;
            item.p_j = -1;
            item.moment = Vec3d{0, 0, 0};
            item.friction = Vec3d{0, 0, 0};
            for (size_t drvs_idx = 0; drvs_idx < drvs.get_size(); drvs_idx++)
            {
              item.id_j = drvs_idx; // we store the driver idx
              if (drvs.type(drvs_idx) == DRIVER_TYPE::CYLINDER)
              {
                item.type = 4;
                Cylinder &driver = drvs.get_typed_driver<Cylinder>(drvs_idx); // std::get<Cylinder>(drvs.data(drvs_idx)) ;
                add_driver_interaction(driver, add_contact, item, n_particles, rVerlet, t_a, id_a, vertex_cell_a, shps);
              }
              else if (drvs.type(drvs_idx) == DRIVER_TYPE::SURFACE)
              {
                item.type = 5;
                Surface &driver = drvs.get_typed_driver<Surface>(drvs_idx); //std::get<Surface>(drvs.data(drvs_idx));
                add_driver_interaction(driver, add_contact, item, n_particles, rVerlet, t_a, id_a, vertex_cell_a, shps);
              }
              else if (drvs.type(drvs_idx) == DRIVER_TYPE::BALL)
              {
                item.type = 6;
                Ball &driver = drvs.get_typed_driver<Ball>(drvs_idx); //std::get<Ball>(drvs.data(drvs_idx));
                add_driver_interaction(driver, add_contact, item, n_particles, rVerlet, t_a, id_a, vertex_cell_a, shps);
              }
              else if (drvs.type(drvs_idx) == DRIVER_TYPE::STL_MESH)
              {
                Stl_mesh &driver = drvs.get_typed_driver<Stl_mesh>(drvs_idx); //std::get<STL_MESH>(drvs.data(drvs_idx));
                // driver.grid_indexes_summary();
                add_driver_interaction(driver, cell_a, add_contact, item, n_particles, rVerlet, t_a, id_a, rx_a, ry_a, rz_a, vertex_cell_a, orient_a, shps);
              }
            }
          }

          manager.update_extra_storage<true>(storage);
          assert(interaction_test::check_extra_interaction_storage_consistency(storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));
          assert(migration_test::check_info_value(storage.m_info.data(), storage.m_info.size(), 1e6));
        }
        //    GRID_OMP_FOR_END
      }
			//lout << "end of nbh_polyhedron gpu" << std::endl;
		}
	};

	template <class GridT> using UpdateGridCellInteractionGPUTmpl = UpdateGridCellInteractionGPU<GridT>;

	// === register factories ===
	ONIKA_AUTORUN_INIT(nbh_polyhedron_gpu) { OperatorNodeFactory::instance()->register_factory("nbh_polyhedron_gpu", make_grid_variant_operator<UpdateGridCellInteractionGPUTmpl>); }
} // namespace exaDEM
