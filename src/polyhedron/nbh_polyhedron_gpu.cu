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
		//int* interaction_id;
		uint64_t* id_i;
		uint64_t* id_j;
		uint16_t* sub_i;
		uint16_t* sub_j;
		
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
	
	__global__ void fill_active_interactions(
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
			int* nb_incr,
			uint64_t* idi,
			uint64_t* idj,
			uint16_t* subi,
			uint16_t* subj,
			double* ftx,
			double* fty,
			double* ftz,
			double* momx,
			double* momy,
			double* momz,
			int* indices,
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
			
			if(active)
			{
				//interaction_id[prefix] = encode(id_i[idx], id_j[idx], sub_i[idx], sub_j[idx], max_b, max_c, max_d, /*p, p_nbh,*/ type);
				idi[prefix] = id_i[idx];
				idj[prefix] = id_j[idx];
				subi[prefix] = sub_i[idx];
				subj[prefix] = sub_j[idx];
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
	
__device__ __forceinline__
int lower_bound_idi(const uint64_t* __restrict__ arr, int n, uint64_t key)
{
    int low = 0, high = n; // [low, high)
    while (low < high) {
        int mid = low + ((high - low) >> 1);
        uint64_t v = arr[mid];
        if (v < key) low = mid + 1;
        else          high = mid;
    }
    return low; // première position où arr[pos] >= key
}

// Parcours des OLD ; recherche binaire dans NEW ; copie OLD -> NEW
__global__ void find_common_elements(
    const uint64_t* __restrict__ idi_new,
    const uint64_t* __restrict__ idj_new,
    const uint16_t* __restrict__ subi_new,
    const uint16_t* __restrict__ subj_new,
    const uint64_t* __restrict__ idi_old,
    const uint64_t* __restrict__ idj_old,
    const uint16_t* __restrict__ subi_old,
    const uint16_t* __restrict__ subj_old,
    const int*      __restrict__ indices_new,   // indexation des sorties NEW
    const int*      __restrict__ indices_old,   // peut être nullptr (voir ci-dessous)
    double* __restrict__ ftx,   // sorties NEW
    double* __restrict__ fty,
    double* __restrict__ ftz,
    double* __restrict__ momx,
    double* __restrict__ momy,
    double* __restrict__ momz,
    const double* __restrict__ ftx_old, // sources OLD
    const double* __restrict__ fty_old,
    const double* __restrict__ ftz_old,
    const double* __restrict__ momx_old,
    const double* __restrict__ momy_old,
    const double* __restrict__ momz_old,
    int size_new,
    int size_old)
{
    int old_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (old_idx >= size_old) return;

    const uint64_t key_idi  = idi_old[old_idx];
    const uint64_t key_idj  = idj_old[old_idx];
    const uint16_t key_subi = subi_old[old_idx];
    const uint16_t key_subj = subj_old[old_idx];

    // Recherche binaire sur NEW (trié par idi)
    int pos = lower_bound_idi(idi_new, size_new, key_idi);
    if (pos == size_new) return;
    if (idi_new[pos] != key_idi) return;

    // Balayage de la "run" idi_new == key_idi pour trouver (idj,subi,subj)
    for (int i = pos; i < size_new && idi_new[i] == key_idi; ++i) {
        if (idj_new[i] == key_idj &&
            subi_new[i] == key_subi &&
            subj_new[i] == key_subj)
        {
            const int index_new = indices_new[i];
            const int index_old = (indices_old ? indices_old[old_idx] : old_idx);

            ftx [index_new] = ftx_old [index_old];
            fty [index_new] = fty_old [index_old];
            ftz [index_new] = ftz_old [index_old];
            momx[index_new] = momx_old[index_old];
            momy[index_new] = momy_old[index_old];
            momz[index_new] = momz_old[index_old];
            return;
        }
    }
    // pas de match -> rien à faire
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
					//cudaMalloc(&interaction_history.interaction_id, total * sizeof(int) );
					cudaMalloc(&interaction_history.id_i, total * sizeof(uint64_t) );
					cudaMalloc(&interaction_history.id_j, total * sizeof(uint64_t) );
					cudaMalloc(&interaction_history.sub_i, total * sizeof(uint16_t) );
					cudaMalloc(&interaction_history.sub_j, total * sizeof(uint16_t) );
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
					uint64_t* id_i;
					//interaction_id.resize(total);
					cudaMalloc(&id_i, total * sizeof(uint64_t));
					//onika::memory::CudaMMVector<int> indices;
					int* indices;
					//indices.resize(total);
					cudaMalloc(&indices, total * sizeof(int));
					
					//printf("MALLOC\n");

					fill_active_interactions<<<nbBlocks, 256>>>(
					data.ft_x, 
					data.ft_y, 
					data.ft_z, 
					data.mom_x, 
					data.mom_y, 
					data.mom_z, 
					data.id_i, 
					data.id_j, 
					data.sub_i, 
					data.sub_j, 
					nb_history_incr.data(), 
					id_i, 
					interaction_history.id_j, 
					interaction_history.sub_i, 
					interaction_history.sub_j, 
					interaction_history.ft_x, 
					interaction_history.ft_y, 
					interaction_history.ft_z, 
					interaction_history.mom_x, 
					interaction_history.mom_y, 
					interaction_history.mom_z, 
					indices, 
					data.size());

					
					d_temp_storage = nullptr;
					temp_storage_bytes = 0;
					
					cub::DeviceRadixSort::SortPairs(
						d_temp_storage, temp_storage_bytes,
						id_i/*.data()*/, interaction_history.id_i/*.data()*/,
						indices/*.data()*/, interaction_history.indices/*.data()*/,
						total);
						
					cudaMalloc(&d_temp_storage, temp_storage_bytes);
					
					cub::DeviceRadixSort::SortPairs(
						d_temp_storage, temp_storage_bytes,
						id_i/*.data()*/, interaction_history.id_i/*.data()*/,
						indices/*.data()*/, interaction_history.indices/*.data()*/,
						total);
						
					cudaFree(d_temp_storage);
					
					cudaFree(id_i);
					cudaFree(indices);
					
					//printf("NEXT: %d\n", i);
					//}
				}
				   }	
				

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
						uint64_t* id_i;
						//interaction_ids_in.resize(interaction_classifier.size());
						cudaMalloc(&id_i, interaction_classifier.size() * sizeof(uint64_t));
						
						onika::memory::CudaMMVector<int> indices_in;
						//int* indices_in;
						indices_in.resize(interaction_classifier.size());
						//cudaMalloc(&indices_in, interaction_classifier.size() * sizeof(int));
						#pragma omp parallel for
						for(int j = 0; j < interaction_classifier.size(); j++)
						{
							indices_in[j] = j;
						}

						//onika::memory::CudaMMVector<int> indices_out;
						int* indices_out;

						//indices_out.resize(interaction_classifier.size());
						cudaMalloc(&indices_out, interaction_classifier.size() * sizeof(int) );
						
						void* d_temp_storage = nullptr;
						size_t temp_storage_bytes = 0;
						
						cub::DeviceRadixSort::SortPairs(
							d_temp_storage, temp_storage_bytes,
							interaction_classifier.id_i/*.data()*/, id_i/*.data()*/,
							indices_in.data()/*.data()*/, indices_out/*.data()*/,
							interaction_classifier.size());
						
						cudaMalloc(&d_temp_storage, temp_storage_bytes);
					
						cub::DeviceRadixSort::SortPairs(
							d_temp_storage, temp_storage_bytes,
							interaction_classifier.id_i/*.data()*/, id_i/*.data()*/,
							indices_in.data()/*.data()*/, indices_out/*.data()*/,
							interaction_classifier.size());
							
						cudaFree(d_temp_storage);
						
						auto& interaction_history = update.waves[i];
						
						find_common_elements<<<nbBlocks, 256>>>( id_i, 
						interaction_classifier.id_j, 
						interaction_classifier.sub_i, 
						interaction_classifier.sub_j,
						interaction_history.id_i, 
						interaction_history.id_j, 
						interaction_history.sub_i, 
						interaction_history.sub_j,
						indices_out, 
						interaction_history.indices, 
						interaction_classifier.ft_x, 
						interaction_classifier.ft_y, 
						interaction_classifier.ft_z, 
						interaction_classifier.mom_x, 
						interaction_classifier.mom_y, 
						interaction_classifier.mom_z, 
						interaction_history.ft_x, 
						interaction_history.ft_y, 
						interaction_history.ft_z, 
						interaction_history.mom_x, 
						interaction_history.mom_y, 
						interaction_history.mom_z, 
						interaction_classifier.size(), 
						interaction_history.size);
						
						cudaDeviceSynchronize();

						//cudaFree(interaction_history.interaction_id);
						cudaFree(interaction_history.id_i);
						cudaFree(interaction_history.id_j);
						cudaFree(interaction_history.sub_i);
						cudaFree(interaction_history.sub_j);
						cudaFree(interaction_history.ft_x);
						cudaFree(interaction_history.ft_y);
						cudaFree(interaction_history.ft_z);
						cudaFree(interaction_history.mom_x);
						cudaFree(interaction_history.mom_y);
						cudaFree(interaction_history.mom_z);
						cudaFree(interaction_history.indices);
						
						cudaFree(id_i);
						//cudaFree(interaction_ids_out);
						//cudaFree(indices_in);
						cudaFree(indices_out);
						//}
						
						//onika::memory::CudaMMVector<int> nb_active;
						//nb_active.resize(1);
						
						//search_active_interactions2<<<nbBlocks, 256>>>( interaction_classifier.ft_x, interaction_classifier.ft_y, interaction_classifier.ft_z, interaction_classifier.mom_x, interaction_classifier.mom_y, interaction_classifier.mom_z, nb_active.data(), interaction_classifier.size() );
						//cudaDeviceSynchronize();
						
						//printf("TYPE%d  %d/%d\n", i, nb_active[0], interaction_classifier.size());
						
						//actives[i] = nb_active[0];
						//}
					}
				}
				
				//printf("GPU Version :\n");
				//printf("    Vertex - Vertex : %d / %d\n", actives[0], total_nb_int[0]);
				//printf("    Vertex - Edge   : %d / %d\n", actives[1], total_nb_int[1]);
				//printf("    Vertex - Face   : %d / %d\n", actives[2], total_nb_int[2]);
				//printf("    Edge - Edge     : %d / %d\n", actives[3], total_nb_int[3]);
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
