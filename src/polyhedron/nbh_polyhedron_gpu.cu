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
#include <exaDEM/nbh_polyhedron_block.h>
#include <exaDEM/nbh_polyhedron_particle.h>
#include <exaDEM/nbh_polyhedron_pair.h>

#include <cassert>

namespace exaDEM
{
	using namespace exanb;
	using VertexArray = ::onika::oarray_t<::exanb::Vec3d, EXADEM_MAX_VERTICES>;

	template <typename GridT, class = AssertGridHasFields<GridT>> class UpdateGridCellInteractionGPU : public OperatorNode
	{
		using ComputeFields = FieldSet<>;
		static constexpr ComputeFields compute_field_set{};

		ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
		ADD_SLOT(Domain , domain, INPUT , REQUIRED );
		ADD_SLOT(exanb::GridChunkNeighbors, chunk_neighbors, INPUT, OPTIONAL, DocString{"Neighbor list"});
		ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
		ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
		ADD_SLOT(shapes, shapes_collection, INPUT, DocString{"Collection of shapes"});
		ADD_SLOT(double, rcut_inc, INPUT_OUTPUT, DocString{"value added to the search distance to update neighbor list less frequently. in physical space"});
		ADD_SLOT(Drivers, drivers, INPUT, DocString{"List of Drivers"});
		ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});

    using VectorTypes = onika::memory::CudaMMVector<NumberOfPolyhedronInteractionPerTypes>;

    ADD_SLOT(bool, block_version, false, PRIVATE);
    ADD_SLOT(bool, pair_version, false, PRIVATE);
    ADD_SLOT(bool, particle_version, false, PRIVATE);
    ADD_SLOT(VectorTypes, nbOfInteractionsCell, PRIVATE);
    ADD_SLOT(VectorTypes, prefixInteractionsCell, PRIVATE);

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
					const VertexArray *__restrict__ vertices, 
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
					auto& vertices_i = vertices[p];
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
						vec3r v = conv_to_vec3r(vertices[p][i]);
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
					const VertexArray *__restrict__ vertices, 
					shapes &shps)
			{
				for (size_t p = 0; p < n_particles; p++)
				{
					const auto va = vertices[p];
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

			lout << "start nbh_polyhedron_gpu" << std::endl;

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

      // declare n int and prefix
      auto& number_of_interactions_cell = *nbOfInteractionsCell;
      auto& prefix_interactions_cell = *prefixInteractionsCell;

      if(number_of_interactions_cell.size() != cell_size)
      {
			  number_of_interactions_cell.resize(cell_size);
			  prefix_interactions_cell.resize(cell_size);
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

				// resize classifier
				classifier.resize(total_nb_int, ResizeClassifier::POLYHEDRON);
				cudaDeviceSynchronize();

				/** Now, we fill the classifier */
  			lout << "Run fill_classifier_gpu ... " << std::endl;
				fill_classifier_block<block_x, block_y><<<GridSize, BlockSize>>>(
						onika::cuda::vector_data(classifier.waves),
						grid->cells(),
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
        lout << "NBH polyhedron Pair version" << std::endl;
				lout << "Start get_number_of_interations_pair ..." << std::endl;
        /** Define cuda block and grid size */
				constexpr int block_x = 16;
				constexpr int block_y = 16;
				dim3 BlockSize(block_x, block_y, 1);
				dim3 GridSize(cell_size,1,1);

        /** first count the number of interactions per cell */
				get_number_of_interations_pair<block_x, block_y><<<GridSize, BlockSize>>>(
						grid->cells(),
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
				NumberOfPolyhedronInteractionPerTypes total_nb_int;
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
  			lout << "Run fill_classifier_gpu ... " << std::endl;
				fill_classifier_pair<block_x, block_y><<<GridSize, BlockSize>>>(
						onika::cuda::vector_data(classifier.waves),
						grid->cells(),
						grid->dimension(),
						chunk_neighbors->data(),
						shps,
						rVerlet,
						onika::cuda::vector_data(prefix_interactions_cell),
						cell_ptr);
				lout << "End fill_classifier_gpu" << std::endl;
			}

      if( *particle_version ) //particle version
      {  
        lout << "NBH polyhedron Particle version" << std::endl;
				lout << "Start get_number_of_interations_gpu ..." << std::endl;
        /** Define cuda block and grid size */
				constexpr int block_x = 128;
				dim3 BlockSize(block_x, 1, 1);
				dim3 GridSize(cell_size,1,1);

        /** first count the number of interactions per cell */
				get_number_of_interations<block_x><<<GridSize, BlockSize>>>(
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
				NumberOfPolyhedronInteractionPerTypes total_nb_int;
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
  			lout << "Run fill_classifier ... " << std::endl;
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

					const unsigned int n_particles = cells[cell_a].size();
					CellExtraDynamicDataStorageT<Interaction> &storage = interactions[cell_a];

					assert(interaction_test::check_extra_interaction_storage_consistency(storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));

					if (n_particles == 0)
						continue;

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
					const auto *__restrict__ vertices_a = cells[cell_a][field::vertices];
					ONIKA_ASSUME_ALIGNED(vertices_a);

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
						item.id_j = size_t(-1);
						item.cell_j = size_t(-1);
						item.p_j = size_t(-1);
						item.moment = Vec3d{0, 0, 0};
						item.friction = Vec3d{0, 0, 0};
						for (size_t drvs_idx = 0; drvs_idx < drvs.get_size(); drvs_idx++)
						{
							item.id_j = drvs_idx; // we store the driver idx
							if (drvs.type(drvs_idx) == DRIVER_TYPE::CYLINDER)
							{
								item.type = 4;
								Cylinder &driver = drvs.get_typed_driver<Cylinder>(drvs_idx); // std::get<Cylinder>(drvs.data(drvs_idx)) ;
								add_driver_interaction(driver, add_contact, item, n_particles, rVerlet, t_a, id_a, vertices_a, shps);
							}
							else if (drvs.type(drvs_idx) == DRIVER_TYPE::SURFACE)
							{
								item.type = 5;
								Surface &driver = drvs.get_typed_driver<Surface>(drvs_idx); //std::get<Surface>(drvs.data(drvs_idx));
								add_driver_interaction(driver, add_contact, item, n_particles, rVerlet, t_a, id_a, vertices_a, shps);
							}
							else if (drvs.type(drvs_idx) == DRIVER_TYPE::BALL)
							{
								item.type = 6;
								Ball &driver = drvs.get_typed_driver<Ball>(drvs_idx); //std::get<Ball>(drvs.data(drvs_idx));
								add_driver_interaction(driver, add_contact, item, n_particles, rVerlet, t_a, id_a, vertices_a, shps);
							}
							else if (drvs.type(drvs_idx) == DRIVER_TYPE::STL_MESH)
							{
								Stl_mesh &driver = drvs.get_typed_driver<Stl_mesh>(drvs_idx); //std::get<STL_MESH>(drvs.data(drvs_idx));
																																							// driver.grid_indexes_summary();
								add_driver_interaction(driver, cell_a, add_contact, item, n_particles, rVerlet, t_a, id_a, rx_a, ry_a, rz_a, vertices_a, orient_a, shps);
							}
						}
					}

					manager.update_extra_storage<true>(storage);
					assert(interaction_test::check_extra_interaction_storage_consistency(storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));
					assert(migration_test::check_info_value(storage.m_info.data(), storage.m_info.size(), 1e6));
				}
				//    GRID_OMP_FOR_END
			}
			lout << "end of nbh_polyhedron gpu" << std::endl;
		}
	};

	template <class GridT> using UpdateGridCellInteractionGPUTmpl = UpdateGridCellInteractionGPU<GridT>;

	// === register factories ===
	ONIKA_AUTORUN_INIT(nbh_polyhedron_gpu) { OperatorNodeFactory::instance()->register_factory("nbh_polyhedron_gpu", make_grid_variant_operator<UpdateGridCellInteractionGPUTmpl>); }
} // namespace exaDEM
