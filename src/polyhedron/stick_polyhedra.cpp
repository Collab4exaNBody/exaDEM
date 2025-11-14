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
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction_manager.hpp>
#include <exaDEM/interaction/migration_test.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/traversal.h>

#include <cassert>

namespace exaDEM
{
  using namespace exanb;

  Vec3d normalize(Vec3d&& in) { return in / exanb::norm(in); }

  template <typename GridT, class = AssertGridHasFields<GridT>> class StickPolyhedraOperator : public OperatorNode
  {
    using ComputeFields = FieldSet<>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
    ADD_SLOT(Domain, domain, INPUT, REQUIRED);
    ADD_SLOT(exanb::GridChunkNeighbors, chunk_neighbors, INPUT, OPTIONAL, DocString{"Neighbor list"});
    ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
    ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
    ADD_SLOT(double, sticking_threshold, INPUT, REQUIRED, DocString{""});
    ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});

    public:
    inline std::string documentation() const override final
    {
      return R"EOF(
       )EOF";
    }

    inline void execute() override final
    {
      lout << "=================================="  << std::endl;
      lout << "Stick polyhedra ... " << std::endl;
      auto& g = *grid;
      auto& vertex_fields = *cvf;
      const auto cells = g.cells();
      const size_t n_cells = g.number_of_cells(); // nbh.size();
      const IJK dims = g.dimension();
      auto &interactions = ges->m_data;
      shapes &shps = *shapes_collection;
      double dn_crit = *sticking_threshold;
      Mat3d xform = domain->xform();
      bool is_xform = !domain->xform_is_identity();

      assert(interactions.size() == n_cells);

      if (!chunk_neighbors.has_value())
      {
#       pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_cells; i++)
          interactions[i].initialize(0);
        return;
      }

      auto [cell_ptr, cell_size] = traversal_real->info();

#     pragma omp parallel
      {
        // local storage per thread
        PlaceholderInteraction item;
        item.clear_placeholder();
        InteractionManager manager;
        std::vector<PlaceholderInteraction> local;
#       pragma omp for schedule(dynamic)
        for (size_t ci = 0; ci < cell_size; ci++)
        {
          size_t cell_i = cell_ptr[ci];
          auto& vertex_cell_i = vertex_fields[cell_i];
          IJK loc_i = grid_index_to_ijk(dims, cell_i);

          const unsigned int n_particles = cells[cell_i].size();
          CellExtraDynamicDataStorageT<PlaceholderInteraction> &storage = interactions[cell_i];

          assert(interaction_test::check_extra_interaction_storage_consistency(storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));

          if (n_particles == 0)
          {
            storage.initialize(0);
            continue;
          }

          // reset it
          manager.reset(n_particles);

          // Reset storage, interaction history was stored in the manager
          storage.initialize(n_particles);
          auto &info_particles = storage.m_info;

          // Get data pointers
          const uint64_t *__restrict__ id_i = cells[cell_i][field::id];
          ONIKA_ASSUME_ALIGNED(id_i);
          const auto *__restrict__ rx_i = cells[cell_i][field::rx];
          ONIKA_ASSUME_ALIGNED(rx_i);
          const auto *__restrict__ ry_i = cells[cell_i][field::ry];
          ONIKA_ASSUME_ALIGNED(ry_i);
          const auto *__restrict__ rz_i = cells[cell_i][field::rz];
          ONIKA_ASSUME_ALIGNED(rz_i);
          const auto *__restrict__ t_i = cells[cell_i][field::type];
          ONIKA_ASSUME_ALIGNED(t_i);

          // Define a function to add a new interaction if a contact is possible.
					auto add_contact = [](
              std::vector<PlaceholderInteraction>& local,
							size_t p, 
							PlaceholderInteraction &item, 
							int sub_i, 
							int sub_j, 
							double dn0) -> void
					{
						item.pair.pi.sub = sub_i;
						item.pair.pj.sub = sub_j;
						item.as<InnerBondInteraction>().dn0 = dn0;
						local.push_back(item);
					};

					// Fill particle ids in the interaction storage
					for (size_t it = 0; it < n_particles; it++)
					{
						info_particles[it].pid = id_i[it];
					}

					// Second, we add interactions between two polyhedra.

					apply_cell_particle_neighbors(
							*grid, 
							*chunk_neighbors, 
							cell_i, 
							loc_i, 
							std::false_type() /* not symetric */,
							// capture
							[&g, 
							&vertex_fields, 
							&cells, 
							&info_particles, 
							cell_i, 
							&item, 
							&shps, 
							dn_crit, 
							id_i, 
							rx_i, 
							ry_i, 
							rz_i, 
							t_i, 
							&vertex_cell_i, 
							&add_contact, 
							xform, 
							is_xform,
							&manager,
              &local
							](
									size_t p_i, 
									size_t cell_j, 
									unsigned int p_j, 
									size_t p_j_index)
							{
								// default value of the interaction studied (A or i -> B or j)
								const uint64_t id_j = cells[cell_j][field::id][p_j];
								if (id_i[p_i] >= id_j)
								{
									if (!g.is_ghost_cell(cell_j))
										return;
								}

								// Get particle pointers for the particle j.
								VertexField& vertex_cell_j = vertex_fields[cell_j];
								ParticleVertexView vertices_j = {p_j, vertex_cell_j};
								auto& cellj = cells[cell_j];
								const uint32_t typej = cellj[field::type][p_j];
								double rxj = cellj[field::rx][p_j];
								double ryj = cellj[field::ry][p_j];
								double rzj = cellj[field::rz][p_j];

								// Get particle pointers for the particle i.
								ParticleVertexView vertices_i = {p_i, vertex_cell_i};
								double rxi = rx_i[p_i];
								double ryi = ry_i[p_i];
								double rzi = rz_i[p_i];

								if( is_xform )
								{
									Vec3d tmp = {rxj, ryj, rzj};
									tmp = xform * tmp;
									rxj = tmp.x;
									ryj = tmp.y;
									rzj = tmp.z;
									tmp = {rxi, ryi, rzi};
									tmp = xform * tmp;
									rxi = tmp.x;
									ryi = tmp.y;
									rzi = tmp.z;
								}

								const shape *shpi = shps[t_i[p_i]];
								const shape *shpj = shps[typej];

								// Add interactions
								item.pair.type = InteractionTypeId::StickedParticles;

								auto& pi = item.i(); // particle i (id, cell id, particle position, sub vertex)
								pi.id = id_i[p_i];
								pi.p = p_i;
								pi.cell = cell_i;
								const int nfi = shpi->get_number_of_faces();

								auto& pj = item.j(); // particle j (id, cell id, particle position, sub vertex)
								pj.id = id_j;
								pj.p = p_j;
								pj.cell = cell_j;
								const int nfj = shpj->get_number_of_faces();

								bool found = false;

								double criterion = shpj->minskowski() + shpi->minskowski() + dn_crit;

								for (int i = 0; i < nfi && !found; i++)
								{
									auto [vi, size_i] = shpi->get_face(i);
									Vec3d vi0 = vertices_i[vi[0]];
									Vec3d vi1 = vertices_i[vi[1]];
									Vec3d vi2 = vertices_i[vi[2]];

									Vec3d ni = normalize(exanb::cross(vi1 - vi0, vi2 - vi0));

									for (int j = 0; j < nfj && !found; j++)
									{
										auto [vj, size_j] = shpj->get_face(j);

										if( size_i != size_j ) continue;

										Vec3d vj0 = vertices_j[vj[0]];
										Vec3d vj1 = vertices_j[vj[1]];
										Vec3d vj2 = vertices_j[vj[2]];

										Vec3d nj = normalize(exanb::cross(vj1 - vj0, vj2 - vj0));
										Vec3d nij = exanb::cross(ni, nj);
										double dotij = exanb::norm(nij);
										if(dotij > 1.e-10) continue;

										//if( std::abs(exanb::dot(nj, vj0) - exanb::dot(ni, vi0)) > criterion ) continue;

										found = true;


										for(int ivf=0 ; ivf<size_i ; ivf++)
										{
											bool vertex_not_found = true;
											for(int jvf=0 ; jvf<size_j ; jvf++) 
											{
												if(exaDEM::filter_vertex_vertex(dn_crit, vertices_i, vi[ivf], shpi, vertices_j, vj[jvf], shpj))
												{
													auto contact = detection_vertex_vertex(vertices_i, vi[ivf], shpi, vertices_j, vj[jvf], shpj);
													add_contact(local, p_i, item, vi[ivf], vj[jvf], contact.dn);
													vertex_not_found = false;
													break;
												}
											}
											if( vertex_not_found ) 
											{
												color_log::warning("stick_polyhedra", "It is impossible to glue sides "
														+ std::to_string(i) + " and " + std::to_string(j)
														+ " together; vertex " + std::to_string(ivf) + " cannot be bonded.");
												found = false;
												local.clear();
												break;
											}
										}
									}
								}



								/*								// get particle j data.
																	const int nv = shp->get_number_of_vertices();
																	const int nv_j = shp_j->get_number_of_vertices();

																	pj.id = id_j;
																	for (int i = 0; i < nv; i++)
																	{
																	for (int j = 0; j < nv_j; j++)
																	{
																	if(exaDEM::filter_vertex_vertex(dn_crit, vertices_a, i, shp, vertices_b, j, shp_j))
																	{
																	auto contact = detection_vertex_vertex(vertices_a, i, shp, vertices_b, j, shp_j);
																	add_contact(local, p_i, item, i, j, contact.dn);
																	}
																	}
																	}
								 */
								bool check = check_stiked_face(local, p_i, vertices_i);
								if(check) manager.add(local);
								local.clear();
							});

					constexpr bool use_history = false;
					manager.update_extra_storage<use_history>(storage); // copy manager data in storage.
					assert(interaction_test::check_extra_interaction_storage_consistency(storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));
					assert(migration_test::check_info_value(storage.m_info.data(), storage.m_info.size(), 1e6));
				}  //    GRID_OMP_FOR_END
			}
		}
	};

	template <class GridT> using StickPolyhedraOperatorTmpl = StickPolyhedraOperator<GridT>;

	// === register factories ===
	ONIKA_AUTORUN_INIT(nbh_polyhedron) 
	{ 
		OperatorNodeFactory::instance()->register_factory("stick_polyhedra", make_grid_variant_operator<StickPolyhedraOperator>); 
	}
} // namespace exaDEM
