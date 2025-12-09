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

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction_manager.hpp>
#include <exaDEM/interaction/migration_test.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/polyhedron/nbh_polyhedron_driver.hpp>
#include <exaDEM/traversal.h>

#include <cassert>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT>> 
    class UpdateGridCellInteractionWithOBBTree : public OperatorNode
  {
    using ComputeFields = FieldSet<>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
    ADD_SLOT(Domain, domain, INPUT, REQUIRED);
    ADD_SLOT(exanb::GridChunkNeighbors, chunk_neighbors, INPUT, OPTIONAL, DocString{"Neighbor list"});
    ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
    ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
    ADD_SLOT(double, rcut_inc, INPUT, REQUIRED, DocString{"value added to the search distance to update neighbor list less frequently. in physical space"});
    ADD_SLOT(Drivers, drivers, INPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});
    ADD_SLOT(bool, enable_obb_tree, INPUT, REQUIRED);

    public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This function builds the list of interactions per particle (polyhedron). Interactions are between two particles or a particle and a driver. In this function, frictions and moments are updated if the interactions are still actived. Note that, a list of non-empty cells is built during this function.

        YAML example [no option]:

          - nbh_polyhedron_obb_tree

        Note: Before calling nbh_polyhedron_obb_tree, you need to use the operator: enable_obb_tree in input_data
       )EOF";
    }

    inline void execute() override final
    {
      auto& g = *grid;
      auto& vertex_fields = *cvf;
      const auto cells = g.cells();
      const size_t n_cells = g.number_of_cells(); // nbh.size();
      const IJK dims = g.dimension();
      auto &interactions = ges->m_data;
      auto &shps = *shapes_collection;
      double rVerlet = *rcut_inc;
      Mat3d xform = domain->xform();
      bool is_xform = !domain->xform_is_identity();
      if (drivers.has_value() && is_xform)
      {
        if(drivers->get_size() > 0)
        {
          lout<< "[nbh_polyhedron, ERROR] Contact detection with drivers is deactivated when the simulation box is deformed." << std::endl;
          std::exit(0);
        }
      }

      // if grid structure (dimensions) changed, we invalidate thie whole data
      if (interactions.size() != n_cells)
      {
        ldbg << "[nbh_polyhedron, WARNING] Number of cells has changed, reset friction data" << std::endl;
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

#     pragma omp parallel
      {
        // local storage per thread
        PlaceholderInteraction item;
        item.clear_placeholder();
        InteractionManager manager;
        std::vector<std::pair<subBox, subBox>> intersections;
#       pragma omp for schedule(dynamic)
        for (size_t ci = 0; ci < cell_size; ci++)
        {
          size_t cell_a = cell_ptr[ci];
          auto& vertex_cell_a = vertex_fields[cell_a];
          IJK loc_a = grid_index_to_ijk(dims, cell_a);

          const unsigned int n_particles = cells[cell_a].size();
          CellExtraDynamicDataStorageT<PlaceholderInteraction> &storage = interactions[cell_a];

          assert(interaction_test::check_extra_interaction_storage_consistency(storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));

          if (n_particles == 0)
            continue;

          // Extract history before reset it
          const size_t data_size = storage.m_data.size();
          PlaceholderInteraction *__restrict__ data_ptr = storage.m_data.data();
          extract_history(manager.hist, data_ptr, data_size);
          std::sort(manager.hist.begin(), manager.hist.end());
          manager.reset(n_particles);

          // Move persistent interactions in the InteractionManager
          update_persistent_interactions(manager, storage);
          manager.update_ignore_interaction();

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
          auto add_contact = [&manager](PlaceholderInteraction &item, int sub_i, int sub_j) -> void
          {
            item.pair.pi.sub = sub_i;
            item.pair.pj.sub = sub_j;
            manager.add_item(item);
          };

          // Fill particle ids in the interaction storage
          for (size_t it = 0; it < n_particles; it++)
          {
            info_particles[it].pid = id_a[it];
          }

          // First, interaction between a polyhedron and a driver
          if (drivers.has_value())
          {
            auto& pi = item.i(); // particle i (id, cell id, particle position, sub vertex)
            auto& pd = item.driver(); // particle driver (id, cell id, particle position, sub vertex)

            auto &drvs = *drivers;
            pi.cell = cell_a;
            // By default, if the interaction is between a particle and a driver
            // Data about the particle j is set to -1
            // Except for id_j that contains the driver id
            pd.id = -1;
            pd.cell = -1;
            pd.p = -1;
            for (size_t drvs_idx = 0; drvs_idx < drvs.get_size(); drvs_idx++)
            {
              pd.id = drvs_idx; // we store the driver idx
              if (drvs.type(drvs_idx) == DRIVER_TYPE::CYLINDER)
              {
                item.pair.type = 4;
                Cylinder &driver = drvs.get_typed_driver<Cylinder>(drvs_idx); // std::get<Cylinder>(drvs.data(drvs_idx)) ;
                add_driver_interaction(driver, add_contact, item, n_particles, rVerlet, t_a, id_a, vertex_cell_a, shps);
              }
              else if (drvs.type(drvs_idx) == DRIVER_TYPE::SURFACE)
              {
                item.pair.type = 5;
                Surface &driver = drvs.get_typed_driver<Surface>(drvs_idx); //std::get<Surface>(drvs.data(drvs_idx));
                add_driver_interaction(driver, add_contact, item, n_particles, rVerlet, t_a, id_a, vertex_cell_a, shps);
              }
              else if (drvs.type(drvs_idx) == DRIVER_TYPE::BALL)
              {
                item.pair.type = 6;
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

          // Second, we add interactions between two polyhedra.

          apply_cell_particle_neighbors(*grid, *chunk_neighbors, cell_a, loc_a, std::false_type() /* not symetric */,
              [&g, &vertex_fields, &cells, &info_particles, &intersections, cell_a, &item, &shps, rVerlet, id_a, rx_a, ry_a, rz_a, t_a, orient_a, &vertex_cell_a, &add_contact, xform, is_xform](size_t p_a, size_t cell_b, unsigned int p_b, size_t p_nbh_index)
              {
              // default value of the interaction studied (A or i -> B or j)
              const uint64_t id_nbh = cells[cell_b][field::id][p_b];
              if (id_a[p_a] >= id_nbh)
              {
              if (!g.is_ghost_cell(cell_b))
              return;
              }

              VertexField& vertex_cell_b = vertex_fields[cell_b];
              ParticleVertexView vertices_b = {p_b, vertex_cell_b};
              ParticleVertexView vertices_a = {p_a, vertex_cell_a};

              // Get particle pointers for the particle b.
              const uint32_t type_nbh = cells[cell_b][field::type][p_b];
              const Quaternion& orient_nbh = cells[cell_b][field::orient][p_b];
              double rx_nbh = cells[cell_b][field::rx][p_b];
              double ry_nbh = cells[cell_b][field::ry][p_b];
              double rz_nbh = cells[cell_b][field::rz][p_b];
              double rx = rx_a[p_a];
              double ry = ry_a[p_a];
              double rz = rz_a[p_a];

              if( is_xform )
              {
                Vec3d tmp = {rx_nbh, ry_nbh, rz_nbh};
                tmp = xform * tmp;
                rx_nbh = tmp.x;
                ry_nbh = tmp.y;
                rz_nbh = tmp.z;
                tmp = {rx, ry, rz};
                tmp = xform * tmp;
                rx = tmp.x;
                ry = tmp.y;
                rz = tmp.z;
              }

              // prev
              const shape *shp = shps[t_a[p_a]];
              const shape *shp_nbh = shps[type_nbh];

              const Quaternion &orient = orient_a[p_a];
              quat conv_orient_i = quat{vec3r{orient.x, orient.y, orient.z}, orient.w};
              quat conv_orient_j = quat{vec3r{orient_nbh.x, orient_nbh.y, orient_nbh.z}, orient_nbh.w};

              intersections.clear();
              vec3r ra = {rx, ry, rz};
              vec3r rb = {rx_nbh, ry_nbh, rz_nbh};

              quat QAconj = conv_orient_i.get_conjugated();
              vec3r posB_relativeTo_posA = QAconj * (rb - ra);
              quat QB_relativeTo_QA = QAconj * conv_orient_j;

              // Fill intersections
              OBBtree<subBox>::TreeIntersectionIds(
                  shp->obbtree.root, 
                  shp_nbh->obbtree.root, 
                  intersections,
                  1.0,//cells[cell_a][field::homothety][p_a], 
                  1.0,//cells[cell_b][field::homothety][p_b], 
                  0.5*rVerlet,
                  posB_relativeTo_posA, 
                  QB_relativeTo_QA);

              auto set_info_i = [&item] (uint64_t id, size_t p, size_t cid) -> void { item.pair.pi.id = id; item.pair.pi.p = p; item.pair.pi.cell = cid; };
              auto set_info_j = [&item] (uint64_t id, size_t p, size_t cid) -> void { item.pair.pj.id = id; item.pair.pj.p = p; item.pair.pj.cell = cid; };

              for (size_t c = 0; c < intersections.size(); c++) 
              {
                size_t i = intersections[c].first.isub;
                size_t j = intersections[c].second.isub;
                int i_nbPoints = intersections[c].first.nbPoints;
                int j_nbPoints = intersections[c].second.nbPoints;
                //lout << " i " << i << " j " << j << " i_nbPoints " << i_nbPoints << " j_nbPoints " << j_nbPoints << std::endl;

                if (i_nbPoints == 1) 
                {
                  if (j_nbPoints == 1) 
                  {  
                    set_info_i(id_a[p_a], p_a, cell_a);
                    set_info_j(id_nbh,    p_b, cell_b);
                    item.pair.type = 0; // === Vertex - Vertex
                    if (exaDEM::filter_vertex_vertex(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh))
                    {
                      add_contact(item, i, j);
                    }
                  } 
                  else if (j_nbPoints == 2) 
                  {
                    set_info_i(id_a[p_a], p_a, cell_a);
                    set_info_j(id_nbh,    p_b, cell_b);
                    item.pair.type = 1; // === vertex edge
                    bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
                    if (contact)
                    {
                      add_contact(item, i, j);
                    }
                  } 
                  else if (j_nbPoints >= 3) 
                  {
                    set_info_i(id_a[p_a], p_a, cell_a);
                    set_info_j(id_nbh,    p_b, cell_b);
                    item.pair.type = 2; // === vertex face
                    bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
                    if (contact)
                    {
                      add_contact(item, i, j);
                    }
                  }
                } 
                else if (i_nbPoints == 2) 
                {
                  if (j_nbPoints == 1) 
                  {
                    /** warning, a -> j and b -> i */
                    set_info_i(id_nbh,    p_b, cell_b);
                    set_info_j(id_a[p_a], p_a, cell_a);
                    item.pair.type = 1; // === vertex edge
                    bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
                    if (contact)
                    {
                      add_contact(item, j, i);
                    }
                  } 
                  else if (j_nbPoints == 2) 
                  {
                    set_info_i(id_a[p_a], p_a, cell_a);
                    set_info_j(id_nbh,    p_b, cell_b);
                    // === edge edge
                    item.pair.type = 3; 
                    bool contact = exaDEM::filter_edge_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
                    if (contact)
                    {
                      add_contact(item, i, j);
                    }
                  }
                } 
                else if (i_nbPoints >= 3) 
                {
									if (j_nbPoints == 1)
									{
										/** warning, a -> j and b -> i */
										set_info_i(id_nbh,    p_b, cell_b);
										set_info_j(id_a[p_a], p_a, cell_a);
										item.pair.type = 2; // === vertex face
										bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
										if (contact)
										{
											add_contact(item, j, i);
										}
									}
								}
							}  // end loop over intersections
							});

					manager.update_extra_storage<true>(storage);
					assert(interaction_test::check_extra_interaction_storage_consistency(storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));
					assert(migration_test::check_info_value(storage.m_info.data(), storage.m_info.size(), 1e6));
				}
				//    GRID_OMP_FOR_END
			}
		}
	};

	// === register factories ===
	ONIKA_AUTORUN_INIT(nbh_polyhedron_obb_tree) { OperatorNodeFactory::instance()->register_factory("nbh_polyhedron_obb_tree", make_grid_variant_operator<UpdateGridCellInteractionWithOBBTree>); }
} // namespace exaDEM
