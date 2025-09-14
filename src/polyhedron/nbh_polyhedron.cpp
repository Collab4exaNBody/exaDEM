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
#include <exaDEM/nbh_polyhedron_driver.hpp>
#include <exaDEM/traversal.h>

#include <cassert>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT>> class UpdateGridCellInteraction : public OperatorNode
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

    public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This function builds the list of interactions per particle (polyhedron). Interactions are between two particles or a particle and a driver. In this function, frictions and moments are updated if the interactions are still actived. Note that, a list of non-empty cells is built during this function.

        YAML example [no option]:

          - nbh_polyhedron
       )EOF";
    }

    inline void check_slots()
    {
      if (drivers.has_value() && !domain->xform_is_identity())
      {
        if(drivers->get_size() > 0)
        {
          color_log::error("nbh_polyhedron", "Contact detection with drivers is deactivated when the simulation box is deformed.");
        }
      }
    }

    inline void execute() override final
    {
      check_slots();
      auto& g = *grid;
      auto& vertex_fields = *cvf;
      const auto cells = g.cells();
      const size_t n_cells = g.number_of_cells(); // nbh.size();
      const IJK dims = g.dimension();
      auto &interactions = ges->m_data;
      shapes &shps = *shapes_collection;
      double rVerlet = *rcut_inc;
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

          // Second, we add interactions between two polyhedra.

          apply_cell_particle_neighbors(*grid, *chunk_neighbors, cell_a, loc_a, std::false_type() /* not symetric */,
              [&g, &vertex_fields, &cells, &info_particles, cell_a, &item, &shps, rVerlet, id_a, rx_a, ry_a, rz_a, t_a, orient_a, &vertex_cell_a, &add_contact, xform, is_xform](size_t p_a, size_t cell_b, unsigned int p_b, size_t p_nbh_index)
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
              const Quaternion orient_nbh = cells[cell_b][field::orient][p_b];
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

              // Eliminate if two polyhedra are two far away if there is not intersection between their OBBs.
              OBB obb_i = shp->obb;
              OBB obb_j = shp_nbh->obb;
              const Quaternion &orient = orient_a[p_a];
              quat conv_orient_i = quat{vec3r{orient.x, orient.y, orient.z}, orient.w};
              quat conv_orient_j = quat{vec3r{orient_nbh.x, orient_nbh.y, orient_nbh.z}, orient_nbh.w};
              obb_i.rotate(conv_orient_i);
              obb_j.rotate(conv_orient_j);
              obb_i.translate(vec3r{rx, ry, rz});
              obb_j.translate(vec3r{rx_nbh, ry_nbh, rz_nbh});

              obb_i.enlarge(rVerlet);
              obb_j.enlarge(rVerlet);

              if (!obb_i.intersect(obb_j))
                return;

              // Add interactions
              item.id_i = id_a[p_a];
              item.p_i = p_a;

              item.cell_i = cell_a;
              item.p_j = p_b;
              item.cell_j = cell_b;

              const Vec3d r = {rx, ry, rz};

              // get particle j data.
              const int nv = shp->get_number_of_vertices();
              const int ne = shp->get_number_of_edges();
              const int nf = shp->get_number_of_faces();
              const int nv_nbh = shp_nbh->get_number_of_vertices();
              const int ne_nbh = shp_nbh->get_number_of_edges();
              const int nf_nbh = shp_nbh->get_number_of_faces();

              item.id_j = id_nbh;
              // exclude possibilities with obb
              for (int i = 0; i < nv; i++)
              {
                auto vi = shp->get_vertex(i, r, orient);
                OBB obbvi;
                obbvi.center = {vi.x, vi.y, vi.z};
                obbvi.enlarge(shp->m_radius);
                if (obb_j.intersect(obbvi))
                {
                  item.type = 0; // === Vertex - Vertex
                  for (int j = 0; j < nv_nbh; j++)
                  {
                    if (exaDEM::filter_vertex_vertex(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh))
                    {
                      add_contact(p_a, item, i, j);
                    }
                  }

                  item.type = 1; // === vertex edge
                  for (int j = 0; j < ne_nbh; j++)
                  {
                    bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
                    if (contact)
                    {
                      add_contact(p_a, item, i, j);
                    }
                  }

                  item.type = 2; // === vertex face
                  for (int j = 0; j < nf_nbh; j++)
                  {
                    bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
                    if (contact)
                    {
                      add_contact(p_a, item, i, j);
                    }
                  }
                }
              }

              item.type = 3; // === edge edge
              for (int i = 0; i < ne; i++)
              {
                for (int j = 0; j < ne_nbh; j++)
                {
                  bool contact = exaDEM::filter_edge_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
                  if (contact)
                  {
                    add_contact(p_a, item, i, j);
                  }
                }
              }

              // interaction of from particle j to particle i
              item.cell_j = cell_a;
              item.id_j = id_a[p_a];
              item.p_j = p_a;

              item.cell_i = cell_b;
              item.p_i = p_b;
              item.id_i = id_nbh;

              for (int j = 0; j < nv_nbh; j++)
              {
                auto vj = vertices_b[j];//shp->get_vertex(j, r_nbh, orient_nbh);
                OBB obbvj;
                obbvj.center = {vj.x, vj.y, vj.z};
                obbvj.enlarge(shp_nbh->m_radius);

                if (obb_i.intersect(obbvj))
                {
                  item.type = 1; // === vertex edge
                  for (int i = 0; i < ne; i++)
                  {
                    bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
                    if (contact)
                    {
                      add_contact(p_a, item, j, i);
                    }
                  }

                  item.type = 2; // === vertex face
                  for (int i = 0; i < nf; i++)
                  {
                    bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
                    if (contact)
                    {
                      add_contact(p_a, item, j, i);
                    }
                  }
                }
              }
              });

          manager.update_extra_storage<true>(storage);
          assert(interaction_test::check_extra_interaction_storage_consistency(storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));
          assert(migration_test::check_info_value(storage.m_info.data(), storage.m_info.size(), 1e6));
        }
        //    GRID_OMP_FOR_END
      }
    }
  };

  template <class GridT> using UpdateGridCellInteractionTmpl = UpdateGridCellInteraction<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(nbh_polyhedron) { OperatorNodeFactory::instance()->register_factory("nbh_polyhedron", make_grid_variant_operator<UpdateGridCellInteraction>); }
} // namespace exaDEM
