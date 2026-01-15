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

#include <chrono>
#include <ctime>
#include <mpi.h>
#include <string>
#include <numeric>

#include <onika/math/basic_types_yaml.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <onika/math/basic_types_stream.h>
#include <onika/log.h>
//#include "ustamp/vector_utils.h"
#include <onika/file_utils.h>
#include <exanb/core/domain.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/simple_block_rcb.h>

// rsa_mpi stuff
#include <rsa_data_storage.hxx>
#include <rsa_random.hxx>
#include <rsa_domain.hxx>
#include <rsa_decoration.hxx>
#include <operator_algorithm.hxx>
#include <radius_generator.hxx>

namespace exaDEM
{
  using namespace exanb;
  template <typename GridT> class InitRSA : public OperatorNode
  {
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(Domain, domain, INPUT_OUTPUT);
    ADD_SLOT(double, enlarge_bounds, INPUT, 0.0);
    ADD_SLOT(std::vector<bool>, periodicity, INPUT, OPTIONAL, DocString{"if set, overrides domain's periodicity stored in file with this value"});
    ADD_SLOT(bool, expandable, INPUT, OPTIONAL, DocString{"if set, override domain expandability stored in file"});
    ADD_SLOT(AABB, bounds, INPUT, REQUIRED, DocString{"if set, override domain's bounds, filtering out particle outside of overriden bounds"});
    ADD_SLOT(int, type, INPUT, 0);
    ADD_SLOT(bool, pbc_adjust_xform, INPUT, true);

    ADD_SLOT(double, radius, INPUT, REQUIRED); // optional. if no species given, type ids are allocated automatically
    ADD_SLOT(double, rcut_max, INPUT_OUTPUT, 0.0);

  public:
    inline void execute() override final
    {
      //-------------------------------------------------------------------------------------------
      using ParticleTupleIO = onika::soatl::FieldTuple<field::_rx, field::_ry, field::_rz, field::_id, field::_type, field::_radius>;
      using ParticleTuple = decltype(grid->cells()[0][0]);

      assert(grid->number_of_particles() == 0);

      // MPI Initialization
      int rank = 0, np = 1;
      MPI_Comm_rank(*mpi, &rank);
      MPI_Comm_size(*mpi, &np);

      AABB b = *bounds;
      constexpr int DIM = 3;
      constexpr int method = 1;
      constexpr int ghost_layer = 1;
      std::array<double, DIM> domain_inf = {b.bmin.x, b.bmin.y, b.bmin.z};
      std::array<double, DIM> domain_sup = {b.bmax.x, b.bmax.y, b.bmax.z};
      rsa_domain<DIM> rsa_domain(domain_inf, domain_sup, ghost_layer, *radius);

      size_t seed = 0;
      algorithm::uniform_generate<DIM, method>(rsa_domain, *radius, 6000, 10, seed);
      auto spheres = rsa_domain.extract_spheres();

      if (rank == 0)
      {
        /** FILE_BOUNDS sounds wrong in this context, but it works. */
        compute_domain_bounds(*domain, exanb::ReadBoundsSelectionMode::FILE_BOUNDS, *enlarge_bounds, b, b, *pbc_adjust_xform);
      }

      // compute indexes
      int ns = spheres.size();
      MPI_Exscan(MPI_IN_PLACE, &ns, 1, MPI_INT, MPI_SUM, *mpi);

      // send bounds and size_box values to all cores
      MPI_Bcast(&(*domain), sizeof(Domain), MPI_CHARACTER, 0, *mpi);
      assert(check_domain(*domain));
      grid->set_offset(IJK{0, 0, 0});
      grid->set_origin(domain->bounds().bmin);
      grid->set_cell_size(domain->cell_size());
      grid->set_dimension(domain->grid_dimension());

      // add particles
      std::vector<ParticleTupleIO> particle_data;
      ParticleTupleIO pt;
      int ParticleType = *type;
      double ParticleRadius = *radius;
      particle_data.resize(spheres.size());
      for (size_t s = 0; s < spheres.size(); s++)
      {
        auto pos = spheres[s].center;
        auto id = ns + s;
        pt = ParticleTupleIO(pos[0], pos[1], pos[2], id, ParticleType, ParticleRadius);
        particle_data[s] = pt;
      }

      // Fill grid, particles will migrate accross mpi processed via the operator migrate_cell_particles
      for (auto p : particle_data)
      {
        Vec3d r{p[field::rx], p[field::ry], p[field::rz]};
        IJK loc = domain_periodic_location(*domain, r); // grid.locate_cell(r);
        assert(grid->contains(loc));
        assert(min_distance2_between(r, grid->cell_bounds(loc)) < grid->epsilon_cell_size2());
        p[field::rx] = r.x;
        p[field::ry] = r.y;
        p[field::rz] = r.z;
        ParticleTuple t = p;
        grid->cell(loc).push_back(t, grid->cell_allocator());
      }

      uint64_t n_particles = particle_data.size();
      uint64_t n;
      MPI_Reduce(&n_particles, &n, 1, MPI_UINT64_T, MPI_SUM, 0, *mpi);

      // Display information
      lout << "=================================" << std::endl;
      lout << "Particles        = " << n << std::endl;
      lout << "Domain XForm     = " << domain->xform() << std::endl;
      lout << "Domain bounds    = " << domain->bounds() << std::endl;
      lout << "Domain size      = " << bounds_size(domain->bounds()) << std::endl;
      lout << "Real size        = " << bounds_size(domain->bounds()) * Vec3d{domain->xform().m11, domain->xform().m22, domain->xform().m33} << std::endl;
      lout << "Cell size        = " << domain->cell_size() << std::endl;
      lout << "Grid dimensions  = " << domain->grid_dimension() << " (" << grid_cell_count(domain->grid_dimension()) << " cells)" << std::endl;
      lout << "=================================" << std::endl;

      grid->rebuild_particle_offsets();
      *rcut_max = std::max(*rcut_max, *radius);
    }
  };

  // === register factories ===
  __attribute__((constructor)) static void register_factories() { OperatorNodeFactory::instance()->register_factory("init_rsa", make_grid_variant_operator<InitRSA>); }

} // namespace exaDEM
