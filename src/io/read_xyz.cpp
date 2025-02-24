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

namespace exaDEM
{
  using namespace exanb;

  struct ParticleType
  {
    static inline constexpr size_t MAX_STR_LEN = 16;

    double m_mass = 1.0;
    double m_radius = 1.0;
    char m_name[MAX_STR_LEN] = {'\0'};

    inline void set_name(const std::string &s)
    {
      if (s.length() >= MAX_STR_LEN)
      {
        std::cerr << "Particle name too long : length=" << s.length() << ", max=" << (MAX_STR_LEN - 1) << "\n";
        std::abort();
      }
      std::strncpy(m_name, s.c_str(), MAX_STR_LEN);
      m_name[MAX_STR_LEN - 1] = '\0';
    }
    inline std::string name() const { return m_name; }
  };

  using ParticleTypes = onika::memory::CudaMMVector<ParticleType>;

  template <typename GridT> class ReadXYZ : public OperatorNode
  {
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(std::string, file, INPUT, REQUIRED);
    ADD_SLOT(ReadBoundsSelectionMode, bounds_mode, INPUT, ReadBoundsSelectionMode::FILE_BOUNDS);
    ADD_SLOT(Domain, domain, INPUT_OUTPUT);
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(double, enlarge_bounds, INPUT, 0.0);
    ADD_SLOT(bool, pbc_adjust_xform, INPUT, false);
    ADD_SLOT(bool, adjust_bounds_to_particles, INPUT, false);
    ADD_SLOT(ParticleTypes, particle_types, INPUT); // optional. if no species given, type ids are allocated automatically

  public:
    inline void execute() override final
    {
      //-------------------------------------------------------------------------------------------
      // Reading datas from YAML or previous input
      std::string file_name = onika::data_file_path(*file);

      if (*pbc_adjust_xform)
      {
        if (!domain->xform_is_identity())
        {
          lerr << "pbc_adjust_xform needs initial XForm to be identity, resetting XForm" << std::endl;
          domain->set_xform(make_identity_matrix());
        }
      }

      //      grid = GridT();
      //-------------------------------------------------------------------------------------------
      std::string basename;
      std::string::size_type p = file_name.rfind("/");
      if (p != std::string::npos)
        basename = file_name.substr(p + 1);
      else
        basename = file_name;
      lout << "======== " << basename << " ========" << std::endl;
      //-------------------------------------------------------------------------------------------

      using ParticleTupleIO = onika::soatl::FieldTuple<field::_rx, field::_ry, field::_rz, field::_id, field::_type>;
      using ParticleTuple = decltype(grid->cells()[0][0]);

      assert(grid->number_of_particles() == 0);

      // MPI Initialization
      int rank = 0, np = 1;
      MPI_Comm_rank(*mpi, &rank);
      MPI_Comm_size(*mpi, &np);

      // useful later
      std::string line;
      //      double size_box = 0.0;
      uint64_t n_particles = 0;

      std::map<std::string, unsigned int> typeMap;
      unsigned int nextTypeId = 0;
      if (particle_types.has_value())
      {
        for (size_t i = 0; i < particle_types->size(); i++)
        {
          typeMap[particle_types->at(i).m_name] = i;
        }
        nextTypeId = particle_types->size();
      }

      std::vector<ParticleTupleIO> particle_data;

      // Get max and min positions
      // Need to define the size of the box
      // NOTE : only one processor need to do that
      if (rank == 0)
      {
        std::ifstream file;
        file.open(file_name, std::ifstream::in);
        if (!file.is_open())
        {
          lerr << "Error in reading xyz : file " << file_name << " not found !" << std::endl;
          std::abort();
        }

        // Read number of atoms
        getline(file, line);
        ssize_t n_atoms = -1;
        std::stringstream(line) >> n_atoms;

        std::getline(file, line);
        double box_size_x = -1.0;
        double box_size_y = -1.0;
        double box_size_z = -1.0;
        std::stringstream(line) >> box_size_x >> box_size_y >> box_size_z;
        std::stringstream(line) >> box_size_x >> box_size_y >> box_size_z;

        AABB file_bounds = {{0., 0., 0.}, {box_size_x, box_size_y, box_size_z}};
        lout << "File bounds      = " << file_bounds << std::endl;

        // We need to define limits of the domain from the .xyz file
        // and the position of particles
        // For that, we check the max and min position from all particles
        double min_x = std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double min_z = std::numeric_limits<double>::max();
        double max_x = std::numeric_limits<double>::lowest();
        double max_y = std::numeric_limits<double>::lowest();
        double max_z = std::numeric_limits<double>::lowest();

        // read one line at a time
        while (std::getline(file, line))
        {
          std::string type;
          double x = 0.0, y = 0.0, z = 0.0;

          // first value not necessary here
          std::stringstream(line) >> type >> x >> y >> z;

          min_x = std::min(min_x, x);
          max_x = std::max(max_x, x);

          min_y = std::min(min_y, y);
          max_y = std::max(max_y, y);

          min_z = std::min(min_z, z);
          max_z = std::max(max_z, z);

          if (typeMap.find(type) == typeMap.end())
          {
            typeMap[type] = nextTypeId;
            ++nextTypeId;
          }

          particle_data.push_back(ParticleTupleIO(x, y, z, n_particles++, typeMap[type]));
        }

        ldbg << "min position xyz file : " << min_x << " " << min_y << " " << min_z << std::endl;
        ldbg << "max position xyz file : " << max_x << " " << max_y << " " << max_z << std::endl;

        // DOMAIN
        AABB computed_bounds = {{min_x, min_y, min_z}, {max_x, max_y, max_z}};
        ldbg << "computed_bounds  = " << computed_bounds << std::endl;

        if (*adjust_bounds_to_particles)
        {
          assert(*enlarge_bounds >= 0);
          if ((*enlarge_bounds) == 0)
            lout << "Warning, enlarge_bounds is equal to 0" << std::endl;
          file_bounds = {{min_x - (*enlarge_bounds), min_y - (*enlarge_bounds), min_z - (*enlarge_bounds)}, {max_x + (*enlarge_bounds), max_y + (*enlarge_bounds), max_z + (*enlarge_bounds)}};
          lout << "File bounds (fit)= " << file_bounds << std::endl;
        }

        // domain->m_bounds = bounds;
        compute_domain_bounds(*domain, *bounds_mode, *enlarge_bounds, file_bounds, computed_bounds, *pbc_adjust_xform);
        if (*pbc_adjust_xform && !domain->xform_is_identity())
        {
          Mat3d inv_xform = domain->inv_xform();
          for (auto &p : particle_data)
          {
            Vec3d r = inv_xform * Vec3d{p[field::rx], p[field::ry], p[field::rz]};
            p[field::rx] = r.x;
            p[field::ry] = r.y;
            p[field::rz] = r.z;
          }
        }
        lout << "Particles        = " << particle_data.size() << std::endl;
        lout << "Domain XForm     = " << domain->xform() << std::endl;
        lout << "Domain bounds    = " << domain->bounds() << std::endl;
        lout << "Domain size      = " << bounds_size(domain->bounds()) << std::endl;
        lout << "Real size        = " << bounds_size(domain->bounds()) * Vec3d{domain->xform().m11, domain->xform().m22, domain->xform().m33} << std::endl;
        lout << "Cell size        = " << domain->cell_size() << std::endl;
        lout << "Grid dimensions  = " << domain->grid_dimension() << " (" << grid_cell_count(domain->grid_dimension()) << " cells)" << std::endl;
      }

      // send bounds and size_box values to all cores
      MPI_Bcast(&(*domain), sizeof(Domain), MPI_CHARACTER, 0, *mpi);
      assert(check_domain(*domain));

      grid->set_offset(IJK{0, 0, 0});
      grid->set_origin(domain->bounds().bmin);
      grid->set_cell_size(domain->cell_size());
      grid->set_dimension(domain->grid_dimension());

      if (rank == 0)
      {
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
      }

      lout << "============================" << std::endl;

      grid->rebuild_particle_offsets();

#     ifndef NDEBUG
        bool particles_inside_cell = check_particles_inside_cell(*grid);
        assert(particles_inside_cell);
#     endif
    }
  };

  // === register factories ===
  __attribute__((constructor)) static void register_factories() { OperatorNodeFactory::instance()->register_factory("read_xyz", make_grid_variant_operator<ReadXYZ>); }

} // namespace exaDEM
