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
#include <string>

#include <onika/math/basic_types_yaml.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <onika/math/basic_types_stream.h>
#include <onika/log.h>
// #include "ustamp/vector_utils.h"
#include <onika/file_utils.h>
#include <exanb/core/domain.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exaDEM/color_log.hpp>
#include <exaDEM/shapes.hpp>

namespace exaDEM {
using namespace exanb;

struct ParticleType {
  static inline constexpr size_t MAX_STR_LEN = 16;

  double m_mass = 1.0;
  double m_radius = 1.0;
  char m_name[MAX_STR_LEN] = {'\0'};

  inline void set_name(const std::string& s) {
    if (s.length() >= MAX_STR_LEN) {
      color_log::error("ParticleType::set_name",
                       "Particle name too long : length=" + std::to_string(s.length()) +
                       ", max=" + std::to_string(MAX_STR_LEN - 1));
    }
    std::strncpy(m_name, s.c_str(), MAX_STR_LEN);
    m_name[MAX_STR_LEN - 1] = '\0';
  }
  inline std::string name() const {
    return m_name;
  }
};

using ParticleTypes = onika::memory::CudaMMVector<ParticleType>;

template <typename GridT>
class AddXYZ : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT_OUTPUT);
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(std::string, file, INPUT, REQUIRED);
  ADD_SLOT(Domain, domain, INPUT);
  ADD_SLOT(AABB, filter, INPUT, OPTIONAL, DocString{"Filter particles."});
  ADD_SLOT(Vec3d, shift, INPUT, OPTIONAL, DocString{"shift the positions."});
  ADD_SLOT(double, rcut_max, INPUT, REQUIRED, DocString{"rcut_max"});
  ADD_SLOT(ParticleTypes, particle_types, INPUT, OPTIONAL);
  ADD_SLOT(shapes, shapes_collection, INPUT, OPTIONAL, DocString{"Collection of shapes"});

  std::string operator_name() {
    return "add_xyz";
  }

  bool check_slot() {
    if (*rcut_max <= 0.0) {
      color_log::error(operator_name(),
                       "The radius of the sphere encompassing the largest particle is not [>0]."
                       "Verify that you are not using this operator instead of a reader or lattice operator."
                       "This operator does not initialize the domain.");
      return false;
    }
    return true;
  }

 public:
  inline void execute() final {
    bool success = check_slot();

    if (!success) {
      color_log::error(operator_name(), "Wrong slot definition.");
    }

    //-------------------------------------------------------------------------------------------
    // Reading datas from YAML or previous input
    std::string file_name = onika::data_file_path(*file);

    AABB Filter;
    Vec3d shifter = *shift;
    auto& g = *grid;
    auto& d = *domain;
    bool apply_filter = filter.has_value();
    spin_mutex_array cell_locks;

    if (apply_filter) {
      Filter = *filter;
      enlarge(Filter, -(*rcut_max));  // reduce AABB

      if (Filter.bmin.x > Filter.bmax.x ||
          Filter.bmin.y > Filter.bmax.y ||
          Filter.bmin.z > Filter.bmax.z) {
        color_log::warning(operator_name(),
                           "The filter area is too small. No particles can be deposited.");
      }
    }

    //-------------------------------------------------------------------------------------------
    std::string basename;
    std::string::size_type p = file_name.rfind("/");
    if (p != std::string::npos) {
      basename = file_name.substr(p + 1);
    } else {
      basename = file_name;
    }
    //-------------------------------------------------------------------------------------------

    using ParticleTupleIO = onika::soatl::FieldTuple<field::_rx, field::_ry, field::_rz, field::_id, field::_type>;
    using ParticleTuple = decltype(g.cells()[0][0]);

    // MPI Initialization
    int rank = 0, np = 1;
    MPI_Comm_rank(*mpi, &rank);
    MPI_Comm_size(*mpi, &np);

    g.rebuild_particle_offsets();

    // useful later
    std::string line;
    int n_particles = 0;

    std::map<std::string, unsigned int> typeMap;
    if (particle_types.has_value()) {
      for (size_t i = 0; i < particle_types->size(); i++) {
        typeMap[particle_types->at(i).m_name] = i;
      }
    }

    if (shapes_collection.has_value()) {
      auto& shps = *shapes_collection;
      for (size_t i = 0; i < shps.size(); i++) {
        typeMap[shps[i]->m_name] = i;
      }
    }

    // get max ID
    unsigned long long next_id = 0;
    const size_t n_cells = g.number_of_cells();
    auto cells = g.cells();
    cell_locks.resize(n_cells);

#   pragma omp parallel for schedule(dynamic) reduction(max:next_id)
    for(size_t cell_i=0; cell_i<n_cells; cell_i++) {
      if( ! g.is_ghost_cell(cell_i) ) {
        size_t n = cells[cell_i].size();
        for(size_t p=0 ; p<n ; p++) {
          const unsigned long long id = cells[cell_i][field::id][p];
          next_id = std::max(next_id, id);
        }
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &next_id, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, *mpi);
    ++ next_id; // start right after greatest id

    std::vector<ParticleTupleIO> particle_data;

    if (rank == 0) {
      std::ifstream file;
      file.open(file_name, std::ifstream::in);
      if (!file.is_open()) {
        color_log::error("add_xyz", "File " + file_name + " not found !");
      }

      // Read number of atoms
      getline(file, line);
      int n_filtered_particles = 0;
      ssize_t n_atoms = -1;
      std::stringstream(line) >> n_atoms;

      std::getline(file, line);
      double box_size_x = -1.0;
      double box_size_y = -1.0;
      double box_size_z = -1.0;
      std::stringstream(line) >> box_size_x >> box_size_y >> box_size_z;
      std::stringstream(line) >> box_size_x >> box_size_y >> box_size_z;

      ldbg << " box: [" << box_size_x << ", "
          << box_size_y << ", "
          << box_size_z << "]" << std::endl; 

      // read one line at a time
      while (std::getline(file, line)) {
        std::string type;
        double x = 0.0, y = 0.0, z = 0.0;

        // first value not necessary here
        std::stringstream(line) >> type >> x >> y >> z;

        Vec3d p = {x,y,z};
        p += shifter;
        if (apply_filter) {
          if (!is_inside(Filter, p)) {
            n_filtered_particles++;
            continue;
          }
        }
        n_particles++;
        std::cout << "Add: p: " << p
            << " id: " << next_id
            << " type: " << typeMap[type] << " " << type << std::endl;
        particle_data.push_back(ParticleTupleIO(p.x, p.y, p.z, next_id++, typeMap[type]));
      }
      if (n_filtered_particles>0) {
        color_log::highlight(operator_name(),
                             "Number of filtered particles: "
                             + std::to_string(n_filtered_particles));

      }
    }

    MPI_Bcast(&n_particles, 1, MPI_INT, 0, *mpi);
    particle_data.resize(n_particles);
    MPI_Bcast(particle_data.data(), n_particles * sizeof(ParticleTupleIO),
              MPI_BYTE, 0, MPI_COMM_WORLD);

    auto dims = g.dimension();
    long local_particles = 0;
#   pragma omp parallel for reduction(+:local_particles)
    for (auto p : particle_data) {
      Vec3d r{p[field::rx], p[field::ry], p[field::rz]};
      const IJK loc = g.locate_cell(r);

      if (g.contains(loc) && is_inside(d.bounds() , r) && is_inside(g.grid_bounds(), r)) {
        p[field::rx] = r.x;
        p[field::ry] = r.y;
        p[field::rz] = r.z;
        ParticleTuple pt = p;
        size_t cell_i = grid_ijk_to_index(dims, loc);
        cell_locks[cell_i].lock();
        cells[cell_i].push_back(pt, g.cell_allocator());
        cell_locks[cell_i].unlock();
        local_particles++;
      }
    }

    if (particle_data.size()>0) {
      color_log::highlight(operator_name(),
                           "Number of added particles: " +
                           std::to_string(particle_data.size()));
    }

    // Some checks
    long global_particles = 0;
    MPI_Reduce(&local_particles, &global_particles, 1, MPI_LONG, MPI_SUM, 0, *mpi);

    if(rank == 0 && global_particles != (long)particle_data.size()) {
      color_log::error(operator_name(),
                       "Number of added particle is " + 
                       std::to_string(local_particles) +
                       " instead of " +
                       std::to_string(particle_data.size()));
    }

    g.rebuild_particle_offsets();
    assert(check_particles_inside_cell(*grid));
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(exadem_add_xyz) {
  OperatorNodeFactory::instance()->register_factory(
      "add_xyz",
      make_grid_variant_operator<AddXYZ>);
}
}  // namespace exaDEM
