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
#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/particle_type_id.h>
#include <exanb/core/xform.h>

#include <exaDEM/drivers.hpp>
#include <exaDEM/dump_rockable_api.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/shape_reader.hpp>
#include <exaDEM/shapes.hpp>
#include <filesystem>  // C++17

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT> >
class DumpWriterConfRockable : public OperatorNode {
  // attributes processed during computation
  using ComputeFields = FieldSet<field::_vrot, field::_arot>;
  static constexpr ComputeFields compute_field_set{};

  ADD_SLOT(GridT, grid,
           INPUT_OUTPUT);
  ADD_SLOT(Domain, domain,
           INPUT);
  ADD_SLOT(MPI_Comm, mpi,
           INPUT, MPI_COMM_WORLD);
  ADD_SLOT(ParticleTypeMap, particle_type_map,
           INPUT);
  ADD_SLOT(Drivers, drivers,
           INPUT_OUTPUT, REQUIRED,
           DocString{"List of Drivers"});
  ADD_SLOT(shapes, shapes_collection,
           INPUT_OUTPUT,
           DocString{"Collection of shapes"});
  ADD_SLOT(std::string, dir_name,
           INPUT, REQUIRED,
           DocString{"Write an Output file containing stress tensors."});
  ADD_SLOT(long, timestep,
           INPUT, REQUIRED,
           DocString{"Iteration number"});
  ADD_SLOT(double, physical_time,
           INPUT, REQUIRED);
  ADD_SLOT(double, dt,
           INPUT, REQUIRED);

 public:
  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
         This operator writes a minimal rockable output file to use the rockable tool `see`.

         YAML example:

         iteration_dump_writer:
           - write_conf_rockable

         No Option.
        )EOF";
  }

  inline void execute() final {
    auto& shps = *shapes_collection;
    auto& drvs = *drivers;
    int rank, mpi_size;
    MPI_Comm_rank(*mpi, &rank);
    MPI_Comm_size(*mpi, &mpi_size);

    bool defbox = !domain->xform_is_identity();
    LinearXForm xform;
    if (defbox) {
      xform.m_matrix = domain->xform();
    }

    shapes all_shapes = shps;
    int n_drivers = 0;          // keep only stl mesh drivers
    std::stringstream sdriver;  // stream for drivers
    for (size_t did = 0; did < drvs.get_size(); did++) {  // driver idx
      if (drvs.type(did) == DRIVER_TYPE::STL_MESH) {
        n_drivers++;
        Stl_mesh& D = drvs.get_typed_driver<Stl_mesh>(did);
        if (!all_shapes.contains(D.shp)) {  // add shape
          all_shapes.add_shape(D.shp);
        }
        sdriver << D.shp.m_name << " " << 0 /* group */ << " "
                << 0 /* cluster */ << " " << 1 /* homothety */ << " "
                << D.center.x << " " << D.center.y << " " << D.center.z << " "
                << D.vel.x << " " << D.vel.y << " " << D.vel.z << " " << D.acc.x
                << " " << D.acc.y << " " << D.acc.z << " " << D.quat.w << " "
                << D.quat.x << " " << D.quat.y << " " << D.quat.z << " "
                << D.vrot.x << " " << D.vrot.y << " " << D.vrot.z << " " << 0
                << " " << 0 << " " << 0; /* arot */
        sdriver << std::endl;
      }
    }

    namespace fs = std::filesystem;
    std::string full_dir = (*dir_name) + "/conf_rockable";
    fs::path path(full_dir);
    fs::create_directory(*dir_name);

    int step = 0;
    std::string filename = full_dir + "/conf" + std::to_string(step);

    if (rank == 0) {
      if (!fs::exists(path)) {
        fs::create_directory(path);
      }
      while (fs::exists(filename)) {
        step++;
        filename = full_dir + "/conf" + std::to_string(step);
      }
      lout << " creating dump " << filename << " ... " << std::endl;
    }

    // what we need:
    // For particles: #name group cluster homothety pos.x pos.y pos.z vel.x
    // vel.y vel.z acc.x acc.y acc.z Q.w Q.x Q.y Q.z vrot.x vrot.y vrot.z arot.x
    // arot.y arot.z For interactions:

    // Local
    std::vector<rockable::Particle> particles;
    const auto& cells = grid->cells();
    IJK dims = grid->dimension();
    const ssize_t gl = grid->ghost_layers();

#   pragma omp parallel
    {
      std::vector<rockable::Particle> TLS_particles;
      GRID_OMP_FOR_BEGIN(dims - 2 * gl, _, loc_no_gl) {
        const IJK loc = loc_no_gl + gl;
        const size_t i = grid_ijk_to_index(dims, loc);
        const size_t n_particles = cells[i].size();
        const auto& cell = cells[i];

        auto* __restrict__ type = cell[field::type];
        double* __restrict__ h = cell[field::homothety];
        double* __restrict__ rx = cell[field::rx];
        double* __restrict__ ry = cell[field::ry];
        double* __restrict__ rz = cell[field::rz];
        double* __restrict__ vx = cell[field::vx];
        double* __restrict__ vy = cell[field::vy];
        double* __restrict__ vz = cell[field::vz];
        double* __restrict__ fx = cell[field::fx];
        double* __restrict__ fy = cell[field::fy];
        double* __restrict__ fz = cell[field::fz];
        Quaternion* __restrict__ quat = cell[field::orient];
        Vec3d* __restrict__ vrot = cell[field::vrot];
        Vec3d* __restrict__ arot = cell[field::arot];

        size_t current_size = TLS_particles.size();
        TLS_particles.resize(current_size + n_particles);
        for (size_t j = 0; j < n_particles; j++) {
          rockable::Particle& p = TLS_particles[current_size + j];
          p.group = 0;
          p.cluster = 0;
          p.type = type[j];
          p.pos = Vec3d(rx[j], ry[j], rz[j]);
          if (defbox) {
            p.pos = xform.transformCoord(p.pos);
          }
          p.vel = Vec3d(vx[j], vy[j], vz[j]);
          p.acc = Vec3d(fx[j], fy[j], fz[j]);
          p.Q = quat[j];
          p.vrot = vrot[j];
          p.arot = arot[j];
          p.homothety = h[j];
        }
      }
      GRID_OMP_FOR_END
#pragma omp critical
      {
        particles.insert(particles.end(), TLS_particles.begin(),
                         TLS_particles.end());
      }
    }

    // MPI stuff
    const int mpi_root = 0;
    std::vector<int> mpi_n_particles(mpi_size, 0);
    std::vector<int> mpi_displ(mpi_size, 0);

    long number_of_particles = particles.size();
    long all_particles = 0;
    MPI_Gather(&number_of_particles, 1, MPI_INT, mpi_n_particles.data(), 1,
               MPI_INT, mpi_root, *mpi);

    if (rank == mpi_root) {
      // get the number of particles in bytes
      all_particles =
          std::accumulate(mpi_n_particles.begin(), mpi_n_particles.end(), 0);
      int tot = 0;
      for (size_t proc = 0; proc < mpi_n_particles.size(); proc++) {
        mpi_n_particles[proc] *= sizeof(rockable::Particle);
        mpi_displ[proc] = tot;
        tot += mpi_n_particles[proc];
      }
    }

    std::vector<rockable::Particle> mpi_particles(all_particles);
    MPI_Gatherv(particles.data(), particles.size() * sizeof(rockable::Particle),
                MPI_CHAR, mpi_particles.data(), mpi_n_particles.data(),
                mpi_displ.data(), MPI_CHAR, mpi_root, *mpi);

    if (rank == 0) {
      std::ofstream file;
      file.open(filename);

      int prec = 13;
      file << "Rockable 29-11-2018" << std::endl;
      file << "t " << *physical_time << std::endl;
      file << "dt " << *dt << std::endl;
      file << "iconf " << step << std::endl;  //*timestep << std::endl;
      file << "periodicity " << domain->periodic_boundary_x() << " "
           << domain->periodic_boundary_y() << " "
           << domain->periodic_boundary_z() << std::endl;
      file << "nDriven 0" << std::endl;
      file << "shapeFile shape.shp" << std::endl;
      file << "nDriven " << n_drivers << std::endl;
      std::stringstream spart;
      spart.precision(prec);
      spart << "precision " << prec << std::endl;
      spart << "Particles " << all_particles + n_drivers << std::endl;
      if (n_drivers > 0) spart << sdriver.rdbuf();

      // write data each 100,000 particles
      for (int p = 0; p < all_particles; p++) {
        rockable::stream(spart, mpi_particles[p], shps);
        spart << std::endl;
      }
      file << spart.rdbuf();

      /** write shapes, it remove old shape file */
      std::string shape_file_path = full_dir + "/shape.shp";
      exaDEM::write_shps(all_shapes, shape_file_path);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(sim_dump_writer_conf_rockable) {
  OperatorNodeFactory::instance()->register_factory(
      "write_conf_rockable",
      make_grid_variant_operator<DumpWriterConfRockable>);
}
}  // namespace exaDEM
