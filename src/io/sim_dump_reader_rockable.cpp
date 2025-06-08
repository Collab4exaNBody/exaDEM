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
# include <chrono>
# include <ctime>
# include <mpi.h>
# include <string>
# include <numeric>

# include <onika/math/basic_types_yaml.h>
# include <onika/scg/operator.h>
# include <onika/scg/operator_slot.h>
# include <onika/scg/operator_factory.h>
# include <exanb/core/make_grid_variant_operator.h>
# include <exanb/core/grid.h>
# include <onika/math/basic_types_stream.h>
# include <onika/log.h>
# include <onika/file_utils.h>
# include <exanb/core/domain.h>
# include <exanb/core/check_particles_inside_cell.h>
# include <exaDEM/shapes.hpp>
# include <exaDEM/dump_rockable_api.hpp>
# include <exaDEM/drivers.h>
# include <exaDEM/stl_mesh.h>

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

  template <typename GridT> class DumpReaderConfRockable : public OperatorNode
  {
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(std::string, filename, INPUT, REQUIRED);
    ADD_SLOT(ReadBoundsSelectionMode, bounds_mode, INPUT, ReadBoundsSelectionMode::FILE_BOUNDS);
    ADD_SLOT(shapes, shapes_collection, OUTPUT, DocString{"Collection of shapes"});
    ADD_SLOT(Domain, domain, INPUT_OUTPUT);
    ADD_SLOT(AABB, bounds, INPUT, OPTIONAL, DocString{"This option overide the domain bounds."});
    ADD_SLOT(double, enlarge_bounds, INPUT, 0.0);
    ADD_SLOT(bool, pbc_adjust_xform, INPUT, false);
    ADD_SLOT(ParticleTypeMap, particle_type_map, OUTPUT );
    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});

    // overloaded slots
    ADD_SLOT(double, physical_time, INPUT_OUTPUT);
    ADD_SLOT(double, dt, INPUT_OUTPUT);

  public:
    inline void execute() override final
    {
      //-------------------------------------------------------------------------------------------
      // Reading datas from YAML or previous input
      std::string file_name = onika::data_file_path(*filename);

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

      using ParticleTupleIO = onika::soatl::FieldTuple<field::_rx, field::_ry, field::_rz, field::_id, field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz, field::_vrot, field::_arot, field::_orient, field::_type,  field::_inertia,  field::_mass,  field::_radius,  field::_homothety >;
      using ParticleTuple = decltype(grid->cells()[0][0]);

      assert(grid->number_of_particles() == 0);

      // MPI Initialization
      int rank = 0, np = 1;
			MPI_Comm_rank(*mpi, &rank);
			MPI_Comm_size(*mpi, &np);

			uint64_t n_particles = 0;
			std::vector<ParticleTupleIO> particle_data;

			rockable::ConfReader manager;
			std::ifstream file;
			file.open(file_name, std::ifstream::in);
			if (!file.is_open())
			{
				lout << "[ERROR, read_conf_rockable] File " << file_name << " not found !" << std::endl;
				std::exit(EXIT_FAILURE);
			}
			manager.read_stream(file);       
			shapes shps = manager.shps;
			*particle_type_map = manager.ptm;
			*shapes_collection = shps;
			if( manager.t > 0.0 ) *physical_time = manager.t;
			if( manager.dt > 0.0 ) *dt = manager.dt;

			if (rank == 0)
			{
				std::vector<rockable::Particle>& rockable_particles = manager.particles;
				double min_x = std::numeric_limits<double>::max();
				double min_y = std::numeric_limits<double>::max();
				double min_z = std::numeric_limits<double>::max();
				double max_x = std::numeric_limits<double>::lowest();
				double max_y = std::numeric_limits<double>::lowest();
				double max_z = std::numeric_limits<double>::lowest();

				n_particles = rockable_particles.size();
				particle_data.resize(n_particles);
				for(size_t p = 0 ; p < rockable_particles.size() ; p++ )
				{
					const rockable::Particle& rp = rockable_particles[p];
					Vec3d& pos = rockable_particles[p].pos;
					min_x = std::min(min_x, pos.x);
					max_x = std::max(max_x, pos.x);

					min_y = std::min(min_y, pos.y);
					max_y = std::max(max_y, pos.y);

					min_z = std::min(min_z, pos.z);
					max_z = std::max(max_z, pos.z);
					ParticleTupleIO& ptio = particle_data[p];
					// ID
					ptio[field::id] = p;
					// positions
					ptio[field::rx] = rp.pos.x;
					ptio[field::ry] = rp.pos.y;
					ptio[field::rz] = rp.pos.z;
					// velocities
					ptio[field::vx] = rp.vel.x;
					ptio[field::vy] = rp.vel.y;
					ptio[field::vz] = rp.vel.z;
					// accelerations
					ptio[field::fx] = rp.acc.x;
					ptio[field::fy] = rp.acc.y;
					ptio[field::fz] = rp.acc.z;
					// angular fields
					ptio[field::orient] = rp.Q;
					ptio[field::vrot] = rp.vrot;
					ptio[field::arot] = rp.arot;
					ptio[field::homothety] = rp.homothety;

					// shapes 
					ptio[field::type] = rp.type;
					const auto& shp = shps[ptio[field::type]];
					double d = manager.densities[ptio[field::type]];
					ptio[field::mass] = d * shp->get_volume();
					ptio[field::inertia] = ptio[field::mass] * shp->get_Im();
					ptio[field::radius] = shp->compute_max_rcut();
				}
 
				AABB particle_bounds = {{min_x, min_y, min_z},{ max_x, max_y, max_z}};
				AABB file_bounds = particle_bounds;
				if(bounds.has_value())
				{
					file_bounds = *bounds; 
				}
				lout << "Domain bounds      = " << file_bounds << std::endl;


				// domain->m_bounds = bounds;
				compute_domain_bounds(*domain, *bounds_mode, *enlarge_bounds, file_bounds, file_bounds, *pbc_adjust_xform);
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
					ldbg << "ID: " << t[field::id] << " pos " << "(" << t[field::rx] << "," << t[field::ry] << "," << t[field::rz] << ")" << std::endl;
					ldbg << "ID: " << t[field::id] << " vel " << "(" << t[field::vx] << "," << t[field::vy] << "," << t[field::vz] << ")" << std::endl;
					ldbg << "ID: " << t[field::id] << " acc " << "(" << t[field::fx] << "," << t[field::fy] << "," << t[field::fz] << ")" << std::endl;
					ldbg << "ID: " << t[field::id] << " quat " << t[field::orient].w << " " << t[field::orient].x << " " << t[field::orient].y << " " << t[field::orient].z << std::endl;
					ldbg << "ID: " << t[field::id] << " vrot " << "(" << t[field::vrot].x << "," << t[field::vrot].y << "," << t[field::vrot].z << ")" << std::endl;
					ldbg << "ID: " << t[field::id] << " arot " << "(" << t[field::arot].x << "," << t[field::arot].y << "," << t[field::arot].z << ")" << std::endl;
					ldbg << "ID: " << t[field::id] << " h " << t[field::homothety] << std::endl;
					ldbg << "ID: " << t[field::id] << " radius " << t[field::radius] << std::endl;
					ldbg << "ID: " << t[field::id] << " mass " << t[field::mass] << std::endl;
					ldbg << "ID: " << t[field::id] << " intertia " << t[field::inertia] << std::endl;
					grid->cell(loc).push_back(t, grid->cell_allocator());
				}
			}

			lout << "============================" << std::endl;
			grid->rebuild_particle_offsets();
			assert(check_particles_inside_cell(*grid));


			// Particles within the "nDriven" section become stl_mesh with a motion type "STATIONARY"
			if( manager.nDriven > 0 )
			{
				auto& drvs = *drivers;
        int next_id = drvs.get_size();
        for(int id = 0 ; id < manager.nDriven ; id++)
        {
          Driver_params motion = Driver_params();
          Stl_params state = Stl_params(); 

          auto& particle = manager.drivers[id];
          state.center = particle.pos;
          state.vel = particle.pos; // will be reset by the motion type
          state.vrot = particle.vrot; // will move evant with the STATIONARY motion type
          state.quat = particle.Q;

          int type = particle.type;
          shape shp = *(manager.shps[type]);
          Stl_mesh driver = {state, motion};
          driver.set_shape(shp);
          driver.initialize();
          drvs.add_driver(next_id + id, driver);
        }
			}
		}
	};

	template <typename GridT> using DumpReaderConfRockableTmpl = DumpReaderConfRockable<GridT>;

	// === register factories ===
	ONIKA_AUTORUN_INIT(sim_dump_writer_conf_rockable)
	{
		OperatorNodeFactory::instance()->register_factory("read_conf_rockable", make_grid_variant_operator<DumpReaderConfRockableTmpl>);
	}
} // namespace exaDEM
