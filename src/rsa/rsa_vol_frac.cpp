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
#include <exanb/grid_cell_particles/particle_region.h>


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

	struct RSAParameters{
		double radius;
		double volume_fraction;
		int type;
	};
}

namespace YAML
{
	using exaDEM::RSAParameters;

	template <> struct convert<RSAParameters>
	{ 
		static bool decode(const Node &node, RSAParameters &v)
		{
			if (node.size() != 3)
			{
				return false;
			}
			v.radius = node[0].as<double>();
			v.volume_fraction = node[1].as<double>();
			v.type = node[2].as<int>();
			return true;
		}
	};
}

namespace exaDEM
{
	template <typename GridT> class RSAVolFrac : public OperatorNode
	{
		ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
		ADD_SLOT(Domain, domain, INPUT_OUTPUT);
		ADD_SLOT(GridT, grid, INPUT_OUTPUT);
		ADD_SLOT(double, enlarge_bounds, INPUT, 0.0);
		ADD_SLOT(ReadBoundsSelectionMode, bounds_mode, INPUT, ReadBoundsSelectionMode::FILE_BOUNDS);
		ADD_SLOT(std::vector<bool>, periodicity, INPUT, OPTIONAL, DocString{"if set, overrides domain's periodicity stored in file with this value"});
		ADD_SLOT(bool, expandable, INPUT, OPTIONAL, DocString{"if set, override domain expandability stored in file"});
		ADD_SLOT(AABB, bounds, INPUT, REQUIRED, DocString{"if set, override domain's bounds, filtering out particle outside of overriden bounds"});
		ADD_SLOT(bool, pbc_adjust_xform, INPUT, true);

		ADD_SLOT(std::vector<RSAParameters>, params, INPUT, REQUIRED, DocString{"List of RSA particle parameters, where each entry is [radius, volume fraction, type]."});
		ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
		ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);

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
			double exclusion_distance = 0.;

			// domain size
			std::array<double, DIM> domain_inf = {b.bmin.x, b.bmin.y, b.bmin.z};
			std::array<double, DIM> domain_sup = {b.bmax.x, b.bmax.y, b.bmax.z};

			// get the list of parameters
			std::vector<RSAParameters> list = *params;
			std::sort(list.begin(), list.end(), [](RSAParameters& a, RSAParameters& b) -> bool {return b.radius < a.radius;});

			// gen
			rsa_domain<DIM> rsa_domain(domain_inf, domain_sup, ghost_layer, list[0].radius);
			std::vector<tuple<double, double, int>> cast_list;
			for( auto& it : list ) cast_list.push_back(make_tuple(it.radius, it.volume_fraction, it.type));



      for( auto& it : cast_list) lout << "RSA Parameters, Type : " << std::get<2>(it) << " radius: " << std::get<0>(it) << " volume fraction: " << std::get<1>(it) << std::endl;
  
			sac_de_billes::RadiusGenerator<DIM> radius_generator(cast_list, rsa_domain.get_total_volume());  

			size_t seed = 0;
			algorithm::uniform_generate<DIM, method>(rsa_domain, radius_generator, 6000, 10, seed);
			auto spheres = rsa_domain.extract_spheres();

			if (rank == 0)
			{
				compute_domain_bounds(*domain, *bounds_mode, *enlarge_bounds, b, b, *pbc_adjust_xform);
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
			for (size_t s = 0; s < spheres.size(); s++)
			{
				auto pos = spheres[s].center;
				auto rad = spheres[s].radius;
				auto type = spheres[s].phase;
				auto id = ns + s;
				pt = ParticleTupleIO(pos[0], pos[1], pos[2], id, type, rad);
				particle_data.push_back(pt);
			}

			bool is_region = region.has_value();
			ParticleRegionCSGShallowCopy prcsg;
			if( is_region )
			{
				if (!particle_regions.has_value())
				{
					fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
				}

				if (region->m_nb_operands == 0)
				{
					ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
					region->build_from_expression_string(particle_regions->data(), particle_regions->size());
				}
				prcsg =  *region;
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
				if( is_region )
				{
					if( prcsg.contains(r) )
					{
						grid->cell(loc).push_back(t, grid->cell_allocator());
					}
				}
				else
				{
					grid->cell(loc).push_back(t, grid->cell_allocator());
				}
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
		}
	};

	// === register factories ===
	__attribute__((constructor)) static void register_factories() { OperatorNodeFactory::instance()->register_factory("rsa_vol_frac", make_grid_variant_operator<RSAVolFrac>); }

} // namespace exaDEM
