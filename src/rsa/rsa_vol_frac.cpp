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

#include <onika/math/basic_types_yaml.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <onika/math/basic_types_stream.h>
#include <onika/log.h>
#include <onika/file_utils.h>

#include <exanb/core/domain.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/simple_block_rcb.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/core/particle_type_id.h>
#include <mpi.h>

#include <ctime>
#include <string>
#include <numeric>

// exaDEM stuff
#include <exaDEM/color_log.hpp>
#include <exaDEM/shapes.hpp>

// rsa_mpi stuff
#include <rsa_data_storage.hxx>
#include <rsa_random.hxx>
#include <rsa_domain.hxx>
#include <rsa_decoration.hxx>
#include <operator_algorithm.hxx>
#include <radius_generator.hxx>

namespace exaDEM {
struct ParticleType {
  static inline constexpr size_t MAX_STR_LEN = 16;
  double m_mass = 1.0;
  double m_radius = 1.0;
  char m_name[MAX_STR_LEN] = {'\0'};

  inline void set_name(const std::string& s) {
    if (s.length() >= MAX_STR_LEN) {
      std::cerr << "Particle name too long : length=" << s.length() << ", max=" << (MAX_STR_LEN - 1) << "\n";
      std::abort();
    }
    std::strncpy(m_name, s.c_str(), MAX_STR_LEN);
    m_name[MAX_STR_LEN - 1] = '\0';
  }
  inline std::string name() const {
    return m_name;
  }
};

using ParticleTypes = onika::memory::CudaMMVector<ParticleType>;

struct RSAParameters {
  double radius;
  double volume_fraction;
  std::string type;
};
}  // namespace exaDEM

namespace YAML {
using exaDEM::RSAParameters;

template <>
struct convert<RSAParameters> {
  static bool decode(const Node& node, RSAParameters& v) {
    if (node.size() < 2 && node.size() > 3) {
      return false;
    }

    if (node.size() == 3) {
      v.radius = node[0].as<double>();
      v.volume_fraction = node[1].as<double>();
      v.type = node[2].as<std::string>();
    }

    if (node.size() == 2) {
      if (!node["vf"] || !node["type"]) {
        std::string msg = "For polyhedra, please define: { vf: 0.1, type: Particle1}, ";
        msg += "vf is the volume fraction, and uses the option use_shape: ";
        msg += "true if it's not already done.";
        color_log::error("rsa_vol_frac", msg);
      }
      v.radius = -1;
      v.volume_fraction = node["vf"].as<double>();
      v.type = node["type"].as<std::string>();
    }
    return true;
  }
};
}  // namespace YAML

namespace exaDEM {
template <typename GridT>
class RSAVolFrac : public OperatorNode {
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD,
           DocString{"MPI communicator used for parallel RSA particle generation."});
  ADD_SLOT(Domain, domain, INPUT_OUTPUT,
           DocString{"Simulation domain in which particles are inserted. Can be updated during initialization."});
  ADD_SLOT(GridT, grid, INPUT_OUTPUT,
           DocString{"Grid structure to be filled with particles. Will be modified by the RSA operator."});
  ADD_SLOT(ParticleTypeMap, particle_type_map, INPUT, REQUIRED,
           DocString{"Mapping between particle type names and their internal identifiers."});
  ADD_SLOT(double, enlarge_bounds, INPUT, 0.0,
           DocString{"Optional value to enlarge the domain bounds by a fixed margin."});
  ADD_SLOT(ReadBoundsSelectionMode, bounds_mode, INPUT, ReadBoundsSelectionMode::FILE_BOUNDS,
           DocString{"Controls how bounds are interpreted: from file or overridden manually."});
  ADD_SLOT(std::vector<bool>, periodicity, INPUT, OPTIONAL,
           DocString{"If set, overrides the periodicity of the domain stored in the input file."});
  ADD_SLOT(bool, expandable, INPUT, OPTIONAL,
           DocString{"If set, overrides the domain expandability flag stored in the input file."});
  ADD_SLOT(AABB, bounds, INPUT, REQUIRED,
           DocString{"Overrides the domain bounds. Particles outside these bounds will be filtered out."});
  ADD_SLOT(bool, pbc_adjust_xform, INPUT, true,
           DocString{"If true, adjusts particle transformations to enforce periodic boundary conditions."});
  ADD_SLOT(std::vector<RSAParameters>, params, INPUT, REQUIRED,
           DocString{"List of RSA particle parameters. Each entry defines [radius, volume fraction, type name]."});
  ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL,
           DocString{"Optional region-based filtering for particle placement."});
  ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL,
           DocString{
               "Optional CSG (constructive solid geometry) region used to restrict where particles can be placed."});
  ADD_SLOT(shapes, shapes_collection, INPUT, OPTIONAL, DocString{"Collection of shapes"});
  ADD_SLOT(bool, use_shape, INPUT, false,
           DocString{"This option uses the shape data to fill informations like the 'radius'. Please, do not use it "
                     "with spheres."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
      This operator generates a grid populated with a set of particles using the RSA MPI library. Each particle set is defined by a radius, a volume fraction, and a particle type name. This operator should only be used during the initialization phase of a simulation."

      For spheres:

        - particle_type:
           type: [ Sphere0 , Sphere1 , Sphere2 ]
        - rsa_vol_frac:
            periodic: [true,true,false]
            bounds: [ [ 10 , 10 , 10 ] , [ 14, 14, 14] ]
            params: [[0.5, 0.1, Sphere2], [0.25, 0.1, Sphere1], [0.125, 0.1, Sphere0]]

      For polyhedra:
        - read_shape_file:
           filename: shapes.shp
           rename: [PolyR, Octahedron]
        - read_shape_file:
           filename: shapes.shp
           rename: [ PolyRSize2, OctahedronSize2]
           scale:  [        2.0,             2.0]
        - rsa_vol_frac:
           bounds: [ [0 ,0 , 0 ], [40 , 10 , 40 ] ]
           params: [ {vf: 0.055,           type: PolyR},
                     {vf: 0.055,      type: Octahedron},
                     {vf: 0.055,      type: PolyRSize2},
                     {vf: 0.055, type: OctahedronSize2}]
           use_shape: true
        )EOF";
  }

  std::string operator_name() { return "rsa_vol_frac"; }

  void type_not_found(std::string type_name) {
    color_log::error(operator_name(), "The type [" + type_name + "] is not defined", false);
    std::string msg = "Available types are = ";
    for (auto& it : *particle_type_map) {
      msg += it.first + " ";
    }
    msg += ".";
    color_log::error(operator_name(), msg);
  }

  inline void execute() final {
    //-------------------------------------------------------------------------------------------
    using ParticleTupleIO =
        onika::soatl::FieldTuple<field::_rx, field::_ry, field::_rz, field::_id, field::_type, field::_radius>;
    using ParticleTuple = decltype(grid->cells()[0][0]);

    assert(grid->number_of_particles() == 0);

    // Get the name of the particle types
    const auto type_map = *particle_type_map;

    // Get MPI variables
    int rank = 0, np = 1;
    MPI_Comm_rank(*mpi, &rank);
    MPI_Comm_size(*mpi, &np);

    AABB b = *bounds;
    constexpr int DIM = 3;
    constexpr int method = 1;
    constexpr int ghost_layer = 1;

    // domain size
    std::array<double, DIM> domain_inf = {b.bmin.x, b.bmin.y, b.bmin.z};
    std::array<double, DIM> domain_sup = {b.bmax.x, b.bmax.y, b.bmax.z};

    // check domain size
    double dcs = domain->cell_size();
    for (int dim = 0; dim < DIM; dim++) {
      if (std::fmod(domain_sup[dim] - domain_inf[dim], dcs) > 1e-14) {
        lout << "\033[1;33mThe domain may be ill-formed. Please specify a domain that is a multiple of the cell size ("
             << dcs << "). If you want to define a subdomain, please use a region.\033[0m" << std::endl;
      }
    }

    // get the list of parameters
    std::vector<RSAParameters> list = *params;

    // Fill radius if the option use_shape is activated
    if (*use_shape) {
      const shapes& shps = *shapes_collection;
      for (auto& it : list) {
        auto type = type_map.find(it.type);
        if (type == type_map.end()) {
          type_not_found(it.type);
        }

        it.radius = shps[type->second]->compute_max_rcut();
      }
    }

    // Check that radius are well-defined
    for (auto& it : list) {
      if (it.radius <= 0.0) {
        std::string msg = "radius is negative for type ";
        msg += it.type;
        msg += ". Please, verify that use_shape is set to true.";
        color_log::error("rsa_vol_frac", msg);
      }
    }

    double r_max = 0.0;
    for (auto& it : list) {
      r_max = std::max(r_max, it.radius);
    }

    // gen
    rsa_domain<DIM> rsa_domain(domain_inf, domain_sup, ghost_layer, r_max);
    std::vector<tuple<double, double, int>> cast_list;

    for (auto& it : list) {
      std::string type_name = it.type;
      auto type = type_map.find(type_name);
      if (type == type_map.end()) {
        type_not_found(type_name);
      }
      cast_list.push_back(make_tuple(it.radius, it.volume_fraction, type->second));
      std::sort(cast_list.begin(), cast_list.end(),
                [](const tuple<double, double, int>& a, const tuple<double, double, int>& b) -> bool {
                  return std::get<0>(a) > std::get<0>(b);
                });
    }
    for (auto& it : cast_list) {
      lout << "RSA Parameters, Type: " << std::get<2>(it) << ", radius: " << std::get<0>(it)
           << ", volume fraction: " << std::get<1>(it) << std::endl;
    }

    sac_de_billes::RadiusGenerator<DIM> radius_generator(cast_list, rsa_domain.get_total_volume());

    size_t seed = 0;
    algorithm::uniform_generate<DIM, method>(rsa_domain, radius_generator, 6000, 10, seed);
    auto spheres = rsa_domain.extract_spheres();

    if (rank == 0) {
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
    for (size_t s = 0; s < spheres.size(); s++) {
      auto pos = spheres[s].center;
      auto rad = spheres[s].radius;
      auto type = spheres[s].phase;
      auto id = ns + s;
      pt = ParticleTupleIO(pos[0], pos[1], pos[2], id, type, rad);
      particle_data.push_back(pt);
    }

    bool is_region = region.has_value();
    ParticleRegionCSGShallowCopy prcsg;
    if (is_region) {
      if (!particle_regions.has_value()) {
        fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
      }

      if (region->m_nb_operands == 0) {
        ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
        region->build_from_expression_string(particle_regions->data(), particle_regions->size());
      }
      prcsg = *region;
    }

    // Fill grid, particles will migrate accross mpi processed
    // using the operator migrate_cell_particles
    for (auto p : particle_data) {
      Vec3d r{p[field::rx], p[field::ry], p[field::rz]};
      IJK loc = domain_periodic_location(*domain, r);  // grid.locate_cell(r);
      assert(grid->contains(loc));
      assert(min_distance2_between(r, grid->cell_bounds(loc)) < grid->epsilon_cell_size2());
      p[field::rx] = r.x;
      p[field::ry] = r.y;
      p[field::rz] = r.z;
      ParticleTuple t = p;
      t[field::homothety] = 1.0;
      if (is_region) {
        if (prcsg.contains(r)) {
          grid->cell(loc).push_back(t, grid->cell_allocator());
        }
      } else {
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
    lout << "Real size        = "
        << bounds_size(domain->bounds()) * Vec3d{domain->xform().m11, domain->xform().m22, domain->xform().m33}
    << std::endl;
    lout << "Cell size        = " << domain->cell_size() << std::endl;
    lout << "Grid dimensions  = " << domain->grid_dimension() << " (" << grid_cell_count(domain->grid_dimension())
        << " cells)" << std::endl;
    lout << "=================================" << std::endl;
    grid->rebuild_particle_offsets();
  }
};

// === register factories ===
__attribute__((constructor)) static void register_factories() {
  OperatorNodeFactory::instance()->register_factory("rsa_vol_frac", make_grid_variant_operator<RSAVolFrac>);
}
}  // namespace exaDEM
