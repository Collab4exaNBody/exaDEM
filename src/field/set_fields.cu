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

// Onika

#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// exaNBody
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/particle_type_id.h>

// ExaDEM

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exaDEM/color_log.hpp>
#include <exaDEM/set_fields.hpp>
#include <exaDEM/shapes.hpp>
#include <vector>

namespace exaDEM {
namespace set_fields_detail {

// Counter-based random numbers (usable on host and device): the value only depends on (particle id, stream), so it is
// thread-safe, reproducible, and independent of the MPI decomposition and of the type order.
ONIKA_HOST_DEVICE_FUNC inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// uniform value in the open interval (0,1)
ONIKA_HOST_DEVICE_FUNC inline double uniform01(uint64_t id, uint64_t stream) {
  const uint64_t h = mix64(mix64(id) ^ (stream * 0xd2b74407b1ce6e93ULL));
  return (static_cast<double>(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

// Box-Muller, `stream` selects an independent gaussian sequence per particle
ONIKA_HOST_DEVICE_FUNC inline double gaussian(uint64_t id, uint64_t stream, double sigma) {
  const double two_pi = 8.0 * std::atan(1.0);
  const double u1 = uniform01(id, 2 * stream);
  const double u2 = uniform01(id, 2 * stream + 1);
  return sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
}

// uniform random rotation (Shoemake)
ONIKA_HOST_DEVICE_FUNC inline Quaternion random_quaternion_from_id(uint64_t id, uint64_t stream) {
  const double two_pi = 8.0 * std::atan(1.0);
  const double u1 = uniform01(id, stream);
  const double u2 = uniform01(id, stream + 1);
  const double u3 = uniform01(id, stream + 2);
  const double a = std::sqrt(1.0 - u1);
  const double b = std::sqrt(u1);
  Quaternion q;
  q.w = a * std::sin(two_pi * u2);
  q.x = a * std::cos(two_pi * u2);
  q.y = b * std::sin(two_pi * u3);
  q.z = b * std::cos(two_pi * u3);
  return q;
}

// streams used by the generators
constexpr uint64_t stream_velocity = 0;          // 0, 1, 2
constexpr uint64_t stream_angular_velocity = 3;  // 3, 4, 5
constexpr uint64_t stream_quaternion = 6;        // 6, 7, 8 (uniform)

// Everything set on the particles of one type.
struct TypeInit {
  bool defined = false;
  uint32_t group = 0;
  double homothety = 1.0;
  Vec3d velocity = {0, 0, 0};
  double sigma_velocity = 0.0;
  Vec3d angular_velocity = {0, 0, 0};
  double sigma_angular_velocity = 0.0;
  Quaternion quaternion = {1, 0, 0, 0};
  bool random_quaternion = false;
  double mass = 1.0;
  double radius = 1.0;
  Vec3d inertia = {0, 0, 0};
};

// One pass over the grid: each particle looks up the data of its own type.
struct SetFieldsKernel {
  const ParticleRegionCSGShallowCopy region_;
  const bool use_region_;
  const TypeInit* table_;
  const size_t table_size_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(double rx, double ry, double rz, const uint64_t id, uint32_t type,
                                                uint32_t& group, double& homothety, double& vx, double& vy, double& vz,
                                                double& mass, double& radius, Vec3d& vrot, Vec3d& inertia,
                                                Quaternion& orient) const {
    if (type >= table_size_) return;
    const TypeInit& t = table_[type];
    if (!t.defined) return;
    if (use_region_) {
      const Vec3d r = {rx, ry, rz};
      if (!region_.contains(r, id)) return;
    }

    group = t.group;
    homothety = t.homothety;
    mass = t.mass;
    radius = t.radius;
    inertia = t.inertia;

    vx = t.velocity.x;
    vy = t.velocity.y;
    vz = t.velocity.z;
    if (t.sigma_velocity != 0.0) {
      vx += gaussian(id, stream_velocity + 0, t.sigma_velocity);
      vy += gaussian(id, stream_velocity + 1, t.sigma_velocity);
      vz += gaussian(id, stream_velocity + 2, t.sigma_velocity);
    }

    vrot = t.angular_velocity;
    if (t.sigma_angular_velocity != 0.0) {
      vrot.x += gaussian(id, stream_angular_velocity + 0, t.sigma_angular_velocity);
      vrot.y += gaussian(id, stream_angular_velocity + 1, t.sigma_angular_velocity);
      vrot.z += gaussian(id, stream_angular_velocity + 2, t.sigma_angular_velocity);
    }

    orient = t.random_quaternion ? random_quaternion_from_id(id, stream_quaternion) : t.quaternion;
  }
};
}  // namespace set_fields_detail
}  // namespace exaDEM

namespace exanb {
template <>
struct ComputeCellParticlesTraits<exaDEM::set_fields_detail::SetFieldsKernel> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb

namespace exaDEM {

template <typename GridT, class = AssertGridHasFields<GridT, field::_type, field::_group>>
class SetFields : public OperatorNode {
  using ComputeFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_type, field::_group,
                                 field::_homothety, field::_vx, field::_vy, field::_vz, field::_mass, field::_radius,
                                 field::_vrot, field::_inertia, field::_orient>;
  static constexpr ComputeFields compute_fields{};

  ADD_SLOT(GridT, grid, INPUT_OUTPUT);
  ADD_SLOT(std::vector<double>, density, INPUT, OPTIONAL,
           DocString{"List of density values. If not defined, density is 1"});
  ADD_SLOT(std::vector<double>, radius, INPUT, OPTIONAL,
           DocString{"List of radius values. If not defined, radius is 0.5 for "
                     "spheres, do not define it for polyhedra."});
  ADD_SLOT(std::vector<double>, homothety, INPUT, OPTIONAL,
           DocString{"List of homothty values [only used by polyhedra]. If not "
                     "defined, homothety is 1."});
  ADD_SLOT(std::vector<Vec3d>, velocity, INPUT, OPTIONAL,
           DocString{"List of velocity values. If not defined, velocity is [0,0,0]."});
  ADD_SLOT(std::vector<double>, sigma_velocity, INPUT, OPTIONAL,
           DocString{"Standard deviation (sigma). If not defined, the normal "
                     "distribution is not applied."});
  ADD_SLOT(std::vector<Vec3d>, angular_velocity, INPUT, OPTIONAL,
           DocString{"List of angular velocity values. If not defined, angular "
                     "velocity is [0,0,0]."});
  ADD_SLOT(std::vector<double>, sigma_angular_velocity, INPUT, OPTIONAL,
           DocString{"Standard deviation (sigma). If not defined, the normal "
                     "distribution is not applied."});
  ADD_SLOT(std::vector<Quaternion>, quaternion, INPUT, OPTIONAL,
           DocString{"List of orientations. If not defined, quaternion is [w = 1,0,0,0]"});
  ADD_SLOT(std::vector<bool>, random_quaternion, INPUT, OPTIONAL,
           DocString{"Choice if the orientation is random or not. If not "
                     "defined, random is false."});
  ADD_SLOT(std::vector<uint32_t>, group, INPUT, OPTIONAL,
           DocString{"Group index per type. If not defined, group is 0 for all particles."});
  ADD_SLOT(ParticleTypeMap, particle_type_map, INPUT, REQUIRED);
  ADD_SLOT(std::vector<std::string>, type, INPUT, REQUIRED, DocString{"Particle type names"});

  // outputs
  ADD_SLOT(double, rcut_max, INPUT_OUTPUT, 0.0, DocString{"rcut_max"});
  ADD_SLOT(uint32_t, n_groups, OUTPUT, DocString{"Number of distinct groups (max group index + 1)"});

  // others
  ADD_SLOT(bool, polyhedra, INPUT, REQUIRED, DocString{"Define if the kind of particles is polyhedron or sphere."});
  ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
  ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);
  ADD_SLOT(shapes, shapes_collection, INPUT, OPTIONAL, DocString{"Collection of shapes"});

  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
        This operator sets the fields (group, homothety, velocity, angular velocity, mass, radius,
        inertia, orientation) of all particles according to their type.

        YAML examples:

          init_polyhedra:
            - set_fields:
               polyhedra: true
               type:              [ alpha3, Octahedron ]
               group:             [      0,          1 ]
               velocity:          [ [0,0,0],   [0,0,0] ]
               sigma_velocity:    [     0.1,       0.1 ]
               random_quaternion: [    true,      true ]

          init_spheres:
            - set_fields:
               polyhedra: false
               type:           [ Sphere1, Sphere2 ]
               group:          [       0,       1 ]
               radius:         [     0.5,     0.5 ]
               density:        [    0.02,    0.02 ]
               velocity:       [ [0,0,0], [0,0,0] ]
               sigma_velocity: [     0.1,     0.1 ]
               region: Region

        Developer details:

          The values of each type are gathered in a table, then applied to the grid in a single pass.
          Only the particles of a listed type (and inside the region, if any) are modified.
          Random values (sigma_velocity, sigma_angular_velocity, random_quaternion) are generated
          per particle from its id: they do not depend on the number of MPI processes or on the
          order of the types.
      )EOF";
  }

  inline std::string operator_name() { return "set_fields"; }

  template <typename T>
  void check_size(const char* slot_name, const std::vector<T>& v, size_t n_types) {
    if (v.size() != n_types) {
      color_log::error(operator_name(), "The slot [" + std::string(slot_name) + "] has " + std::to_string(v.size()) +
                                            " values, but " + std::to_string(n_types) + " types are defined.");
    }
  }

  void check_slots() {
    if (grid->number_of_cells() == 0) {
      color_log::error(operator_name(),
                       "The grid is not defined. Please define a grid before "
                       "calling set_fields.");
    }

    if (*polyhedra && (!shapes_collection.has_value() || shapes_collection->size() == 0)) {
      color_log::error(operator_name(), "You are defining polyhedra without using shapes.");
    }
    if (!(*polyhedra) && shapes_collection.has_value()) {
      color_log::error(operator_name(), "Shapes are defined in sphere mode.");
    }

    const size_t n_types = type->size();
    if (density.has_value()) check_size("density", *density, n_types);
    if (radius.has_value()) check_size("radius", *radius, n_types);
    if (homothety.has_value()) check_size("homothety", *homothety, n_types);
    if (velocity.has_value()) check_size("velocity", *velocity, n_types);
    if (sigma_velocity.has_value()) check_size("sigma_velocity", *sigma_velocity, n_types);
    if (angular_velocity.has_value()) check_size("angular_velocity", *angular_velocity, n_types);
    if (sigma_angular_velocity.has_value()) check_size("sigma_angular_velocity", *sigma_angular_velocity, n_types);
    if (quaternion.has_value()) check_size("quaternion", *quaternion, n_types);
    if (random_quaternion.has_value()) check_size("random_quaternion", *random_quaternion, n_types);
    if (group.has_value()) check_size("group", *group, n_types);
  }

 public:
  inline void execute() final {
    check_slots();

    const auto& type_map = *particle_type_map;
    const auto& types = *type;

    ParticleRegionCSGShallowCopy prcsg = {};
    const bool is_region = region.has_value();
    if (is_region) {
      if (!particle_regions.has_value()) {
        fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
      }
      if (region->m_nb_operands == 0) {
        region->build_from_expression_string(particle_regions->data(), particle_regions->size());
      }
      prcsg = *region;
    }

    // Number of distinct groups, exposed for downstream operators.
    uint32_t max_group = 0;
    if (group.has_value()) {
      for (auto g : *group) max_group = std::max(max_group, g);
    }
    *n_groups = max_group + 1;

    // Build the table of data per type
    // managed memory: the table is read by the kernel on host or device
    onika::memory::CudaMMVector<set_fields_detail::TypeInit> table;
    lout << "======= Particle Fields =========" << std::endl;
    for (size_t i = 0; i < types.size(); i++) {
      const std::string& type_name = types[i];
      auto it = type_map.find(type_name);
      if (it == type_map.end()) {
        lout << "The type [" << type_name << "] is not defined" << std::endl;
        lout << "Available types are = ";
        for (auto& available : type_map) {
          lout << available.first << " ";
        }
        lout << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const size_t type_id = it->second;
      if (type_id >= table.size()) table.resize(type_id + 1);

      set_fields_detail::TypeInit& t = table[type_id];
      t.defined = true;
      double d = density.has_value() ? (*density)[i] : 1.0;
      if (homothety.has_value()) t.homothety = (*homothety)[i];
      if (velocity.has_value()) t.velocity = (*velocity)[i];
      if (sigma_velocity.has_value()) t.sigma_velocity = (*sigma_velocity)[i];
      if (angular_velocity.has_value()) t.angular_velocity = (*angular_velocity)[i];
      if (sigma_angular_velocity.has_value()) t.sigma_angular_velocity = (*sigma_angular_velocity)[i];
      if (quaternion.has_value()) t.quaternion = (*quaternion)[i];
      if (random_quaternion.has_value()) t.random_quaternion = (*random_quaternion)[i];
      if (group.has_value()) t.group = (*group)[i];

      lout << "[>> " << type_name << " <<]" << std::endl;
      lout << "Velocity         = " << t.velocity;
      if (sigma_velocity.has_value()) lout << ", standard deviation (sigma): " << t.sigma_velocity;
      lout << std::endl;
      lout << "Angular velocity = " << t.angular_velocity;
      if (sigma_angular_velocity.has_value()) lout << ", standard deviation (sigma): " << t.sigma_angular_velocity;
      lout << std::endl;
      lout << "Density          = " << d << std::endl;
      lout << "Homothety        = " << t.homothety << std::endl;
      if (!t.random_quaternion) {
        lout << "Quaternion       = [w: " << t.quaternion.w << ", v: (" << t.quaternion.x << "," << t.quaternion.y
             << "," << t.quaternion.z << ")]" << std::endl;
      } else {
        lout << "Quaternion       = random" << std::endl;
      }

      if (*polyhedra) {
        const shapes& shps = *shapes_collection;
        if (type_id >= shps.size() || shps[type_id]->name_ != type_name) {
          color_log::error(operator_name(), "We can't find the shape related to the type " + type_name +
                                                ". Please verify that you have load all shape files.");
        }
        const auto& shp = shps[type_id];
        t.mass = d * shp->compute_volume(t.homothety);
        t.inertia = t.mass * shp->compute_Im(t.homothety);
        if (radius.has_value()) {
          color_log::warning(operator_name(),
                             "The radius slot is ignored when using polyhedra, "
                             "it is automaticly deducted from the shape file.");
        }
        t.radius = shp->compute_max_rcut();
        lout << "Radius (poly)    = " << t.radius << std::endl;
      } else {
        if (!radius.has_value()) {
          color_log::error(operator_name(), "You should define a radius: radius: \"[1.0]\"");
        }
        t.radius = (*radius)[i];
        const double pi = 4 * std::atan(1);
        const double V = (4.0 / 3.0) * pi * t.radius * t.radius * t.radius;
        t.mass = V * d;
        const double inertia_value = 0.4 * t.mass * t.radius * t.radius;
        t.inertia = {inertia_value, inertia_value, inertia_value};
        lout << "Radius           = " << t.radius << std::endl;
      }
      *rcut_max = std::max(*rcut_max, 2 * t.radius);
      lout << "Mass             = " << t.mass << std::endl;
      lout << "Inertia          = " << t.inertia << std::endl;
    }

    // Apply to the grid in a single pass
    set_fields_detail::SetFieldsKernel func = {prcsg, is_region, table.data(), table.size()};
    compute_cell_particles(*grid, false, func, compute_fields, parallel_execution_context());
    lout << "=================================" << std::endl;
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(set_fields) {
  OperatorNodeFactory::instance()->register_factory("set_fields", make_grid_variant_operator<SetFields>);
}
}  // namespace exaDEM
