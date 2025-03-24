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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/particle_type_id.h>
#include <memory>
#include <exaDEM/shapes.hpp>
#include <exaDEM/set_fields.h>
#include <exaDEM/random_quaternion.h>

namespace exaDEM
{

  struct jammy
  {
    jammy(double var) { dist = std::normal_distribution<>(0, var); }

    inline int operator()(double &val)
    {
      val += dist(seed);
      seed();
      return 0;
    }

    inline int operator()(Vec3d &val)
    {
      val.x += dist(seed);
      seed();
      val.y += dist(seed);
      seed();
      val.z += dist(seed);
      seed();
      return 0;
    }

    std::normal_distribution<> dist;
    std::default_random_engine seed;
  };


  struct field_manager
  {
    bool set_t = false; // type
    bool set_d = false; // density
    bool set_v = false; // velocity
    bool set_rnd_v = false;
    bool set_r = false; // radius
    bool set_q = false; // quaternion
    bool set_rnd_q = false;
    bool set_i = true; // inertia (should be to true)
    bool set_ang_v = false;
    bool set_rnd_ang_v = false;
  };

  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_type>> class SetFields : public OperatorNode
  {
    // fields : vx, vy, vz, mass, radius, anv, inertia, quat
    using ComputeFields = FieldSet<field::_type, field::_vx, field::_vy, field::_vz, field::_mass, field::_radius, field::_vrot, field::_inertia, field::_orient>;
    using ComputeRegionFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_type, field::_vx, field::_vy, field::_vz, field::_mass, field::_radius, field::_vrot, field::_inertia, field::_orient>;
    static constexpr ComputeFields compute_fields{};
    static constexpr ComputeRegionFields compute_region_fields{};
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    // vector version
    ADD_SLOT(std::vector<double>, density, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<double>, radius, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<Vec3d>, velocity, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<double>, sigma_velocity, INPUT, OPTIONAL, DocString{"Standard deviation"});
    ADD_SLOT(std::vector<Vec3d>, angular_velocity, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<double>, sigma_angular_velocity, INPUT, OPTIONAL, DocString{"Standard deviation"});
    ADD_SLOT(std::vector<Quaternion>, quaternion, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<bool>, random_quaternion, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(ParticleTypeMap, particle_type_map, INPUT, REQUIRED );
    ADD_SLOT(std::vector<std::string>, type, INPUT, REQUIRED);

    // outputs
    ADD_SLOT(double, rcut_max, INPUT_OUTPUT, DocString{"rcut_max"});

    // others
    ADD_SLOT(bool, polyhedra, INPUT, REQUIRED, DocString{""});
    ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
    ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);
    ADD_SLOT(shapes, shapes_collection, INPUT, OPTIONAL, DocString{"Collection of shapes"});

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator fills type id to all particles. 
        )EOF";
    }

    public:
    inline void execute() override final
    {

      const auto& type_map = *particle_type_map; 
      const auto& types = *type;

      bool is_region  = region.has_value();


      if(shapes_collection.has_value())
      {
        if(!(*polyhedra))
        {
          lout << "[ERROR], shapes are defined in sphere mode" << std::endl;
          std::exit(EXIT_FAILURE);  
        }
        size_t size_shps = shapes_collection->get_size();
        if(size_shps == 0 && (*polyhedra))
        {
          lout << "[ERROR], you are defining polyhedra without using shapes" << std::endl;
          std::exit(EXIT_FAILURE);  
        }
      }

      field_manager mat; // multi-materials
      mat.set_t         = type.has_value();
      mat.set_d         = density.has_value();
      mat.set_r         = radius.has_value();
      mat.set_v         = velocity.has_value();
      mat.set_rnd_v     = sigma_velocity.has_value();
      mat.set_ang_v     = angular_velocity.has_value();
      mat.set_rnd_ang_v = sigma_angular_velocity.has_value();
      mat.set_q         = quaternion.has_value();
      mat.set_rnd_q     = random_quaternion.has_value();

      for(auto& type_name : types)
      {
        if( type_map.find(type_name) == type_map.end())
        {
          lout << "The type [" << type_name << "] is not defined" << std::endl;
          lout << "Available types are = ";
          for(auto& it : type_map) lout << it.first << " ";
          lout << std::endl;
          std::exit(EXIT_FAILURE);  
        }
        int64_t type_id = type_map.at(type_name); 
        // default values;
        double vx = 0;
        double vy = 0;
        double vz = 0;
        double r = 1.0; // it will be replaced if the polyhedra is to true.
        double d = 1.0;
        double m = 1.0;
        Vec3d ang_v = {0,0,0};
        Quaternion quat = {1,0,0,0};
        Vec3d inertia;
        double sigma_v, sigma_ang_v;

        if(mat.set_d) { auto& dd = *density; d = dd[type_id]; }
        if(mat.set_v) { auto& vv = *velocity; const Vec3d& v = vv[type_id]; vx = v.x; vy = v.y; vz = v.z; }
        if(mat.set_ang_v) { auto& ang_vv = *angular_velocity; ang_v = ang_vv[type_id]; }
        if(mat.set_q) { auto& qq = *quaternion; quat = qq[type_id]; }


        lout << "Particle Initialization: " << std::endl;
        lout << "["<<type_name<<"]:" << std::endl;;
        lout << "- velocity: (" << vx << "," << vy << "," << vz << ") ";
        if(mat.set_rnd_v)
        {
          sigma_v = (*sigma_velocity)[type_id];
          lout << ", standart deviation (sigma): " << sigma_v;
        }
        lout << std::endl; 
        lout << "- angular velocity: " << ang_v;
        if(mat.set_rnd_ang_v)
        {
          sigma_ang_v = (*sigma_angular_velocity)[type_id];
          lout << ", standart deviation (sigma): " << sigma_ang_v;
        }
        lout << std::endl; 
        lout << "- density: " << d << std::endl;;
        if( !mat.set_rnd_q ) lout << "- quaternion: [w: " << quat.w << ", v: (" << quat.x << "," << quat.y << "," << quat.z << ")]" ;
        else lout << "- quaternion: random";
        lout << std::endl; 


        if(*polyhedra)
        {
          const shapes& shps = *shapes_collection;
          const auto& shp = shps[type_id];
          m         = d * shp->get_volume();
          inertia   = m * shp->get_Im();

          if( mat.set_r ) { lout << "[WARNING] The radius slot is ignored when using polyhedra, it is automaticly deducted from the shape file."<< std::endl; }
          r = shp->compute_max_rcut();
          *rcut_max = std::max(*rcut_max, 2 * r); // r * maxrcut
          lout << "- radius (polyhedron): " << r << std::endl;;
          lout << "- mass: " << m << std::endl;
          lout << "- inertia: " << inertia << std::endl;
        }
        else // spheres
        {
          if(!mat.set_r) { lout << "You should define a radius: radius: \"[1.0]\"" ; std::exit(0); }
          else
          { 
            auto& rr = *radius; 
            r = rr[type_id]; 
          }
          *rcut_max = std::max(*rcut_max, 2 * r); // r * maxrcut
          const double pi = 4 * std::atan(1);
          const double V = ((4.0)/(3.0)) * pi * r * r * r;
          m = V  * d ;
          const double inertia_value = 0.4 * m * r * r;
          inertia = {inertia_value, inertia_value, inertia_value};
          lout << "- radius: " << r << std::endl;
          lout << "- mass: " << m << std::endl;
          lout << "- inertia: " << inertia << std::endl;
        }

        lout << std::endl;
        if (is_region)
        {
          ParticleRegionCSGShallowCopy prcsg = *region;
          if (!particle_regions.has_value())
          {
            fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
          }

          if (region->m_nb_operands == 0)
          {
            ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
            region->build_from_expression_string(particle_regions->data(), particle_regions->size());
          }

          // fields : vx, vy, vz, mass, radius, anv, inertia, quat
          FilteredSetRegionFunctor<double,double, double, double, double, Vec3d, Vec3d, Quaternion> func = {prcsg, uint32_t(type_id), {vx, vy, vz, m, r, ang_v, inertia, quat}};
          compute_cell_particles(*grid, false, func, compute_region_fields, parallel_execution_context());

          if(mat.set_rnd_v)
          {
            jammy gen(sigma_v);
            FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz> compute_rnd_v;
            GenSetRegionFunctor<jammy> generator = {prcsg, gen};
            compute_cell_particles(*grid, false, generator, compute_rnd_v, parallel_execution_context());
          }

          if(mat.set_rnd_ang_v)
          {
            jammy gen(sigma_ang_v);
            FieldSet<field::_rx, field::_ry, field::_rz, field::_vrot> compute_rnd_ang_v;
            GenSetRegionFunctor<jammy> generator = {prcsg, gen};
            compute_cell_particles(*grid, false, generator, compute_rnd_ang_v, parallel_execution_context());
          }

          if(mat.set_rnd_q) /** Random Quaternion */
          {
            FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_orient> compute_orient;
            RandomQuaternionFunctor RndQuatFunc = {prcsg};
            compute_cell_particles(*grid, false, RndQuatFunc, compute_orient, parallel_execution_context());
          }
        }
        else // no region
        {
          FilteredSetFunctor<double, double, double, double, double, Vec3d, Vec3d, Quaternion> func = {uint32_t(type_id), {vx, vy, vz, m, r, ang_v, inertia, quat}};
          compute_cell_particles(*grid, false, func, compute_fields, parallel_execution_context());

          if(mat.set_rnd_v)
          {
            jammy gen(sigma_v);
            FieldSet<field::_vx, field::_vy, field::_vz> compute_rnd_v;
            GenSetFunctor<jammy> generator = {gen};
            compute_cell_particles(*grid, false, generator, compute_rnd_v, parallel_execution_context());
          }

          if(mat.set_rnd_q) /** Random Quaternion */
          {
            FieldSet<field::_orient> compute_orient;
            RandomQuaternionFunctor RndQuatFunc = {};
            compute_cell_particles(*grid, false, RndQuatFunc, compute_orient, parallel_execution_context());
          }
        }
      }
    }
  };

  template <class GridT> using SetFieldsTmpl = SetFields<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(set_fields) { OperatorNodeFactory::instance()->register_factory("set_fields", make_grid_variant_operator<SetFieldsTmpl>); }

} // namespace exaDEM
