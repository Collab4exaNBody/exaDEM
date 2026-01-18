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
#include <exanb/core/domain.h>

#include <exanb/compute/field_combiners.h>
#include <exanb/core/particle_type_properties.h>
#include <exanb/core/grid_additional_fields.h>

#include <exanb/io/vtk_writer.h>
#include <exanb/io/vtk_writer_binary.h>
#include <exanb/io/vtk_writer_ascii.h>
#include <exanb/io/write_paraview.h>

#include <mpi.h>
#include <string>
#include <regex>

namespace exanb {

template <typename GridT>
class ParaviewGenericWriter : public OperatorNode {
  using StringList = std::vector<std::string>;

  ADD_SLOT(MPI_Comm, mpi, INPUT);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(ParticleTypeProperties, particle_type_properties, INPUT, OPTIONAL);
  ADD_SLOT(Domain, domain, INPUT);
  ADD_SLOT(bool, binary_mode, INPUT, true);
  ADD_SLOT(bool, write_box, INPUT, true);
  ADD_SLOT(bool, write_ghost, INPUT, false);
  ADD_SLOT(std::string, compression, INPUT, "default");
  ADD_SLOT(std::string, filename, INPUT, "output");  // default value for backward compatibility
  ADD_SLOT(StringList, fields, INPUT, StringList({".*"}),
           DocString{"List of regular expressions to select fields to project"});

  template <class... GridFields>
  inline void execute_on_fields(const GridFields&... grid_fields) {
    using has_field_type_t = typename GridT::template HasField<field::_type>;
    static constexpr bool has_field_type = has_field_type_t::value;

    ldbg << "ParaviewGenericWriter: filename=" << *filename << " , write_box=" << std::boolalpha << *write_box
         << " , write_ghost=" << std::boolalpha << *write_ghost << " , binary_mode=" << std::boolalpha << *binary_mode
         << std::endl;

    {
      int s = 0;
      ldbg << "Paraview writer available fields:";
      (..., (ldbg << (((s++) == 0) ? ' ' : ',') << grid_fields.short_name()));
      ldbg << std::endl;
    }

    // field selector function
    const auto& flist = *fields;
    auto field_selector = [&flist](const std::string& name) -> bool {
      for (const auto& f : flist)
        if (std::regex_match(name, std::regex(f))) return true;
      return false;
    };

    // per cell particle attribute accessor
    auto gridacc = grid->cells_accessor();

    // optional fields
    ParticleTypeProperties* optional_type_properties = nullptr;
    if (has_field_type && particle_type_properties.has_value())
      optional_type_properties = particle_type_properties.get_pointer();
    GridAdditionalFields add_fields(grid, optional_type_properties);
    auto [type_real_fields, type_vec3_fields, type_mat3_fields, opt_real_fields, opt_vec3_fields, opt_mat3_fields] =
        add_fields.view();

    ParaviewWriteTools::write_particles(ldbg, *mpi, *grid, gridacc, *domain, *filename, field_selector, *compression,
                                        *binary_mode, *write_box, *write_ghost, type_real_fields, type_vec3_fields,
                                        type_mat3_fields, opt_real_fields, opt_vec3_fields, opt_mat3_fields,
                                        grid_fields...);
  }

  template <class... fid>
  inline void execute_on_field_set(FieldSet<fid...>) {
    int rank = 0;
    MPI_Comm_rank(*mpi, &rank);
    ProcessorRankCombiner processor_id = {{rank}};
    VelocityVec3Combiner velocity = {};
    ForceVec3Combiner force = {};
    execute_on_fields(processor_id, velocity, force, onika::soatl::FieldId<fid>{}...);
  }

 public:
  inline void execute() final {
    using GridFieldSet =
        RemoveFields<typename GridT::field_set_t,
                     FieldSet<field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz> >;
    execute_on_field_set(GridFieldSet{});
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(exadem_write_paraview) {
  // write_paraview -> Ambiguous name when connecting applications
  OperatorNodeFactory::instance()->register_factory("exadem_write_paraview",
                                                    make_grid_variant_operator<ParaviewGenericWriter>);
}

}  // namespace exanb
