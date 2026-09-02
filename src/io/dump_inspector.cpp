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
#include <onika/file_utils.h>
#include <onika/log.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exanb/core/domain.h>
#include <exanb/io/mpi_file_io.h>
#include <exanb/io/sim_dump_io.h>
#include <string>

namespace exaDEM {
using namespace exanb;

/**
 * @brief Reads a .dump checkpoint file's header and prints its content.
 *
 * Only the header is read (particle data is never touched), so this operator does not need to
 * know the field set the dump was written with -- unlike read_dump_particle_interaction and
 * friends, which decode particle data and therefore require a compile-time-known field set.
 */
class DumpInspectorNode : public OperatorNode {
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(std::string, filename, INPUT, REQUIRED, DocString{"Path to the .dump checkpoint file to inspect."});

 public:
  inline bool is_sink() const final { return true; }

  inline std::string documentation() const final {
    return R"EOF(
        Reads a .dump checkpoint file's header and prints its content (version, particle
        count, timestep, physical time, field list, domain). No particle data is read.

        YAML example:

          - dump_inspector:
             filename: CheckpointFiles/exadem_0000012345.dump
      )EOF";
  }

  inline void execute() final {
    std::string file_name = onika::data_file_path(*filename);

    MpiIO file;
    file.open(*mpi, file_name, "r");

    SimDumpHeader header = {};
    file.read(&header);
    file.increment_offset(&header);
    header.post_process();

    file.close();

    lout << "============ " << file_name << " ============" << std::endl;
    lout << "format version = " << header.m_version / 1000 << "." << header.m_version % 1000 << std::endl;
    lout << "particles      = " << header.m_nb_particles << std::endl;
    lout << "time step      = " << header.m_time_step << std::endl;
    lout << "time           = " << header.m_time << std::endl;
    lout << "tuple size     = " << header.m_tuple_size << " bytes" << std::endl;

    lout << "fields (" << header.m_nb_fields << ")   =";
    for (uint32_t i = 0; i < header.m_nb_fields; i++) {
      lout << " " << header.m_fields[i] << "(" << header.m_field_size[i] << "B)";
    }
    lout << std::endl;

    const Domain& domain = header.m_domain;
    lout << "domain bounds  = " << domain.bounds() << " , size=" << domain.bounds_size() << std::endl;
    if (!domain.xform_is_identity()) {
      lout << "domain xform   = " << domain.xform() << " , inv = " << domain.inv_xform() << std::endl;
    }
    lout << "domain         = " << domain << std::endl;

    lout << "opt. size      = " << header.m_optional_header_size << " bytes" << std::endl;
    lout << "data chunks    = " << header.m_chunk_count << std::endl;

    ldbg << "opt. header @  = " << header.m_optional_offset << std::endl;
    ldbg << "chunk table @  = " << header.m_table_offset << std::endl;
    ldbg << "part. data @   = " << header.m_data_offset << std::endl;
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(dump_inspector) {
  OperatorNodeFactory::instance()->register_factory("dump_inspector", make_simple_operator<DumpInspectorNode>);
}
}  // namespace exaDEM
