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

#pragma once

#include <iomanip>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <exaDEM/classifier/classifier.hpp>

namespace exaDEM
{

  namespace itools
  {
    using namespace exanb;

    /** CPU only */
    template <typename GridT> std::stringstream create_buffer(GridT &grid, Classifier &ic)
    {
      std::stringstream stream;
      for (int i = 0; i < Classifier::types; i++)
      {
        size_t size = ic.get_size(i);
        InterationPairWrapper wrapper; 
        if( i < Classifier::typesPP ) wrapper.wrap(ic.get_data<ParticleParticle>(i));
        else if( i == Classifier::InnerBondTypeId ) wrapper.wrap(ic.get_data<InnerBond>(i)); 
        else { lout << "skip interaction type: " << i << std::endl; continue; }

        auto [dn_ptr, cp_ptr, fn_ptr, ft_ptr] = ic.buffer_p(i);

        for (size_t idx = 0; idx < size; idx++)
        {
          double dn = dn_ptr[idx];
          /** filter empty interactions */
          if (dn < 0)
          {
            auto [i, j, type, swap, ghost] = wrapper(idx);
            /** Note that an interaction between two particles present on two sub-domains should not be counted twice. */
            if (filter_duplicates(grid, i, j, type))
            {
              stream << i.id << "," << j.id << ",";
              stream << i.sub << "," << j.sub << ",";
              stream << type << ",";
              stream << dn << ",";
              stream << cp_ptr[idx] << ",";
              stream << fn_ptr[idx] << ",";
              stream << ft_ptr[idx] << std::endl;
            }
          }
        }
      }
      return stream;
    }

    inline void write_file(
        std::stringstream &stream, 
        std::string directory, std::string filename)
    {
      int rank, size;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      std::string subdirectory = directory + "/ExaDEMAnalyses";
      std::string subsubdirectory = subdirectory + "/" + filename;
      if (rank == 0)
      {
        namespace fs = std::filesystem;
        fs::create_directory(directory);
        fs::create_directory(subdirectory);
        fs::create_directory(subsubdirectory);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      std::string name = subsubdirectory + "/" + filename + "_" + std::to_string(rank) + ".txt";
      std::ofstream outFile(name);
      if (!outFile)
      {
        color_log::error("itools::write_file", "Impossible to create the output file: " + name, false);
        return;
      }
      outFile << stream.rdbuf();
    }
  } // namespace itools
} // namespace exaDEM
