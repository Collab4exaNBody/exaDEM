#pragma once

#include <iomanip>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <exaDEM/interaction/classifier.hpp>

namespace exaDEM
{

  namespace itools
  {
    using namespace exanb;

    /** CPU only */
    template<typename GridT>
      std::stringstream create_buffer(GridT& grid, Classifier& ic)
      {
        std::stringstream stream;
        const int ntypes = ic.number_of_waves();
        for( int i = 0 ; i < ntypes ; i++ )  
        {
          auto [i_ptr, size] = ic.get_info(i);
          auto [dn_ptr, cp_ptr, fn_ptr, ft_ptr] = ic.buffer_p(i); 

          for(size_t idx = 0 ; idx < size ; idx++)
          {
            double dn = dn_ptr[idx];
            /** filter empty interactions */
            if(dn < 0)
            {
              auto& I = i_ptr[idx];
              /** Note that an interaction between two particles present on two sub-domains should not be counted twice. */
              if(filter_duplicates(grid, I))
              {
                stream << I.id_i  << "," << I.id_j << ","; 
                stream << I.sub_i << "," << I.sub_j << ","; 
                stream << I.type << ","; 
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

    void write_file(std::stringstream& stream, std::string filename)
    {
      int rank, size;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      std::string directory = "ExaDEMOutputDir";
      std::string subdirectory = directory + "/ExaDEMAnalyses";
      std::string subsubdirectory = subdirectory + "/" + filename;
      if(rank == 0)
      {
        namespace fs = std::filesystem;
        fs::create_directory(directory);
        fs::create_directory(subdirectory);
        fs::create_directory(subsubdirectory);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      std::string name = subsubdirectory  + "/" + filename +  "_" + std::to_string(rank) +  ".txt";
      std::ofstream outFile(name);
      if (!outFile) {
        lerr << "Error : impossible to create the output file: " << name << std::endl;
        return;
      }
      outFile << stream.rdbuf();
    }
  }
}
