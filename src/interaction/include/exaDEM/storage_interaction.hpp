#pragma once

#include <iomanip>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <exaDEM/interaction/classifier.hpp>


namespace exaDEM
{
  using namespace exanb;
  /** CPU only */
  std::stringstream create_buffer(Classifier& ic)
  {
    std::stringstream stream;
    const int ntypes = ic.number_of_waves();
    for( int i = 0 ; i < ntypes ; i++ )  
    {
      auto [i_ptr, size] = ic.get_info(i);
      auto [co_ptr, fn_ptr, ft_ptr] = ic.buffer_p(i); 

      for(size_t idx = 0 ; idx < size ; idx++)
      {
        Vec3d fn = fn_ptr[idx];
        Vec3d ft = ft_ptr[idx];
        /** filter empty interactions */
        if(fn != Vec3d{0,0,0} || ft != Vec3d{0,0,0})
        {
          auto& I = i_ptr[idx];
          stream << I.id_i  << "," << I.id_j << ","; 
          stream << I.sub_i << "," << I.sub_j << ","; 
          stream << I.type << ","; 
          stream << co_ptr[idx] << ","; 
          stream << fn << ","; 
          stream << ft << std::endl;
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

    std::string directory = "ExaDEMAnalysis";
    if(rank == 0)
    {
      namespace fs = std::filesystem;
      fs::create_directory(directory);
    }
    std::string name = directory  + "/" + filename + ".txt";
    std::ofstream outFile(name);
    if (!outFile) {
      lerr << "Error : impossible to create the output file: " << name << std::endl;
      return;
    }
    outFile << stream.rdbuf();
  }
}
