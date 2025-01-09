#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mpi.h>
#include <exanb/core/string_utils.h>

namespace exaDEM
{
  namespace analysis
  {

    struct AnalysisFileManager
    {
      std::filesystem::path path;
      std::string filename ;
      std::stringstream line;
      std::stringstream header;

      void set_path(std::string p) { path = p;}
      void set_filename(std::string f) { filename = f;}

      bool first()
      {
        std::string full_name = path.string() + "/" + this->filename;
        return !std::filesystem::exists(full_name);
      }

     template <typename T>
      void add_element(std::string name, T& new_element, std::string format)
      {
        header << name << " ";
        std::string element = exanb::format_string(format, new_element);
        line << element << " "; 
      }

      void create_directories()
      {
        exanb::ldbg << "create directory " << this->path << std::endl;
        std::filesystem::create_directories( this->path);
      }

      void endl() { line << std::endl; }

      void write()
      {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank != 0) return;

        std::string full_path = path.string() + "/" + this->filename;
        std::ofstream file;
        exanb::ldbg << "trying to open " << full_path << std::endl;
        if(first())
        {
          create_directories();
          file.open(full_path);
          file << header.rdbuf() << std::endl;
        }
        else
        {
          file.open(full_path, std::ofstream::in | std::ofstream::ate);
        }
        file << line.rdbuf();
        file.close();
      }      
    };
  }
}
