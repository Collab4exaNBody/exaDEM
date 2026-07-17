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

#include <mpi.h>
#include <onika/string_utils.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace exaDEM {
namespace analysis {

struct AnalysisFileManager {
  std::filesystem::path path_;
  std::string filename_;
  std::stringstream line_;
  std::stringstream header_;

  void set_path(std::string p) { path_ = p; }
  void set_filename(std::string f) { filename_ = f; }

  bool first() {
    std::string full_name = path_.string() + "/" + this->filename_;
    return !std::filesystem::exists(full_name);
  }

  template <typename T>
  void add_element(std::string name, T& new_element, std::string format) {
    header_ << name << " ";
    std::string element = onika::format_string(format, new_element);
    line_ << element << " ";
  }

  void create_directories() {
    exanb::ldbg << "create directory " << this->path_ << std::endl;
    std::filesystem::create_directories(this->path_);
  }

  void endl() { line_ << std::endl; }

  void write() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return;
    }

    std::string full_path = path_.string() + "/" + this->filename_;
    std::ofstream file;
    exanb::ldbg << "trying to open " << full_path << std::endl;
    if (first()) {
      create_directories();
      file.open(full_path);
      file << header_.rdbuf() << std::endl;
    } else {
      file.open(full_path, std::ofstream::in | std::ofstream::ate);
    }
    file << line_.rdbuf();
    file.close();
  }
};
}  // namespace analysis
}  // namespace exaDEM
