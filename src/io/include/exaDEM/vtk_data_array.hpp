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

#include <zlib.h>

#include <cstdint>
#include <exaDEM/color_log.hpp>
#include <fstream>
#include <onika/math/basic_types.h>
#include <string>
#include <utility>
#include <vector>

namespace exaDEM {

/**
 * @brief Output mode of VTK XML data arrays.
 *
 * In binary mode, every DataArray is zlib-compressed then base64-encoded (inline "binary" format).
 * The VTKFile element must then declare header_type="UInt64" and compressor="vtkZLibDataCompressor",
 * see vtk_file_attributes().
 */
struct vtk_output_mode {
  bool binary = false;
  int compression_level = Z_DEFAULT_COMPRESSION;
};

/**
 * @brief Converts a compression keyword into a zlib compression level.
 * @param compression "default", "min", "max" or a single digit in [0-9].
 */
inline int vtk_compression_level(const std::string& compression) {
  if (compression == "default") return Z_DEFAULT_COMPRESSION;
  if (compression == "min") return Z_NO_COMPRESSION;
  if (compression == "max") return Z_BEST_COMPRESSION;
  if (compression.size() == 1 && compression[0] >= '0' && compression[0] <= '9') return compression[0] - '0';
  color_log::error("vtk_compression_level",
                   "Unknown compression type: " + compression + ". Keywords: max, min, default or [0-9]");
  return Z_DEFAULT_COMPRESSION;
}

/**
 * @brief Attributes of the VTKFile element, depending on the output mode.
 */
inline std::string vtk_file_attributes(const vtk_output_mode& mode) {
  std::string attributes = "version=\"1.0\" byte_order=\"LittleEndian\"";
  if (mode.binary) attributes += " header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\"";
  return attributes;
}

/**
 * @brief Base64 encoding (with padding) of a raw byte buffer.
 */
inline void vtk_encode_base64(const unsigned char* data, size_t size, std::string& out) {
  static constexpr char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  out.reserve(out.size() + 4 * ((size + 2) / 3));
  size_t i = 0;
  for (; i + 2 < size; i += 3) {
    const uint32_t triple = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | uint32_t(data[i + 2]);
    out += table[(triple >> 18) & 0x3F];
    out += table[(triple >> 12) & 0x3F];
    out += table[(triple >> 6) & 0x3F];
    out += table[triple & 0x3F];
  }
  if (i < size) {
    const bool two = (i + 1 < size);
    const uint32_t triple = (uint32_t(data[i]) << 16) | (two ? uint32_t(data[i + 1]) << 8 : 0);
    out += table[(triple >> 18) & 0x3F];
    out += table[(triple >> 12) & 0x3F];
    out += two ? table[(triple >> 6) & 0x3F] : '=';
    out += '=';
  }
}

/**
 * @brief Writes the values of an array as zlib-compressed and base64-encoded data (VTK inline binary format).
 *
 * The data is compressed as a single block. The UInt64 header [number of blocks, block size, last block size,
 * compressed block sizes...] and the compressed data are base64-encoded separately, as expected by VTK.
 */
template <typename T>
inline void write_vtk_compressed_values(std::ofstream& out, const std::vector<T>& values, int compression_level) {
  const size_t raw_size = values.size() * sizeof(T);
  std::string encoded;
  if (raw_size == 0) {
    const uint64_t header[3] = {0, 0, 0};
    vtk_encode_base64(reinterpret_cast<const unsigned char*>(header), sizeof(header), encoded);
  } else {
    uLongf compressed_size = compressBound(raw_size);
    std::vector<unsigned char> compressed(compressed_size);
    const int status = compress2(compressed.data(), &compressed_size, reinterpret_cast<const Bytef*>(values.data()),
                                 raw_size, compression_level);
    if (status != Z_OK) {
      color_log::error("write_vtk_compressed_values", "zlib compression failed");
    }
    const uint64_t header[4] = {1, raw_size, raw_size, compressed_size};
    vtk_encode_base64(reinterpret_cast<const unsigned char*>(header), sizeof(header), encoded);
    vtk_encode_base64(compressed.data(), compressed_size, encoded);
  }
  out << encoded;
}

/**
 * @brief Writes a full DataArray element, in ascii or in compressed binary format.
 * @param type VTK type name (Int32, Float64, ...), it has to match T.
 * @param name Name of the array, omitted if empty.
 * @param ncomp Number of components, omitted if <= 0.
 */
template <typename T>
inline void write_vtk_data_array(std::ofstream& out, const char* type, const std::string& name, int ncomp,
                                 const std::vector<T>& values, const vtk_output_mode& mode) {
  out << "        <DataArray type=\"" << type << "\"";
  if (!name.empty()) out << " Name=\"" << name << "\"";
  if (ncomp > 0) out << " NumberOfComponents=\"" << ncomp << "\"";
  out << " format=\"" << (mode.binary ? "binary" : "ascii") << "\">" << std::endl;
  if (mode.binary) {
    write_vtk_compressed_values(out, values, mode.compression_level);
  } else {
    for (const auto& v : values) {
      if constexpr (sizeof(T) == 1) {
        out << " " << static_cast<int>(v);
      } else {
        out << " " << v;
      }
    }
  }
  out << std::endl;
  out << "        </DataArray>" << std::endl;
}

// Maps a C++ type onto its VTK XML "type" attribute, its number of components
// and the scalar type used to store its components.
template <typename T>
struct vtk_type_info;

template <>
struct vtk_type_info<int32_t> {
  using comp_type = int32_t;
  static constexpr const char* name = "Int32";
  static constexpr int ncomp = 1;
};

template <>
struct vtk_type_info<int64_t> {
  using comp_type = int64_t;
  static constexpr const char* name = "Int64";
  static constexpr int ncomp = 1;
};

template <>
struct vtk_type_info<double> {
  using comp_type = double;
  static constexpr const char* name = "Float64";
  static constexpr int ncomp = 1;
};

template <>
struct vtk_type_info<onika::math::Vec3d> {
  using comp_type = double;
  static constexpr const char* name = "Float64";
  static constexpr int ncomp = 3;
};

template <>
struct vtk_type_info<onika::math::Mat3d> {
  using comp_type = double;
  static constexpr const char* name = "Float64";
  static constexpr int ncomp = 9;
};

template <typename T>
inline void append_value(std::vector<T>& out, T v) {
  out.push_back(v);
}
inline void append_value(std::vector<double>& out, const onika::math::Vec3d& v) {
  out.insert(out.end(), {v.x, v.y, v.z});
}
inline void append_value(std::vector<double>& out, const onika::math::Mat3d& m) {
  out.insert(out.end(), {m.m11, m.m12, m.m13, m.m21, m.m22, m.m23, m.m31, m.m32, m.m33});
}

// One projected cell data array: its output name, whether it is selected for output,
// and the per-cell values accumulated so far.
template <typename T>
struct CellDataField {
  std::string name;
  bool write = false;
  std::vector<typename vtk_type_info<T>::comp_type> buffer;

  explicit CellDataField(std::string field_name) : name(std::move(field_name)) {}

  inline void append(const T& value) {
    if (write) append_value(buffer, value);
  }
};

template <typename T>
inline void write_cell_data_array(std::ofstream& outFile, const CellDataField<T>& field, const vtk_output_mode& mode) {
  if (!field.write) return;
  write_vtk_data_array(outFile, vtk_type_info<T>::name, field.name, vtk_type_info<T>::ncomp, field.buffer, mode);
}

template <typename T>
inline void write_pcell_data_array(std::ofstream& outFile, const CellDataField<T>& field) {
  if (!field.write) return;
  outFile << "      <PDataArray type=\"" << vtk_type_info<T>::name << "\" Name=\"" << field.name
          << "\" NumberOfComponents=\"" << vtk_type_info<T>::ncomp << "\"/>" << std::endl;
}
}  // namespace exaDEM
