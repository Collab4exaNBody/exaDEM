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

#include <onika/cuda/stl_adaptors.h>

namespace exaDEM {

/** @brief Cell indexes for vertices, edges, and faces relative to a cell of the projection grid. */
struct RShapeDriverCellIndexes {
  size_t offset_;    /**< Index of the grid cell. */
  size_t nvertices_; /**< Number of vertices in the grid cell. */
  size_t nedges_;    /**< Number of edges in the grid cell. */
  size_t nfaces_;    /**< Number of faces in the grid cell. */
};

/** @brief List of geometric elements (vertices, edges, faces) projected on the grid for a R-Shape driver. */
struct RShapeDriverGridCellIndexes {
  onika::memory::CudaMMVector<RShapeDriverCellIndexes>
      cells_; /**< List of grid cells with their respective counts of vertices, edges, and faces. */
  onika::memory::CudaMMVector<int> data_; /**< List of vertex, edge, and face indices for all grid cells. */

  /** @brief Reset the grid.
   */
  void reset() {
    cells_.clear();
    data_.clear();
  }

  /** @brief Initialize the grid with a specified number of cells and total element counts.
   * @param number_of_cells The total number of grid cells.
   * @param total_nvertices The total number of vertices across all grid cells.
   * @param total_nedges The total number of edges across all grid cells.
   * @param total_nfaces The total number of faces across all grid cells.
   */
  void initialize(size_t number_of_cells, size_t total_nvertices, size_t total_nedges, size_t total_nfaces) {
    assert(number_of_cells < 1e8);  // 1e8 is an arbitrary value to avoid resizing the grid with a too large size.
                                    // This can be a sign of a bug.
    cells_.clear();
    cells_.resize(number_of_cells);
    data_.clear();
    data_.resize(total_nvertices + total_nedges + total_nfaces);
  }

  /** @brief Fill a grid cell with the given element indices.
   * @param cell_idx The index of the grid cell to fill.
   * @param indices_vertices Span of vertex indices for the grid cell.
   * @param indices_edges Span of edge indices for the grid cell.
   * @param indices_faces Span of face indices for the grid cell.
   */
  inline void fill_cell(size_t cell_idx, std::span<const int> indices_vertices, std::span<const int> indices_edges,
                        std::span<const int> indices_faces) {
    assert(cell_idx < cells_.size());
    // offset is set by the caller, it is the responsibility of the caller to ensure that the offset is correct and that
    // the data is not overwritten. cells[cell_idx].offset = offset;

    RShapeDriverCellIndexes& cell = cells_[cell_idx];

    cell.nvertices_ = indices_vertices.size();
    cell.nedges_ = indices_edges.size();
    cell.nfaces_ = indices_faces.size();
    std::memcpy(data_.data() + cell.offset_, indices_vertices.data(), cell.nvertices_ * sizeof(int));
    std::memcpy(data_.data() + cell.offset_ + cell.nvertices_, indices_edges.data(), cell.nedges_ * sizeof(int));
    std::memcpy(data_.data() + cell.offset_ + cell.nvertices_ + cell.nedges_, indices_faces.data(),
                cell.nfaces_ * sizeof(int));

#ifndef NDEBUG
    if (cell_idx < cells_.size() - 1) {
      size_t next_offset = cells_[cell_idx + 1].offset_;
      size_t current_offset = cell.offset_;
      size_t current_size = cell.nvertices_ + cell.nedges_ + cell.nfaces_;
      assert(current_offset + current_size == next_offset);
    } else {
      // for the last cell, we can only check that the offset + size does not exceed the total size of the data.
      size_t current_offset = cell.offset_;
      size_t current_size = cell.nvertices_ + cell.nedges_ + cell.nfaces_;
      assert(current_offset + current_size == data_.size());
    }
#endif
  }

  /** @brief Print debug information about the grid.
   */
  inline void debug() const {
    exanb::lout << "RShapeDriverGrid:" << std::endl;
    exanb::lout << "Number of cells: " << cells_.size() << std::endl;
    for (size_t i = 0; i < cells_.size(); i++) {
      const RShapeDriverCellIndexes& cell = cells_[i];
      exanb::lout << "Cell " << i << ": offset=" << cell.offset_ << ", nvertices=" << cell.nvertices_
                  << ", nedges=" << cell.nedges_ << ", nfaces=" << cell.nfaces_ << std::endl;
    }
  }
};

/** @brief A structure to access the grid data for a specific cell.
 */
struct RShapeDriverCellAccessor {
 public:
  const int* grid_id_vertices_; /**< List of vertex indices. */
  const int* grid_id_edges_;    /**< List of edge indices. */
  const int* grid_id_faces_;    /**< List of face indices. */
  size_t rshape_nv_;            /**< Number of vertices in the grid cell. */
  size_t rshape_ne_;            /**< Number of edges in the grid cell. */
  size_t rshape_nf_;            /**< Number of faces in the grid cell. */

  /** @brief Build an accessor to the vertex, edge, and face indices of a given grid cell.
   * @param cell_idx Index of the grid cell to access.
   * @param grid_rshape Grid storing the geometric elements projected for the R-Shape driver.
   */
  ONIKA_HOST_DEVICE_FUNC RShapeDriverCellAccessor(size_t cell_idx, const RShapeDriverGridCellIndexes& grid_rshape) {
    using onika::cuda::vector_data;

    const RShapeDriverCellIndexes& cell_info = vector_data(grid_rshape.cells_)[cell_idx];
    const int* data_ptr = vector_data(grid_rshape.data_);
    grid_id_vertices_ = data_ptr + cell_info.offset_;
    grid_id_edges_ = grid_id_vertices_ + cell_info.nvertices_;
    grid_id_faces_ = grid_id_edges_ + cell_info.nedges_;
    rshape_nv_ = cell_info.nvertices_;
    rshape_ne_ = cell_info.nedges_;
    rshape_nf_ = cell_info.nfaces_;
  }

 private:
  /** @brief Default constructor disabled; an accessor must always be bound to a grid cell. */
  RShapeDriverCellAccessor() {}
};
}  // namespace exaDEM