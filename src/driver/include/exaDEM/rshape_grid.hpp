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
  size_t offset;    /**< Index of the grid cell. */
  size_t nvertices; /**< Number of vertices in the grid cell. */
  size_t nedges;    /**< Number of edges in the grid cell. */
  size_t nfaces;    /**< Number of faces in the grid cell. */
};

/** @brief List of geometric elements (vertices, edges, faces) projected on the grid for a R-Shape driver. */
struct RShapeDriverGridCellIndexes {
  onika::memory::CudaMMVector<RShapeDriverCellIndexes>
      cells; /**< List of grid cells with their respective counts of vertices, edges, and faces. */
  onika::memory::CudaMMVector<int> data; /**< List of vertex, edge, and face indices for all grid cells. */

  /** @brief Reset the grid.
   */
  void reset() {
    cells.clear();
    data.clear();
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
    cells.clear();
    cells.resize(number_of_cells);
    data.clear();
    data.resize(total_nvertices + total_nedges + total_nfaces);
  }

  /** @brief Fill a grid cell with the given element indices.
   * @param cell_idx The index of the grid cell to fill.
   * @param indices_vertices Span of vertex indices for the grid cell.
   * @param indices_edges Span of edge indices for the grid cell.
   * @param indices_faces Span of face indices for the grid cell.
   */
  inline void fill_cell(size_t cell_idx, std::span<const int> indices_vertices, std::span<const int> indices_edges,
                        std::span<const int> indices_faces) {
    assert(cell_idx < cells.size());
    // offset is set by the caller, it is the responsibility of the caller to ensure that the offset is correct and that
    // the data is not overwritten. cells[cell_idx].offset = offset;

    RShapeDriverCellIndexes& cell = cells[cell_idx];

    cell.nvertices = indices_vertices.size();
    cell.nedges = indices_edges.size();
    cell.nfaces = indices_faces.size();
    std::memcpy(data.data() + cell.offset, indices_vertices.data(), cell.nvertices * sizeof(int));
    std::memcpy(data.data() + cell.offset + cell.nvertices, indices_edges.data(), cell.nedges * sizeof(int));
    std::memcpy(data.data() + cell.offset + cell.nvertices + cell.nedges, indices_faces.data(),
                cell.nfaces * sizeof(int));

#ifndef NDEBUG
    if (cell_idx < cells.size() - 1) {
      size_t next_offset = cells[cell_idx + 1].offset;
      size_t current_offset = cell.offset;
      size_t current_size = cell.nvertices + cell.nedges + cell.nfaces;
      assert(current_offset + current_size == next_offset);
    } else {
      // for the last cell, we can only check that the offset + size does not exceed the total size of the data.
      size_t current_offset = cell.offset;
      size_t current_size = cell.nvertices + cell.nedges + cell.nfaces;
      assert(current_offset + current_size == data.size());
    }
#endif
  }

  /** @brief Print debug information about the grid.
   */
  inline void debug() const {
    exanb::lout << "RShapeDriverGrid:" << std::endl;
    exanb::lout << "Number of cells: " << cells.size() << std::endl;
    for (size_t i = 0; i < cells.size(); i++) {
      const RShapeDriverCellIndexes& cell = cells[i];
      exanb::lout << "Cell " << i << ": offset=" << cell.offset << ", nvertices=" << cell.nvertices
                  << ", nedges=" << cell.nedges << ", nfaces=" << cell.nfaces << std::endl;
    }
  }
};

/** @brief A structure to access the grid data for a specific cell.
 */
struct RShapeDriverCellAccessor {
 public:
  const int* grid_id_vertices; /**< List of vertex indices. */
  const int* grid_id_edges;    /**< List of edge indices. */
  const int* grid_id_faces;    /**< List of face indices. */
  size_t rshape_nv;            /**< Number of vertices in the grid cell. */
  size_t rshape_ne;            /**< Number of edges in the grid cell. */
  size_t rshape_nf;            /**< Number of faces in the grid cell. */

  /** @brief Build an accessor to the vertex, edge, and face indices of a given grid cell.
   * @param cell_idx Index of the grid cell to access.
   * @param grid_rshape Grid storing the geometric elements projected for the R-Shape driver.
   */
  ONIKA_HOST_DEVICE_FUNC RShapeDriverCellAccessor(size_t cell_idx, const RShapeDriverGridCellIndexes& grid_rshape) {
    using onika::cuda::vector_data;

    const RShapeDriverCellIndexes& cell_info = vector_data(grid_rshape.cells)[cell_idx];
    const int* data_ptr = vector_data(grid_rshape.data);
    grid_id_vertices = data_ptr + cell_info.offset;
    grid_id_edges = grid_id_vertices + cell_info.nvertices;
    grid_id_faces = grid_id_edges + cell_info.nedges;
    rshape_nv = cell_info.nvertices;
    rshape_ne = cell_info.nedges;
    rshape_nf = cell_info.nfaces;
  }

 private:
  /** @brief Default constructor disabled; an accessor must always be bound to a grid cell. */
  RShapeDriverCellAccessor() {}
};
}  // namespace exaDEM