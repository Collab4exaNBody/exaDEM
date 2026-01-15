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

#include <exanb/extra_storage/dynamic_data_storage.hpp>
#include <exanb/extra_storage/migration_buffer.hpp>
#include <exanb/extra_storage/migration_helper.hpp>
#include <exaDEM/interaction/interaction.hpp>

#include <vector>

/************************************************************************************
 * interaction storage for the whole grid.
 ***********************************************************************************/
namespace exaDEM
{
  using namespace exanb;
  /**
   * @brief Struct representing interactions in DEM simulation.
   * It contains a memory-managed vector storing extra dynamic data storage for interactions.
   */
  typedef GridExtraDynamicDataStorageT<Interaction> GridCellParticleInteraction;

  /**
   * @brief Alias for the migration helper for interactions.
   */
  typedef ExtraDynamicDataStorageMigrationHelper<Interaction> InteractionMigrationHelper;

  /**
   * @brief Alias for the cell move buffer for interactions.
   */
  typedef ExtraDynamicDataStorageCellMoveBufferT<Interaction> InteractionCellMoveBuffer;

  /**
   * @brief Alias for the grid move buffer for interactions.
   */
  typedef ExtraDynamicStorageDataGridMoveBufferT<Interaction> InteractionGridMoveBuffer;
} // namespace exaDEM
