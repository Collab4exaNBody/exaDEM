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
}
