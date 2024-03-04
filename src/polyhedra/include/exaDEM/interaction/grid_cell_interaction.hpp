#pragma once 

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/dynamic_data_storage.hpp>
#include <exaDEM/interaction/migration_buffer.hpp>
#include <exaDEM/interaction/migration_helper.hpp>

/************************************************************************************
 * interaction storage for the whole grid.
 ***********************************************************************************/
namespace exaDEM
{
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
