#pragma once 

#include <exaDEM/interaction.hpp>
#include <exaDEM/interaction/dynamic_data_storage.hpp>
#include <exaDEM/interaction/migration_buffer.hpp>
#include <exaDEM/interaction/migration_helper.hpp>

/************************************************************************************
 * interaction storage for the whole grid.
 ***********************************************************************************/
namespace exaDEM
{
	struct GridCellParticleInteraction
	{
		onika::memory::CudaMMVector< CellExtraDynamicDataStorageT<Interaction> > m_data;
		GridCellParticleInteraction() {};
	};

	typedef ExtraDynamicDataStorageMigrationHelper<Interaction> InteractionMigrationHelper;
	typedef ExtraDynamicDataStorageCellMoveBufferT<Interaction> InteractionCellMoveBuffer;
	typedef ExtraDynamicStorageDataGridMoveBufferT<Interaction> InteractionGridMoveBuffer;
}
