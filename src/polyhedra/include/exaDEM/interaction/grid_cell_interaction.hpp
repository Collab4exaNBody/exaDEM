#pragma once 

#include <exaDEM/interaction.hpp>
#include <exaDEM/interaction/cell_interaction.hpp>
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
	typedef ExtraDynamicStorageGridMoveBufferT<Interaction> InteractionGridMoveBuffer;
}
