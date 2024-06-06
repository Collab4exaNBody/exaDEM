#pragma once

#include <exanb/core/basic_types.h>
#include <onika/cuda/cuda.h> // mots cles specifiques
#include <onika/memory/allocator.h> // cudaMMVector
#include <onika/cuda/stl_adaptors.h>
#include <exanb/core/basic_types.h>

namespace exaDEM
{

	using namespace exanb;
	
	struct Cells_GPU
	{
		int num_cells = 0;
		std::vector<int> cells_id;
		std::vector<int> cells_size;
		std::vector<int> cells_start;
		std::vector<int> cells_end;
		
		int num_particles = 0;
		
		onika::memory::CudaMMVector<double> field_rx;
		onika::memory::CudaMMVector<double> field_ry;
		onika::memory::CudaMMVector<double> field_rz;
		
		onika::memory::CudaMMVector<double> field_radius;
		
		onika::memory::CudaMMVector<double> field_vx;
		onika::memory::CudaMMVector<double> field_vy;
		onika::memory::CudaMMVector<double> field_vz;
		
		onika::memory::CudaMMVector<double> field_mass;
		
		onika::memory::CudaMMVector<double> field_fx;
		onika::memory::CudaMMVector<double> field_fy;
		onika::memory::CudaMMVector<double> field_fz;
		
		onika::memory::CudaMMVector<double> field_vrotx;
		onika::memory::CudaMMVector<double> field_vroty;
		onika::memory::CudaMMVector<double> field_vrotz;
		
		onika::memory::CudaMMVector<double> field_momx;
		onika::memory::CudaMMVector<double> field_momy;
		onika::memory::CudaMMVector<double> field_momz;
		
		void reset()
		{
			num_cells = 0;
			cells_id.clear();
			cells_size.clear();
			cells_start.clear();
			cells_end.clear();
			
			num_particles = 0;
			
			field_rx.clear();
			field_ry.clear();
			field_rz.clear();
			
			field_radius.clear();
			
			field_vx.clear();
			field_vy.clear();
			field_vz.clear();
			
			field_mass.clear();
			
			field_fx.clear();
			field_fy.clear();
			field_fz.clear();
			
			field_vrotx.clear();
			field_vroty.clear();
			field_vrotz.clear();
			
			field_momx.clear();
			field_momy.clear();
			field_momz.clear();
		}
		
		void add_cell(int cellule, int size)
		{
			cells_id.push_back(cellule);
			cells_size.push_back(size);
			if(num_cells == 0)
			{
				cells_start.push_back(0);
			}
			else
			{
				cells_start.push_back(cells_end[num_cells - 1]);
			}
			cells_end.push_back(cells_start[num_cells] + size);
			num_cells++;
			num_particles += size;
		}
		
		void prep_GPU()
		{
			field_rx.resize(num_particles);
			field_ry.resize(num_particles);
			field_rz.resize(num_particles);
			
			field_radius.resize(num_particles);
			
			field_vx.resize(num_particles);
			field_vy.resize(num_particles);
			field_vz.resize(num_particles);
			
			field_mass.resize(num_particles);
			
			field_fx.resize(num_particles);
			field_fy.resize(num_particles);
			field_fz.resize(num_particles);
			
			field_vrotx.resize(num_particles);
			field_vroty.resize(num_particles);
			field_vrotz.resize(num_particles);
			
			field_momx.resize(num_particles);
			field_momy.resize(num_particles);
			field_momz.resize(num_particles);
		}
		
		
	};
}
