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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <onika/log.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid_fields.h>
#include <onika/math/basic_types.h>
#include <exanb/compute/reduce_cell_particles.h>

#include <mpi.h>
#include <cstring>

#include <exaDEM/simulation_state.h>
#include <exaDEM/dem_simulation_state.h>
#include <exaDEM/traversal.h>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/itools/itools.hpp>

// ================== Thermodynamic state compute operator ======================
namespace exaDEM
{
  using namespace exanb;
  using namespace exaDEM::itools;

  template <class GridT, class = AssertGridHasFields<GridT, field::_vx, field::_vy, field::_vz, field::_vrot, field::_mass>> struct SimulationStateNode : public OperatorNode
  {
    // compile time constant indicating if grid has type field

    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(GridT, grid, INPUT, REQUIRED);
    ADD_SLOT(Domain, domain, INPUT, REQUIRED);
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});
    ADD_SLOT(SimulationState, simulation_state, OUTPUT);
    ADD_SLOT(double, system_mass, OUTPUT);

    // DEM data
    ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT, DocString{"Interaction lists classified according to their types"});
    ADD_SLOT(Classifier2, ic2, INPUT);
    ADD_SLOT(bool, symetric, INPUT, REQUIRED, DocString{"Use of symetric feature (contact law)"});

    static constexpr FieldSet<field::_vx, field::_vy, field::_vz, field::_vrot, field::_mass> reduce_field_set{};

    template <typename T> inline IOSimInteractionResult reduce_sim_io(Classifier<T> &classifier, bool symetric)
    {
      IOSimInteractionResult res;
      VectorT<IOSimInteractionResult> results;
      int types = classifier.number_of_waves();
      results.resize(types);
      {
        // std::vector<ParallelExecutionWrapper> pexw;
        // pexw.resize(types);
        for (int i = 0; i < types; i++)
        {
          const auto &buffs = classifier.buffers[i];
          auto [data, size] = classifier.get_info(i);
          const double *const dnp = onika::cuda::vector_data(buffs.dn);

          int coef = 1;
          if (i < 4 && symetric)
            coef *= 2;

          InteractionWrapper<T> dataWrapper(data);
          IOSimInteractionFunctor func = {dnp, coef};

          if (size > 0 && dnp != nullptr) // skip it if forces has not been computed
          {
            reduce_data<T, IOSimInteractionFunctor, IOSimInteractionResult>(parallel_execution_context(), dataWrapper, func, size, results[i]);
          }
        }
      } // synchronize
      for (int i = 0; i < types; i++)
      {
        res.update(results[i]);
      }
      return res;
    }
    
    inline IOSimInteractionResult reduce_sim_io2(Classifier2 &classifier, bool symetric)
    {
      IOSimInteractionResult res;
      VectorT<IOSimInteractionResult> results;
      int types = classifier.number_of_waves();
      results.resize(types);
      {
        // std::vector<ParallelExecutionWrapper> pexw;
        // pexw.resize(types);
        for (int i = 0; i < types; i++)
        {
          const auto &buffs = classifier.buffers[i];
          auto [data, size] = classifier.get_info(i);
          const double *const dnp = onika::cuda::vector_data(buffs.dn);

          int coef = 1;
          if (i < 4 && symetric)
            coef *= 2;

          //InteractionWrapper<T> dataWrapper(data);
          IOSimInteractionFunctor2 func = {dnp, coef};
          
          InteractionSOA2 data2;
          
          data2.size2 = data.size2;
          data2.m_type = data.m_type;
          
	data2.ft_x = (double*) malloc(data.size2 * sizeof(double) );
	data2.ft_y = (double*) malloc(data.size2 * sizeof(double) );
	data2.ft_z = (double*) malloc(data.size2 * sizeof(double) );
	
	data2.mom_x = (double*) malloc(data.size2 * sizeof(double) );
	data2.mom_y = (double*) malloc(data.size2 * sizeof(double) );
	data2.mom_z = (double*) malloc(data.size2 * sizeof(double) );
	
	data2.id_i = (uint64_t*) malloc(data.size2 * sizeof(uint64_t) );
	data2.id_j = (uint64_t*) malloc(data.size2 * sizeof(uint64_t) );
	
	data2.cell_i = (uint32_t*) malloc(data.size2 * sizeof(uint32_t) );
	data2.cell_j = (uint32_t*) malloc(data.size2 * sizeof(uint32_t) );
	
	data2.p_i = (uint16_t*) malloc(data.size2 * sizeof(uint16_t) );
	data2.p_j = (uint16_t*) malloc(data.size2 * sizeof(uint16_t) );
	
	data2.sub_i = (uint16_t*) malloc(data.size2 * sizeof(uint16_t) );
	data2.sub_j = (uint16_t*) malloc(data.size2 * sizeof(uint16_t) );
	
	//printf("MALLOC\n");          
          
          cudaMemcpy(data2.ft_x, data.ft_x, data.size2 * sizeof(double), cudaMemcpyDeviceToHost );
          
          //printf("MEMCPY\n");
          
          cudaMemcpy(data2.ft_y, data.ft_y, data.size2 * sizeof(double), cudaMemcpyDeviceToHost );
          cudaMemcpy(data2.ft_z, data.ft_z, data.size2 * sizeof(double), cudaMemcpyDeviceToHost );
          
          cudaMemcpy(data2.mom_x, data.mom_x, data.size2 * sizeof(double), cudaMemcpyDeviceToHost );
          cudaMemcpy(data2.mom_y, data.mom_y, data.size2 * sizeof(double), cudaMemcpyDeviceToHost );
          cudaMemcpy(data2.mom_z, data.mom_z, data.size2 * sizeof(double), cudaMemcpyDeviceToHost );
          
          cudaMemcpy(data2.id_i, data.id_i, data.size2 * sizeof(uint64_t), cudaMemcpyDeviceToHost );
          cudaMemcpy(data2.id_j, data.id_j, data.size2 * sizeof(uint64_t), cudaMemcpyDeviceToHost );
          
          cudaMemcpy(data2.p_i, data.p_i, data.size2 * sizeof(uint16_t), cudaMemcpyDeviceToHost );
          cudaMemcpy(data2.p_j, data.p_j, data.size2 * sizeof(uint16_t), cudaMemcpyDeviceToHost );
          
          cudaMemcpy(data2.sub_i, data.sub_i, data.size2 * sizeof(uint16_t), cudaMemcpyDeviceToHost );
          cudaMemcpy(data2.sub_j, data.sub_j, data.size2 * sizeof(uint16_t), cudaMemcpyDeviceToHost );
          
          cudaMemcpy(data2.cell_i, data.cell_i, data.size2 * sizeof(uint32_t), cudaMemcpyDeviceToHost );
          cudaMemcpy(data2.cell_j, data.cell_j, data.size2 * sizeof(uint32_t), cudaMemcpyDeviceToHost );
          
          //printf("REDUCE\n");
          
          /*for(int i = 0; i < data.size2; i++)
          {
          	printf("INTERACTION[%d] : I(CELL:%d P:%d ID:%d) J(CELL:%d P:%d ID:%d)\n", i, data.cell_i[i], data.p_i[i], data.id_i[i], data.cell_j[i], data.p_j[i], data.id_j[i]);
          }*/

          if (size > 0 && dnp != nullptr) // skip it if forces has not been computed
          {
            reduce_data2<IOSimInteractionFunctor2, IOSimInteractionResult>(parallel_execution_context(), data2, func, size, results[i]);
          }
          
          free(data2.ft_x);
          free(data2.ft_y);
          free(data2.ft_z);
          free(data2.mom_x);
          free(data2.mom_y);
          free(data2.mom_z);
          free(data2.cell_i);
          free(data2.cell_j);
          free(data2.p_i);
          free(data2.p_j);
          free(data2.id_i);
          free(data2.id_j);
          free(data2.sub_i);
          free(data2.sub_j);
          
        }
      } // synchronize
      for (int i = 0; i < types; i++)
      {
        res.update(results[i]);
      }
      return res;
    }

    inline void execute() override final
    {
      //printf("SIM\n");
      MPI_Comm comm = *mpi;
      SimulationState &sim_info = *simulation_state;

      Vec3d kinetic_energy;  // constructs itself with 0s
      Vec3d rotation_energy; // constructs itself with 0s
      double mass = 0.;
      uint64_t total_particles = 0;

      auto [cell_ptr, cell_size] = traversal_real->info();
      exaDEM::simulation_state_variables sim{}; // kinetic_energy, rotation_energy, mass, potential_energy, total_particles};
      ReduceSimulationStateFunctor func = {};
      reduce_cell_particles(*grid, false, func, sim, reduce_field_set, parallel_execution_context(), {}, cell_ptr, cell_size);

      // get interaction informations
      //Classifier<InteractionSOA> &classifier = *ic;
      Classifier2 &classifier = *ic2;
      //exaDEM::itools::IOSimInteractionResult red = reduce_sim_io(classifier, *symetric);
      exaDEM::itools::IOSimInteractionResult red = reduce_sim_io2(classifier, *symetric);

      // reduce partial sums and share the result
      uint64_t active_interactions, total_interactions;
      double dn;
      {
        double tmpDouble[7] = {sim.rotation_energy.x, sim.rotation_energy.y, sim.rotation_energy.z, sim.kinetic_energy.x, sim.kinetic_energy.y, sim.kinetic_energy.z, sim.mass};
        uint64_t tmpUInt64T[3] = {sim.n_particles, red.n_act_interaction, red.n_tot_interaction};
        MPI_Allreduce(MPI_IN_PLACE, tmpDouble, 7, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, &red.min_dn, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(MPI_IN_PLACE, tmpUInt64T, 3, MPI_UINT64_T, MPI_SUM, comm);

        rotation_energy.x = tmpDouble[0];
        rotation_energy.y = tmpDouble[1];
        rotation_energy.z = tmpDouble[2];
        kinetic_energy.x = tmpDouble[3];
        kinetic_energy.y = tmpDouble[4];
        kinetic_energy.z = tmpDouble[5];
        mass = tmpDouble[6];
        dn = red.min_dn;

        total_particles = tmpUInt64T[0];
        active_interactions = tmpUInt64T[1];
        total_interactions = tmpUInt64T[2];
      }

      // Volume
      double volume = 1.0;
      if (!domain->xform_is_identity())
      {
        Mat3d mat = domain->xform();
        Vec3d a{mat.m11, mat.m21, mat.m31};
        Vec3d b{mat.m12, mat.m22, mat.m32};
        Vec3d c{mat.m13, mat.m23, mat.m33};
        volume = dot(cross(a, b), c);
      }
      volume *= bounds_volume(domain->bounds());

      // write results to output
      sim_info.set_kinetic_energy(kinetic_energy);
      sim_info.set_rotation_energy(rotation_energy);
      sim_info.set_mass(mass);
      sim_info.set_volume(volume);
      sim_info.set_particle_count(total_particles);
      sim_info.set_active_interaction_count(active_interactions);
      sim_info.set_interaction_count(total_interactions);
      sim_info.set_dn(dn);

      // for other operators
      *system_mass = mass;
      //printf("SIM END\n");
    }
  };

  template <class GridT> using SimulationStateNodeTmpl = SimulationStateNode<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(simulation_state) { OperatorNodeFactory::instance()->register_factory("simulation_state", make_grid_variant_operator<SimulationStateNodeTmpl>); }

} // namespace exaDEM
