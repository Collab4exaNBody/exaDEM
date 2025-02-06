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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exaDEM/drivers.h>
#include <mpi.h>

namespace exaDEM
{
  using namespace exanb;

  struct force_to_accel
  {
    const double dt;
    const double mass;
    void operator()(Ball& arg)
    {
      arg.weigth = mass;
      arg.f_ra(dt);
      arg.force_to_accel();
    }
    void operator()(Surface& arg)
    {
      arg.weigth = mass;
      arg.force_to_accel();
    }
    void operator()(Stl_mesh& arg)
    {
      arg.force_to_accel();
    }
    void operator()(auto&& arg) { arg.force_to_accel(); }
  };

  struct tmp_reduce
  {
    std::tuple<bool, Vec3d> operator()(Ball& arg)
    {
      if( arg.is_compressive() || arg.is_force_motion() )
      {
        return {true, arg.forces};
      }
      else 
      { 
        return {false, {0,0,0}};
      }
    }
    std::tuple<bool, Vec3d> operator()(Surface& arg)
    {
      if( arg.is_compressive() || arg.is_force_motion() )
      {
        return {true, arg.forces};
      }
      else 
      { 
        return {false, {0,0,0}};
      }
    }
    std::tuple<bool, Vec3d> operator()(auto && arg) { return {false, {0,0,0}};}
  };


  class ForceToAccelDriverFunctor : public OperatorNode
  {
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD, DocString{"MPI communicator for parallel processing."});
    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(double, dt, INPUT, DocString{"dt is the time increment of the timeloop"});
    ADD_SLOT(double, system_mass, INPUT, REQUIRED); 

    public:
    inline std::string documentation() const override final
    {
      return R"EOF(
          This operator updates driver centers using their velocities. Not that accelerations are not used.
        )EOF";
    }

    inline void execute() override final
    {
      // we need to update forces if required
      std::vector <int> ids; // id, forces
      std::vector <double> pack; // id, forces
      tmp_reduce reduce;
      for (size_t id = 0; id < drivers->get_size(); id++)
      {
        auto &driver = drivers->data(id);
        auto [update , forces] = std::visit(reduce, driver);
        if( update )
        {
          ids.push_back(id);
          pack.push_back(forces.x);
          pack.push_back(forces.y);
          pack.push_back(forces.z);
        }
      }

      if(pack.size() > 0)
      {
        std::vector <double> unpack(pack.size());
        MPI_Allreduce(pack.data(), unpack.data(), pack.size(), MPI_DOUBLE, MPI_SUM, *mpi);
        //for(size_t i = 0 ; i < pack.size() ; i++)
        for(size_t i = 0 ; i < ids.size() ; i++)
        {
          Vec3d forces = Vec3d{unpack[i*3], unpack[i*3+1], unpack[i*3+2]};
          int id = ids[i];
          std::visit([&forces](auto&& args) {args.forces = forces;}, drivers->data(id));
        }
      }
      force_to_accel func = {*dt, *system_mass};
      for (size_t id = 0; id < drivers->get_size(); id++)
      {
        auto &driver = drivers->data(id);
        std::visit(func, driver);
      }
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("force_to_accel_driver", make_simple_operator<ForceToAccelDriverFunctor>); }
} // namespace exaDEM
