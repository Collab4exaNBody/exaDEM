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
//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <mpi.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/compute/compute_cell_particles.h>
#include <memory>
#include <exaDEM/common_compute_kernels.h>
#include <exaDEM/compute_hooke_force.h>
//#include <exaDEM/ball_wall.h>

namespace exaDEM
{
  using namespace exanb;
  struct BallWallFunctor
  {
    Vec3d m_ball_center ;
    double m_ball_radius;
    double m_sigma;
    double m_dt;
    double m_kt;
    double m_kn;
    double m_kr;
    double m_mu;
    double m_dampRate;

    ONIKA_HOST_DEVICE_FUNC inline void operator() (
        const double a_rx, const double a_ry, const double a_rz,
        const double a_vx, const double a_vy, const double a_vz,
        const Vec3d& a_vrot, 
        const double a_particle_radius,
        double& a_fx, double& a_fy, double& a_fz, 
        const double a_mass,
        Vec3d& a_mom,
        Vec3d& a_ft, 
        double& ball_f) const
    {
      const Vec3d m_ball_angular_velocity = {0.0 , 0.0 , 0.0};
      const Vec3d m_ball_velocity = {0.0,0.0,0.0}; // TODO : do not touch

      Vec3d pos   = Vec3d{a_rx, a_ry, a_rz};
      Vec3d vel   = Vec3d{a_vx, a_vy, a_vz};
      // === direction
      const auto dir  = pos - m_ball_center;

      // === interpenetration
      const double dn = m_ball_radius - ( norm(dir) + a_particle_radius );
      if(dn > 0.0)
      {
        a_ft = {0.0,0.0,0.0};
        return;
      }

      // === figure out the contact position 
      const auto dir_norm   = dir / norm(dir) ;
      const auto contact  = m_ball_center + dir_norm * m_ball_radius ; // compute contact position between the particle and the ball

      // === compute damp
      const double meff = 1;//mass; // mass ball >>>>> mass i
      const double damp = exaDEM::compute_damp(m_dampRate, m_kn, meff);

      // === relative velocity  
      const auto total_vel = exaDEM::compute_relative_velocity(contact,
          m_ball_center, m_ball_velocity , m_ball_angular_velocity,
          pos     , vel               , a_vrot);

      //const double vn = exanb::dot(total_vel, dir_norm);
      const double vn = exanb::dot(contact-pos, total_vel);

      // === normal force
      const Vec3d fn  = exaDEM::compute_normal_force(m_kn , damp, dn, vn, dir_norm);

      // === compute tangential force
      auto ft     = exaDEM::compute_tangential_force(m_kt, m_dt, vn, dir_norm, total_vel);
      //a_ft      += exaDEM::compute_tangential_force(m_kt, m_dt, vn, dir_norm, total_vel);
      auto threshold_ft   = exaDEM::compute_threshold_ft(m_mu, m_kn, dn);

      exaDEM::fit_tangential_force(threshold_ft, ft);
      //exaDEM::fit_tangential_force(threshold_ft, a_ft);

      const auto f = (-1) *(fn + ft);
      //const auto f = (-1) *(fn + a_ft); 

      // === updates forces
      a_fx += f.x ;
      a_fy += f.y ;
      a_fz += f.z ;

      // === updates ball forcess
      ball_f -= exanb::dot(fn,dir_norm);

      // === updates moments
      const Vec3d tmp   =   m_kr * (m_ball_angular_velocity - vel) * m_dt;
      const auto Ci     =   contact - pos;
      const auto moment_i   =   exanb::cross(Ci, f) + tmp;
      a_mom       +=  moment_i;
    }
  };

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_fx,field::_fy,field::_fz >
    >
    class DriveBallWallRadius : public OperatorNode
    {
      static constexpr Vec3d null= { 0.0, 0.0, 0.0 };
      // attributes processed during computation
      using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_vx ,field::_vy ,field::_vz, field::_vrot, field::_radius , field::_fx ,field::_fy ,field::_fz, field::_mass, field::_mom>;
      static constexpr ComputeFields compute_field_set {};

      ADD_SLOT( MPI_Comm           , mpi                 , INPUT , MPI_COMM_WORLD);
      ADD_SLOT( GridT   , grid        , INPUT_OUTPUT );
      ADD_SLOT( Vec3d   , center       , INPUT   , REQUIRED   , DocString{"Center of the ball"});
      ADD_SLOT( double  , radius       , INPUT   , REQUIRED   , DocString{"Radius of the ball, positive and should be superior to the biggest sphere radius in the ball"});
      ADD_SLOT( double  , sigma        , INPUT   , REQUIRED   , DocString{"sigma"});
      ADD_SLOT( double  , dt                    , INPUT   , REQUIRED   , DocString{"Timestep of the simulation"});
      ADD_SLOT( double  , kt        , INPUT   , REQUIRED   , DocString{"Parameter of the force law used to model contact ball wall/sphere"});
      ADD_SLOT( double  , kn        , INPUT   , REQUIRED   , DocString{"Parameter of the force law used to model contact ball wall/sphere"} );
      ADD_SLOT( double  , kr        , INPUT   , REQUIRED   , DocString{"Parameter of the force law used to model contact ball wall/sphere"});
      ADD_SLOT( double  , mu        , INPUT   , REQUIRED   , DocString{"Parameter of the force law used to model contact ball wall/sphere"});
      ADD_SLOT( double  , damprate        , INPUT   , REQUIRED   , DocString{"Parameter of the force law used to model contact ball wall/sphere"});
      ADD_SLOT( double  , r_vel        , INPUT_OUTPUT   , double(0.0)   , DocString{""});
      ADD_SLOT( double  , r_acc        , INPUT_OUTPUT   , double(0.0)   , DocString{""});

      public:

      inline std::string documentation() const override final
      {
        return R"EOF(
        This operator drives the ball wall radius, the ball works like a boundary condition. The force field used to model the interaction between the ball and particles is Hooke.
        )EOF";
      }

      inline void execute () override final
      {
        // mpi stuff
        MPI_Comm comm = *mpi;
        // first step (time scheme)
        const double sig = *sigma;
        *radius += (*dt) * (*r_vel) + 0.5 * (*dt) * (*dt) * (*r_acc);
        if(sig != 0.0) (*r_vel) += 0.5 * (*dt) * (*r_acc);

        BallWallFunctor func { *center , *radius, *sigma, *dt, *kt, *kn, *kr, *mu, *damprate};
        const double r = (*radius);
        double ball_f = 0.0; // forces applied on the ball
        double ballM = 0.0; // summ of masses of particles inside the ball 
        const double pi   = 4*std::atan(1);
        const double ballS = 4 * pi * r * r; // surface

        auto cells = grid->cells();
        IJK dims = grid->dimension();
        size_t ghost_layers = grid->ghost_layers();
        IJK dims_no_ghost = dims - (2*ghost_layers);

#     pragma omp parallel
        {
          GRID_OMP_FOR_BEGIN(dims_no_ghost,_,loc_no_ghosts, reduction(+: ballM, ball_f))
          {
            IJK loc = loc_no_ghosts + ghost_layers;
            size_t cell_i = grid_ijk_to_index(dims,loc);
            auto& cell_ptr = cells[cell_i];

            // define fields
            auto* __restrict__ _rx = cell_ptr[field::rx];
            auto* __restrict__ _ry = cell_ptr[field::ry];
            auto* __restrict__ _rz = cell_ptr[field::rz];
            auto* __restrict__ _vx = cell_ptr[field::vx];
            auto* __restrict__ _vy = cell_ptr[field::vy];
            auto* __restrict__ _vz = cell_ptr[field::vz];
            auto* __restrict__ _vrot = cell_ptr[field::vrot];
            auto* __restrict__ _r = cell_ptr[field::radius];
            auto* __restrict__ _fx = cell_ptr[field::fx];
            auto* __restrict__ _fy = cell_ptr[field::fy];
            auto* __restrict__ _fz = cell_ptr[field::fz];
            auto* __restrict__ _m = cell_ptr[field::mass];
            auto* __restrict__ _mom = cell_ptr[field::mom];
            Vec3d _fric = {0.0,0.0,0.0}; // not used
            const size_t n = cells[cell_i].size();

            // call BallWallFunctor for each particle
#         pragma omp simd //reduction(+:ball_f, ballM)
            for(size_t j=0;j<n;j++)
            {
              func(
                  _rx[j], _ry[j], _rz[j],
                  _vx[j], _vy[j], _vz[j],
                  _vrot[j], _r[j],
                  _fx[j], _fy[j], _fz[j],
                  _m[j], _mom[j], _fric,
                  ball_f
                  );
              ballM += _m[j];
            }
          }
          GRID_OMP_FOR_END
        }
        // reduce ball_f
        {
          double tmp[2] = {ball_f,ballM};
          MPI_Allreduce(MPI_IN_PLACE, tmp, 2, MPI_DOUBLE, MPI_SUM, comm);
          ball_f = tmp[0];
          ballM = tmp[1];
        }
        // second step (time scheme)
        const double C = 0.5;
        if(ballM != 0.0)
        {
          *r_acc = ( -1 * ball_f - (sig * ballS) - ( (*damprate) * (*r_vel)))/ (ballM * C);
        }
        *r_vel += 0.5 * (*dt) * (*r_acc); 
      }
    };

  template<class GridT> using DriveBallWallRadiusTmpl = DriveBallWallRadius<GridT>;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "drive_ball_wall_radius", make_grid_variant_operator< DriveBallWallRadiusTmpl > );
  }

}
