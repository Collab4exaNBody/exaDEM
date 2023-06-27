#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>
#include <onika/memory/allocator.h> // for ONIKA_ASSUME_ALIGNED macro
#include <exanb/compute/compute_pair_optional_args.h>
#include <vector>
#include <iomanip>

#include <exanb/compute/compute_cell_particles.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/compute_wall.h>

#include <mpi.h>



namespace exaDEM
{
	using namespace exanb;
	template<
		class GridT,
					class = AssertGridHasFields< GridT, field::_fx, field::_fy, field::_fz, field::_friction >
						>
						class ShearConfigurationOperator : public OperatorNode
						{

							void write_blocked_walls_info(std::ofstream& a_out)
							{
								a_out << "     blocked_wall_x_offset: " << *blocked_wall_x_offset << std::endl;
								a_out << "     blocked_wall_y_offset: " << *blocked_wall_y_offset << std::endl;
								a_out << "     blocked_wall_z_offset: " << *blocked_wall_z_offset << std::endl;
							}
							
							void write_compression_walls_info(std::ofstream& a_out)
							{
								a_out << "     compression_wall_x_offset: "       << *compression_wall_x_offset       << std::endl;
								a_out << "     compression_wall_x_sigma: "        << *compression_wall_x_sigma        << std::endl;
								a_out << "     compression_wall_x_velocity: "     << *compression_wall_x_velocity     << std::endl;
								a_out << "     compression_wall_x_acceleration: " << *compression_wall_x_acceleration << std::endl;

								a_out << "     compression_wall_y_offset: "       << *compression_wall_y_offset       << std::endl;
								a_out << "     compression_wall_y_sigma: "        << *compression_wall_y_sigma        << std::endl;
								a_out << "     compression_wall_y_velocity: "     << *compression_wall_y_velocity     << std::endl;
								a_out << "     compression_wall_y_acceleration: " << *compression_wall_y_acceleration << std::endl;

								a_out << "     movable_wall_z_offset: "       << *movable_wall_z_offset << std::endl;
								a_out << "     movable_wall_z_velocity: "     << *movable_wall_z_velocity << std::endl;
								a_out << "     movable_wall_z_start: "        << *movable_wall_z_start                 << std::endl;
							}

							
							void write_hooke_law_parameters(std::ofstream& a_out)
							{
								a_out << "     kt: " << *kt << std::endl;
								a_out << "     kn: " << *kn << std::endl;
								a_out << "     kr: " << *kr << std::endl;
								a_out << "     mu: " << *mu << std::endl;
								a_out << "     damprate: " << *damprate << std::endl;
							}

							void write_header(std::ofstream& a_out)
							{
								a_out << "Driver:" << std::endl;
                a_out << "  - shear_configuration:" << std::endl;
								a_out << "     output_rate: " << *output_rate << std::endl;
							}

							void write_operator()
							{
								const std::string file_name = "driver_shear_configuration.msp";
								std::ofstream file(file_name, std::ofstream::out);
								write_header(file);
								write_blocked_walls_info(file);
								write_compression_walls_info(file);
								write_hooke_law_parameters(file);
								file.close();
							}

							static constexpr Vec3d normx = {1,0,0};
							static constexpr Vec3d normy = {0,1,0};
							static constexpr Vec3d normz = {0,0,1};
							static constexpr Vec3d neg_normx = {-1,0,0};
							static constexpr Vec3d neg_normy = {0,-1,0};
							static constexpr Vec3d neg_normz = {0,0,-1};

							using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_vx ,field::_vy ,field::_vz, field::_vrot, field::_radius , field::_fx ,field::_fy ,field::_fz, field::_mass, field::_mom, field::_friction >;
							static constexpr ComputeFields compute_field_set {};

							ADD_SLOT( MPI_Comm           , mpi                 , INPUT , MPI_COMM_WORLD);
							ADD_SLOT( GridT  , grid            , INPUT_OUTPUT );
							ADD_SLOT( Domain , domain          , INPUT , REQUIRED );
							ADD_SLOT( double , dt              , INPUT , REQUIRED , DocString{"Time simulation increment"});
							ADD_SLOT( long   , timestep        , INPUT , REQUIRED , DocString{"Timestep of the simulation"});
							ADD_SLOT( int  , output_rate       , INPUT , int(-1) , DocString{"Set the output rate of driver_shear_configuration.msp"});

							ADD_SLOT( double , blocked_wall_x_offset  , INPUT , REQUIRED, DocString{"Offset of the unmovable wall toward (Ox) axis"} );
							ADD_SLOT( double , blocked_wall_y_offset  , INPUT , REQUIRED, DocString{"Offset of the unmovable wall toward (Oy) axis"} );
							ADD_SLOT( double , blocked_wall_z_offset  , INPUT , REQUIRED, DocString{"Offset of the unmovable wall toward (Oz) axis"} );

							ADD_SLOT( double , compression_wall_x_offset       , INPUT_OUTPUT , REQUIRED    , DocString{"Offset value of the compression wall toward (xO) axis"});
							ADD_SLOT( double , compression_wall_x_sigma        , INPUT        , REQUIRED    , DocString{"Sigma value of the compression wall toward (xO) axis"});
							ADD_SLOT( double , compression_wall_x_velocity     , INPUT_OUTPUT , double(0.0) , DocString{"Velocity value of the compression wall toward (xO) axis"});
							ADD_SLOT( double , compression_wall_x_acceleration , INPUT_OUTPUT , double(0.0) , DocString{"Acceleration value of the compression wall toward (xO) axis"});

							ADD_SLOT( double , compression_wall_y_offset       , INPUT_OUTPUT , REQUIRED    , DocString{"Offset value of the compression wall toward (yO) axis"});
							ADD_SLOT( double , compression_wall_y_sigma        , INPUT        , REQUIRED    , DocString{"Sigma value of the compression wall toward (yO) axis"});
							ADD_SLOT( double , compression_wall_y_velocity     , INPUT_OUTPUT , double(0.0) , DocString{"Velocity value of the compression wall toward (yO) axis"});
							ADD_SLOT( double , compression_wall_y_acceleration , INPUT_OUTPUT , double(0.0) , DocString{"Acceleration value of the compression wall toward (yO) axis"});

							ADD_SLOT( double , movable_wall_z_offset           , INPUT , REQUIRED, DocString{"Offset from the origin (0,0,0) of the movable wall toward (z0) axis"});
							ADD_SLOT( long   , movable_wall_z_start            , INPUT , long(0), DocString{"Start timestep trigering the displacement of the wall toward (zO) axis"});
							ADD_SLOT( double , movable_wall_z_velocity         , INPUT , double(0), DocString{"Velocity of the movable wall toward the (zO) axis"});

							ADD_SLOT( double , kt , INPUT , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double , kn , INPUT , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double , kr , INPUT , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double , mu , INPUT , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double , damprate  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});

							public:
	
						inline std::string documentation() const override final
							{
								return R"EOF(
        					This operator drives the three unmovable rigid walls, one rigid wall driven by its offset and two compression walls driven by their sigma. The forcefield used to model interation between walls and particles is Hooke.
        				)EOF";
							}

							inline void execute () override final
							{

								// mpi stuff
								MPI_Comm comm = *mpi;

								// compute new offset of the movable wall
								double real_offset = 0.0;
								if( (*movable_wall_z_start) < (*timestep))
								{
									real_offset = (*movable_wall_z_offset) + ((*timestep) - (*movable_wall_z_start)) * (*dt) * (*movable_wall_z_velocity);
								}
								else
								{
									real_offset = *movable_wall_z_offset;
								}

								// geometry
								const double dx = std::abs( (*blocked_wall_x_offset) + (*compression_wall_x_offset));
								const double dy = std::abs( (*blocked_wall_y_offset) + (*compression_wall_y_offset));
								const double dz = std::abs( (*blocked_wall_z_offset) + real_offset);
								const double surface_x = dy * dz;
								const double surface_y = dx * dz;

								// dem
								double sum_forces_x = 0.0;
								double sum_forces_y = 0.0;
								double tot_mass = 0.0;

								const auto _dt = *dt;
								const auto _kt = *kt;
								const auto _kn = *kn;
								const auto _kr = *kr;
								const auto _mu = *mu;
								const auto _dp = *damprate;

								// define walls
								RigidSurfaceFunctor r_wall_x {normx, *blocked_wall_x_offset, _dt, _kt, _kn, _kr, _mu, _dp};
								RigidSurfaceFunctor r_wall_y {normy, *blocked_wall_y_offset, _dt, _kt, _kn, _kr, _mu, _dp};
								RigidSurfaceFunctor r_wall_z {normz, *blocked_wall_z_offset, _dt, _kt, _kn, _kr, _mu, _dp};
								CompressionWallFunctor c_wall_x {neg_normx, (*compression_wall_x_offset), *compression_wall_x_velocity, _dt, _kt, _kn, _kr, _mu, _dp}; // Warning offset is defined with negative normal vector
								CompressionWallFunctor c_wall_y {neg_normy, (*compression_wall_y_offset), *compression_wall_y_velocity,  _dt, _kt, _kn, _kr, _mu, _dp}; 
								MovableWallFunctor m_wall_z {neg_normz, real_offset, *movable_wall_z_velocity, *dt, *kt, *kn, *kr, *mu, *damprate};

								// first step (time scheme)
								const double _dt2_2 = 0.5 * _dt * _dt;
								const double _dt2 = 0.5 * _dt;
								auto compute_step1 = [_dt, _dt2, _dt2_2] (double& a_offset, double& a_velocity, const double a_acceleration, const bool is_sigma_not_null) -> void
								{
									a_offset += _dt * a_velocity + _dt2_2 * a_acceleration; // due to negative normal vector
									if(is_sigma_not_null) a_velocity += _dt2 * a_acceleration; 
								};

								compute_step1 (*compression_wall_x_offset, *compression_wall_x_velocity, *compression_wall_x_acceleration, *compression_wall_x_sigma != 0.0 );
								compute_step1 (*compression_wall_y_offset, *compression_wall_y_velocity, *compression_wall_y_acceleration, *compression_wall_y_sigma != 0.0 );

								auto cells = grid->cells();
								IJK dims = grid->dimension();
								size_t ghost_layers = grid->ghost_layers();
								IJK dims_no_ghost = dims - (2*ghost_layers);

#     pragma omp parallel
								{
									GRID_OMP_FOR_BEGIN(dims_no_ghost,_,loc_no_ghosts, reduction(+: tot_mass, sum_forces_x, sum_forces_y))
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
										auto* __restrict__ _fric = cell_ptr[field::friction];
										const size_t n = cells[cell_i].size();

										// call BallWallFunctor for each particle
										//#         pragma omp simd //reduction(+:sum_f, sum_m)
										for(size_t j=0;j<n;j++)
										{
#define __ARGS_  _rx[j], _ry[j], _rz[j], _vx[j], _vy[j], _vz[j], _vrot[j], _r[j], _fx[j], _fy[j], _fz[j], _m[j], _mom[j], _fric[j]
											r_wall_x(__ARGS_);
											r_wall_y(__ARGS_);
											r_wall_z(__ARGS_);
											c_wall_x(__ARGS_, sum_forces_x);
											c_wall_y(__ARGS_, sum_forces_y);
											m_wall_z(__ARGS_);
#undef __ARGS_AND_SUM
											// compute total mass
											tot_mass += _m[j];
										}
									}
									GRID_OMP_FOR_END
								}
								// mpi sum
								{
									double tmp [3] = {sum_forces_x, sum_forces_y, tot_mass};
									MPI_Allreduce(MPI_IN_PLACE, tmp, 3, MPI_DOUBLE, MPI_SUM, comm);
									sum_forces_x = tmp[0];
									sum_forces_y = tmp[1];
									tot_mass = tmp[2];
								}

								// second step (time scheme)
								const double C = 0.5;
								auto compute_step2 = [_dp, tot_mass, C, _dt2] (double& a_velocity, double& a_acceleration, const double a_sum_forces, const double a_sigma, const double a_surface,  const bool a_is_tot_mass_not_null) -> void
								{
									if(a_is_tot_mass_not_null) a_acceleration = ( a_sum_forces - (a_sigma * a_surface) - ( _dp * a_velocity))/ (tot_mass * C);
									a_velocity += _dt2 * a_acceleration;
								};

								const bool is_tot_mass_not_null = tot_mass != 0;
								compute_step2 (*compression_wall_x_velocity, *compression_wall_x_acceleration, sum_forces_x, *compression_wall_x_sigma, surface_x, is_tot_mass_not_null);
								compute_step2 (*compression_wall_y_velocity, *compression_wall_y_acceleration, sum_forces_y, *compression_wall_y_sigma, surface_y, is_tot_mass_not_null);

								if( (*timestep) % (*output_rate) == 0)
								{
									int rank;
									MPI_Comm_rank(comm, &rank);
									if(rank == 0)
									{
										write_operator();
									}
								}
							}
						};


	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using ShearConfigurationOperatorTemplate = ShearConfigurationOperator<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "shear_configuration", make_grid_variant_operator< ShearConfigurationOperatorTemplate > );
	}
}

