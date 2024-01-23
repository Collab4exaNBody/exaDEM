//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

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
#include <exaDEM/cylinder_wall.h>
#include <exaDEM/shapes.hpp>

namespace exaDEM
{
	using namespace exanb;
	struct PolyhedraRigidWallFunctor
	{
		const shapes& shps;
		Vec3d m_normal;
		double m_offset;
		double m_vel;
		double m_dt;
		double m_kt;
		double m_kn;
		double m_kr;
		double m_mu;
		double m_dampRate;


		ONIKA_HOST_DEVICE_FUNC inline void operator () (
				const double a_rx, const double a_ry, const double a_rz,
				const double a_vx, const double a_vy, const double a_vz,
				const uint8_t type,
				const Vec3d& a_vrot, 
				double a_particle_radius,
				double& a_fx, double& a_fy, double& a_fz, 
				const double a_mass,
				Vec3d& a_mom,
				const Quaternion& a_orientation,
				Vec3d& a_ft) const
		{
			const shape* shp = shps[type];
			Vec3d pos = {a_rx, a_ry, a_rz};
			Vec3d vel = {a_vx, a_vy, a_vz};


			Vec3d f = {0.0,0.0,0.0};
			int nv = shp->get_number_of_vertices();
			double r = shp->m_radius;

			// === pre test
			Vec3d pos_proj = dot(pos, m_normal) * m_normal;
			Vec3d contact_position = m_offset * m_normal;

			Vec3d vec_n = pos_proj - contact_position;
			double n = norm(vec_n);
			double dn = n - a_particle_radius;
			bool no_contact = true;

//			if(dn < 0)
			{
				for(int i = 0 ; i < nv ; i++)
				{

					// === get vertex
					const Vec3d v = shp->get_vertex(i, pos, a_orientation);

					// === project vertex on wall normal 
					pos_proj = dot(v, m_normal) * m_normal;
					
					// === compute distance between projections
					Vec3d dist = pos_proj - m_normal*m_offset;
					//Vec3d dist = m_normal*m_offset - pos_proj;
					double d = exanb::norm(dist);

					// === get directions
					Vec3d n = dist / d ;

					// === compute the real contact point
					contact_position = v - n * (d);

					// === interpenetration depends of if n == normal or n = -normal
					// === note: r is the radius of the vertex
					dn =  d - r;
					//if(dot(n,m_normal) == 1) {dn = d + r ;}

					//std::cout << r << " " << std::endl;
					if(dn <= 0)
					{
						no_contact = false;
						assert(dot(contact_position, m_normal) == m_offset);
/*
						std::cout << v.z << " " << dn << " " << d << " " << r << std::endl;
						std::cout << v.x << " " << v.y << " " << v.z << std::endl;
						std::cout << contact_position.x << " " << contact_position.y << " " << contact_position.z << std::endl;
						std::cout << n.x << " " << n.y << " " << n.z << std::endl;
*/
						// === wall position is equal to the position of the contact point
						Vec3d rigid_surface_center = contact_position;

						// === if velocity
						const Vec3d rigid_surface_velocity = {0,0,0}; //m_normal * m_vel;

						// === no angular velocity
						constexpr Vec3d rigid_surface_angular_velocity = {0.0,0.0,0.0};

						// === the wall mass is >>>> polyhedron mass
						constexpr double meff = 1;

						// === compute hooke force
						exaDEM::hooke_force_core(
								dn, n,  
								m_dt, m_kn, m_kt, m_kr, m_mu, m_dampRate, 1, 
								a_ft, contact_position, pos, vel, f, a_mom, a_vrot,
								rigid_surface_center, rigid_surface_velocity, rigid_surface_angular_velocity
								);
					}
				}
			}
			// === update forces
			a_fx += f.x;
			a_fy += f.y;
			a_fz += f.z;
			if(no_contact) reset(a_ft);
		}
	};


	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_fx,field::_fy,field::_fz >
		>
		class PolyhedraRigidWall : public OperatorNode
		{
			static constexpr Vec3d default_axis = { 1.0, 0.0, 1.0 };
			static constexpr Vec3d null= { 0.0, 0.0, 0.0 };
			// attributes processed during computation
			using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_vx ,field::_vy ,field::_vz, field::_type, field::_vrot, field::_radius , field::_fx ,field::_fy ,field::_fz, field::_mass, field::_mom, field::_orient, field::_friction >;
			static constexpr ComputeFields compute_field_set {};

			ADD_SLOT( GridT   , grid  			              , INPUT_OUTPUT );
			ADD_SLOT( shapes  , shapes_collection, INPUT_OUTPUT , DocString{"Collection of shapes"});
			ADD_SLOT( Vec3d  , normal  , INPUT , Vec3d{1.0,0.0,0.0} , DocString{"Normal vector of the rigid surface"});
			ADD_SLOT( double , offset  , INPUT , 0.0, DocString{"Offset from the origin (0,0,0) of the rigid surface"} );
			ADD_SLOT( double  , dt                        , INPUT 	, REQUIRED 	, DocString{"Timestep of the simulation"});
			ADD_SLOT( double  , kt                        , INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
			ADD_SLOT( double  , kn                        , INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"} );
			ADD_SLOT( double  , kr                        , INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
			ADD_SLOT( double  , mu                        , INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
			ADD_SLOT( double  , damprate                  , INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
        This operator computes forces for interactions beween particles and a cylinder.
        )EOF";
			}

			inline void execute () override final
			{
				const double vel_null = {0.0};
				PolyhedraRigidWallFunctor func {*shapes_collection, *normal, *offset, vel_null, *dt, *kt, *kn, *kr, *mu, *damprate};

				compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
			}
		};

	template<class GridT> using PolyhedraRigidWallTmpl = PolyhedraRigidWall<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "polyhedra_rigid_wall", make_grid_variant_operator< PolyhedraRigidWallTmpl > );
	}

}

