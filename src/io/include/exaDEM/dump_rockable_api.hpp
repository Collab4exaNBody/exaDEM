#pragma once

# include <exanb/core/particle_type_id.h>
# include <exaDEM/shapes.hpp>
# include <exaDEM/interaction/interaction.hpp>
# include <exaDEM/shape_reader.hpp>
# include <fstream>



namespace rockable
{
  using namespace exanb;
  using namespace exaDEM;

  // reader and writer
  struct Particle
  {
    int type; // Use for exaDEM
    int group;    ///< A number that relates to a 'category of bodies'
    int cluster = 0;  ///< A number that identifies the cluster to which the particle belongs

    Vec3d pos;  ///< Position
    Vec3d vel;  ///< Velocity
    Vec3d acc;  ///< Acceleration

    Quaternion Q;      ///< Angular position
    Vec3d vrot;  ///< Angular velocity
    Vec3d arot;  ///< Angular acceleration

    double homothety;  ///< Homothety applied to the shape
  };

  // only used by the reader
  struct Interaction
  {
    int type; // Use for exaDEM
    int i; // i is also id i
    int j; // j is also id j
    int subi;
    int subj;

    Vec3d n;  ///< normal
    double dn;
    Vec3d pos;  ///< Position
    Vec3d vel;  ///< Velocity
    Vec3d fn;  
    Vec3d ft;  
    Vec3d mom;  
    double damp;
  };

  template<typename STREAM>
    void stream(STREAM& output, const Particle& p, exaDEM::shapes& shps)
    {
      output << shps[p.type]->m_name << " " << p.group << " " << p.cluster << " " << p.homothety << " " 
        << p.pos.x << " " << p.pos.y << " " << p.pos.z << " "
        << p.vel.x << " " << p.vel.y << " " << p.vel.z << " " 
        << p.acc.x << " " << p.acc.y << " " << p.acc.z << " " 
        << p.Q.w << " " << p.Q.x << " " << p.Q.y << " " << p.Q.z << " "
        << p.vrot.x << " " << p.vrot.y << " " << p.vrot.z << " "
        << p.arot.x << " " << p.arot.y << " " << p.arot.z;
    }

  template<typename STREAM>
    Particle decrypt_particle(STREAM&& input, ParticleTypeMap& ptm)
    {
      std::string type_name;
      Particle p;
      input >> type_name;
      // debug
      //lout << type_name << std::endl;
      p.type = ptm[type_name];

      input >> p.group >>  p.cluster >>  p.homothety
        >> p.pos.x >>  p.pos.y >>  p.pos.z
        >> p.vel.x >>  p.vel.y >>  p.vel.z
        >> p.acc.x >>  p.acc.y >>  p.acc.z 
        >> p.Q.w >>  p.Q.x >>  p.Q.y >>  p.Q.z
        >> p.vrot.x >>  p.vrot.y >>  p.vrot.z
        >> p.arot.x >>  p.arot.y >>  p.arot.z;
      return p;
    }

  template<typename STREAM>
    Interaction decrypt_interaction(STREAM&& input)
    {
      Interaction I;
      input >> I.i >> I.j >> I.type >> I.subi >> I.subj;
      input >> I.n.x >> I.n.y >> I.n.z >> I.dn; 
      input >> I.pos.x >> I.pos.y >> I.pos.z; 
      input >> I.vel.x >> I.vel.y >> I.vel.z; 
      input >> I.fn.x >> I.fn.y >> I.fn.z; 
      input >> I.ft.x >> I.ft.y >> I.ft.z; 
      input >> I.mom.x >> I.mom.y >> I.mom.z; 
      input >> I.damp;
      return I;
    }

  // skip pos, vel, fn, damp
  // missing, cell_i, cell_j, p_i, p_j
  inline exaDEM::Interaction convert(const rockable::Interaction& input)
  {
    exaDEM::Interaction res;
    res.id_i     = input.i;
    res.id_j     = input.j;
    res.sub_i    = input.subi;
    res.sub_j    = input.subj;
    res.type     = input.type;
    res.friction = input.ft;
    res.moment   = input.mom;
    return res;
  }

  struct ConfReader
  {
    std::string shapeFile = "undefined";
    int precision = 13;
    int Particles = 0;
    int Interactions = 0;
    int nDriven = 0;
    double dt = -1;
    double t = -1;  // physical time
    std::vector<double> densities;
    std::vector<rockable::Particle> particles;
    std::vector<rockable::Particle> drivers;
    std::vector<rockable::Interaction> interactions;
    std::vector<rockable::Interaction> driver_interactions;

    // for exaDEM
    shapes shps;
    ParticleTypeMap ptm;

    bool check()
    {
      if( int(particles.size()) != (Particles - nDriven) ) return false;
      if( int(interactions.size()) > Interactions ) return false; // remove interactions with drivers
      for( auto& it : densities )
      {
        if( it <= 0.0 ) {
          return false; 
        }
      } 
      if( shapeFile == "undefined" ) return false;
      if( nDriven < 0 ) return false;
    }

		void read_particles(std::ifstream &input)
		{
			std::string line;
		  std::getline(input, line); // skip current line
      drivers.resize(nDriven);
			particles.resize(Particles);
			for(size_t d = 0 ; d < drivers.size() ; d++)
			{
				std::getline(input, line);
				if( line[0] != '#' )
				{
					drivers[d] = decrypt_particle(std::stringstream(line), ptm);
				}
				else d--;
			}
			for(size_t p = 0 ; p < particles.size() ; p++)
			{
				std::getline(input, line); 
				particles[p] = decrypt_particle(std::stringstream(line), ptm); 
			}
		}

		void read_interactions(std::ifstream &input)
		{
			std::string line;
			int n_particle_particle = 0;
			int n_driver_particle = 0;
			interactions.resize(Interactions);
			driver_interactions.resize(0);
			for(int it = 0 ; it < Interactions ; it++)
			{
				std::getline(input, line);
				if( line[0] != '#' )
				{
					rockable::Interaction I = decrypt_interaction(std::stringstream(line));

					bool is_driver = false;
					if( I.i < nDriven ) {
						is_driver = true;
					}
					else {
						I.i -= nDriven;
					}
					if( I.j < nDriven ) {
						is_driver = true;
					}
					else {
						I.j -= nDriven;
					}

					if( is_driver ) {
						driver_interactions.push_back(I);
						n_driver_particle++;
					} else {
						interactions[n_particle_particle++] = I;
					}
				}
				else it--;
			}
			interactions.resize(n_particle_particle);
		}

		void read_stream(std::ifstream &input)
		{
			std::string key, line;
			while (std::getline(input, line))
			{
				input >> key;
				if ( key == "t" ) {
					input >> t; 
				}
				else if ( key == "dt" ) {
					input >> dt;
					if(dt > 0.0) lout << "[WARNING, read_conf_rockable] dt is overloaded by the value defined within the rockable conf file" << std::endl;
					else {
						lout << "[ERROR, read_conf_rockable] dt is <= 0.0, please define a corret value of dt into your rockable conf file" << std::endl;  
						std::exit(EXIT_FAILURE);
					}
				}
				else if ( key == "density" ) {
					int group;
					input >> group;
					if(group + 1 >= int(densities.size())) densities.resize(group + 1);
					input >> densities[group];
				}
				else if ( key == "nDriven" ) {
					input >> nDriven; 
					if(Particles > 0) {
						lout << "[ERROR, read_conf_rockable] nDriven is defined after Particles" << std::endl;
						std::exit(EXIT_FAILURE);
					}
				}
				else if ( key == "precision" ) {
					input >> precision; 
				}
				else if ( key == "Particles" ) {
					input >> Particles;
					if( shapeFile == "undefined" ) {
						lout << "[ERROR, read_conf_rockable] shapeFile is not defined before Particles" << std::endl;
						std::exit(EXIT_FAILURE);
					}
          Particles -= nDriven;
					read_particles(input);
				}
				else if ( key == "shapeFile" ) {
					input >> shapeFile;
					exaDEM::read_shp(ptm, shps, shapeFile, false);
				}
				else if (key ==  "Interactions") {
					input >> Interactions;
					read_interactions(input);
				}
				else {
					lout << "[WARNING, read_conf_rockable] the key: " << key << " is skipped." << std::endl;
				}
			}
		}
	};
};
