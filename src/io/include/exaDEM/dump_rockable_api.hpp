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

  /**
   * @brief Data structure representing a single particle for I/O purposes.
   */
  struct Particle
  {
    int type;  ///< Internal type identifier (used by exaDEM)
    int group; ///< Group identifier corresponding to a category of bodies
    int cluster = 0; ///< Cluster identifier to which the particle belongs

    Vec3d pos;  ///< Position
    Vec3d vel;  ///< Linear velocity
    Vec3d acc;  ///< Linear acceleration

    Quaternion Q; ///< Orientation as a quaternion
    Vec3d vrot;   ///< Angular velocity (rotation speed)
    Vec3d arot;   ///< Angular acceleration

    double homothety; ///< Homothety (uniform scaling) factor applied to the shape
  };

  /**
   * @brief Data structure representing an interaction between two particles.
   */
  struct Interaction
  {
    int type;    ///< Contact type identifier (used by exaDEM)
    int i;       ///< ID of the first interacting particle (also referred to as id i)
    int j;       ///< ID of the second interacting particle (also referred to as id j)
    int subi;    ///< Sub-identifier or local index for particle i
    int subj;    ///< Sub-identifier or local index for particle j
    Vec3d n;     ///< Contact normal vector
    double dn;   ///< Penetration depth or normal overlap
    Vec3d pos;   ///< Contact position
    Vec3d vel;   ///< Relative velocity at the contact point
    Vec3d fn;    ///< Normal contact force
    Vec3d ft;    ///< Tangential contact force
    Vec3d mom;   ///< Contact moment (torque)
    double damp; ///< Damping coefficient applied to the interaction
  };

  /**
   * @brief Streams the data of a particle to an output stream.
   * 
   * @tparam STREAM The output stream type (e.g., std::ostream).
   * @param output The output stream to write the data to.
   * @param p The particle to be serialized.
   * @param shps The shape registry used to resolve the shape name from particle type.
   */
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

  /**
   * @brief Deserialize a Particle object from an input stream.
   *
   * @tparam STREAM Type of the input stream (e.g., std::istream).
   * @param input Input stream to read from.
   * @param ptm Map from particle type names to particle type IDs.
   * @return Particle object initialized from the stream data.
   */
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

  /**
   * @brief Deserialize an Interaction object from an input stream.
   *
   * @tparam STREAM Input stream type (e.g., std::istream).
   * @param input The input stream to read from.
   * @return Fully initialized Interaction object.
   */
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

  /**
   * @brief Deserializes an Interaction object from an input stream.
   *
   * @tparam STREAM The input stream type (e.g., std::istream).
   * @param input The input stream to read from.
   * @return The fully initialized Interaction object.
   */
  inline exaDEM::Interaction convert(const rockable::Interaction& input)
  {
    // skip pos, vel, fn, damp
    // missing, cell_i, cell_j, p_i, p_j
    exaDEM::Interaction res;
    auto& pi      = res.i();
    auto& pj      = res.j();
    pi.id         = input.i;
    pj.id         = input.j;
    pi.sub        = input.subi;
    pj.sub        = input.subj;
    res.pair.type = input.type;
    res.friction  = input.ft;
    res.moment    = input.mom;
    return res;
  }

  /**
   * @brief Configuration reader structure for particle and interaction simulation from rockable files.
   */
  struct ConfReader
  {
    std::string shapeFile = "undefined";      ///< Filename of the shape file
    int precision = 13;                       ///< Numeric precision for output or calculations (Particles)
    int n_particles = 0;                      ///< Number of particles in the simulation
    int n_interactions = 0;                   ///< Number of interactions (contacts)
    int nDriven = 0;                          ///< Number of driven particles (or drivers)
    double dt = -1;                           ///< Time step size (negative means undefined)
    double t = -1;                            ///< Physical simulation time (negative means undefined)
    bool periodic[3] = {false, false, false}; ///< Periodic boundary conditions flags in x, y, z directions

    std::vector<double> densities;                      ///< Material densities for particles
    std::vector<rockable::Particle> particles;          ///< Vector of particles in the simulation
    std::vector<rockable::Particle> drivers;            ///< Vector of driver particles (externally controlled)
    std::vector<rockable::Interaction> interactions;    ///< Vector of particle interactions
    std::vector<rockable::Interaction> driver_interactions; ///< Vector of interactions involving drivers

    // For exaDEM integration
    shapes shps;               ///< Registry of shapes indexed by particle type
    ParticleTypeMap ptm;       ///< Map from particle type names to type IDs

    /**
     * @brief Validates the configuration data integrity.
     *
     * Checks consistency between the number of particles, interactions,
     * densities, and other key parameters.
     *
     * @return true if the configuration passes all checks; false otherwise.
     */
    bool check()
    {
      // Check if the number of particles excluding driven ones matches the vector size
      if( int(particles.size()) != (n_particles - nDriven) ) return false;
      // Check that interactions size is not larger than expected (drivers interactions may be excluded)
      if( int(interactions.size()) > n_interactions ) return false; // remove interactions with drivers
                                                                    // Check that all densities are positive
      for( auto& it : densities )
      {
        if( it <= 0.0 ) {
          return false; 
        }
      } 
      // Check if shape file is defined
      if( shapeFile == "undefined" ) return false;
      // Check that number of driven particles is non-negative
      if( nDriven < 0 ) return false;
      // If all checks passed
      return true;
    }

    /**
     * @brief Reads particle data from an input file stream.
     *
     * @param input Input file stream to read from.
     */
    void read_particles(std::ifstream &input)
    {
      std::string line;
      drivers.resize(nDriven);
      particles.resize(n_particles);

      // Read driver particles

      for(int d = 0 ; d < nDriven ; d++)
      {
        std::getline(input, line);

        if( line[0] != '#' )
        {
          drivers[d] = decrypt_particle(std::stringstream(line), ptm);
        }
        else d--;
      }
      // Read regular particles
      for(size_t p = 0 ; p < particles.size() ; p++)
      {
        std::getline(input, line); 
        if( line[0] != '#' )
        {
          particles[p] = decrypt_particle(std::stringstream(line), ptm); 
        }
        else p--;
      }
    }

    /**
     * @brief Reads interaction data from an input file stream.
     *
     * @param input Input file stream to read from.
     */
    void read_interactions(std::ifstream &input)
    {
      std::string line;
      int n_particle_particle = 0;
      int n_driver_particle = 0;
      interactions.resize(n_interactions);
      driver_interactions.clear();
      for(int it = 0 ; it < n_interactions ; it++)
      {
        std::getline(input, line);
        // Skip comment lines starting with '#'
        if (line.empty() || line[0] == '#')
        {
          it--;
          continue;
        }
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
      interactions.resize(n_particle_particle);
    }

    /**
     * @brief Reads and parses the rockable configuration from an input file stream.
     *
     * This function reads the configuration line-by-line, processes known keys,
     * sets the corresponding member variables, and calls relevant functions to
     * read particles and interactions data.
     *
     * @param file Input file stream to read from.
     */
    void read_stream(std::ifstream &file)
    {
      int max_warning_displayed = 40;
      std::string key, line;
      while (std::getline(file, line))
      {
        size_t first_char_pos = line.find_first_not_of(" \t");
        if (first_char_pos == std::string::npos) continue;   // empty line
        if (line[first_char_pos] == '#') continue;           // comment line

        ldbg << line << std::endl;
        std::stringstream input(line);
        input >> key;
        if ( key == "t" ) {
          input >> t; 
        }
        else if ( key == "periodicity" ) {
          int per;
          for(int dim = 0; dim < 3 ; dim++)
          {
            input >> per; 
            periodic[dim] = per != 0;
          }
        }
        else if ( key == "dt" ) {
          input >> dt;
          if(dt > 0.0) color_log::warning("read_conf_rockable", "'dt' is being overridden by the value defined in the rockable configuration file.");
          else color_log::error("read_conf_rockable", "'dt' is <= 0.0. Please define a correct value for 'dt' in your rockable configuration file.");
        }
        else if ( key == "density" ) {
          int group;
          input >> group;
          if(group + 1 >= int(densities.size())) densities.resize(group + 1);
          input >> densities[group];
        }
        else if ( key == "nDriven" ) {
          input >> nDriven; 
          if(n_particles > 0) {
            color_log::error("read_conf_rockable", "'nDriven' is defined after 'Particles'.");
          }
        }
        else if ( key == "precision" ) {
          input >> precision; 
        }
        else if ( key == "Particles" ) {
          input >> n_particles;
          if( shapeFile == "undefined" ) {
            color_log::error("read_conf_rockable", "'shapeFile' is not defined before 'Particles'.");
          }
          n_particles -= nDriven;
          std::getline(input, line);
          read_particles(file);
        }
        else if ( key == "shapeFile" ) {
          input >> shapeFile;
          auto s = exaDEM::read_shps(shapeFile, false);
          exaDEM::register_shapes(ptm, shps, s);
        }
        else if (key ==  "Interactions") {
          input >> n_interactions;
          read_interactions(file);
        }
        else {
          max_warning_displayed--;
          if(max_warning_displayed > 0) color_log::warning("read_conf_rockable", "The key '" + key + "' is skipped.");
        }
      }
      std::string msg = "Rockable reader  = ";
      msg            += std::to_string(nDriven) + " drivers - "; 
      msg            += std::to_string(n_particles) + " particles - ";
      msg            += std::to_string(n_interactions) + " interactions - "; 
      msg            += std::to_string(-max_warning_displayed * (-max_warning_displayed >= 0)) + " warnings not displayed.";
      if(max_warning_displayed < 0) lout << ansi::yellow(msg) << std::endl;
      else lout << ansi::green(msg) << std::endl;
    }
  };
};
