#pragma once

# include <exanb/core/particle_type_id.h>
# include <exaDEM/shapes.hpp>

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
    Particle decrypt(const STREAM& input, ParticleTypeMap& ptm)
    {
      std::string type_name;
      Particle p;
      input >> type_name;
      // debug
      lout << type_name << std::endl;
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
    Interaction decrypt(const STREAM& input)
    {
      Interaction I;
      input >> I.i >> I.j >> I.type >> I.subi >> I.subj;
      input >> I.n >> I.dn; 
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
  exaDEM::Interaction convert(const rockable::Interaction& input)
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
}
