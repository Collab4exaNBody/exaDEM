#pragma once


#include <exaDEM/interaction/placeholder_interaction.hpp>
#include <exaDEM/classifier/interaction_wrapper.hpp>

namespace exaDEM
{
  /*
     struct ResetBreakInterface
     {
     bool* const ptr;
     ONIKA_HOST_DEVICE_FUNC operator()(size_t i) const
     {
     ptr[i] = false;
     }
     };
   */

  template <typename T> using vector_t = onika::memory::CudaMMVector<T>;

  struct Interface
  {
    // Important assumption: interactions are stored contiguously
    size_t loc; // Location in the classifier
    size_t size; // Number of interactions composed this interface
  };

  // Thread Local Storage
  struct InterfaceBuildManager
  {
    std::vector<Interface> data;
  };

  struct InterfaceManager
  {
    vector_t<Interface> data;
    vector_t<uint8_t> break_interface; // warning on gpu 
    void resize(size_t new_size)
    {
      data.clear();
      data.resize(new_size);
      break_interface.resize(new_size);
      std::fill(break_interface.begin(), break_interface.end(), false);
    }
    size_t size() { return data.size(); }
  };


	inline bool check_interface_consistency(
			InterfaceBuildManager& interfaces, 
			ClassifierContainer<InteractionType::StickedParticles>& interactions)
	{
    int res = 0;
    size_t n_interactions = interactions.size();

    #pragma omp parallel for reduction(+: res)
    for(size_t i=0 ; i<n_interactions ; i++)
    {
      auto [loc, size] = interfaces.data[i];

      uint64_t id_i = interactions.particle_id_i(loc);
      uint64_t id_j = interactions.particle_id_j(loc);
      for(size_t next=loc+1; next<loc+size ; next++)
      {
				if(id_i != interactions.particle_id_i(next) 
						|| id_j != interactions.particle_id_j(next))
				{
          res += 1;
				}
			}
		}

    if(res == 0) return true;
 
		color_log::warning("check_interface_consistency", 
				std::to_string(res) + " interface are not defined correctly.\n" 
				+ "The interactions that compose the interface are not all defined between the same particles.");
		return false;
	}

	// CPU only
	inline void rebuild_interface_Manager(InterfaceBuildManager& interfaces, ClassifierContainer<InteractionType::StickedParticles>& interactions)
	{
		interfaces.data.clear();
		size_t n_interactions = interactions.size();
		size_t loc = 0;
		while( loc<n_interactions )
		{
			uint64_t idloci = interactions.particle_id_i(loc);
			uint64_t idlocj = interactions.particle_id_j(loc);
			size_t n = 1;
			uint64_t idni = interactions.particle_id_i(loc + n);
			uint64_t idnj = interactions.particle_id_j(loc + n);
			n++;
			while(loc + n  < n_interactions
					&& idloci == idni 
					&& idlocj == idnj)
			{
				idni = interactions.particle_id_i(loc + n);
				idnj = interactions.particle_id_j(loc + n);
				n++;
			}
      if( loc + n != n_interactions ) n--; // exclude the last element that failed the test 
			//std::cout << loc << "/" << n_interactions << " n " << n << std::endl;
			Interface interface = {loc, n};
      //lout << " loc " << loc << " n " << n << std::endl;
			interfaces.data.push_back(interface);
			loc += n;
		}
	}
}
