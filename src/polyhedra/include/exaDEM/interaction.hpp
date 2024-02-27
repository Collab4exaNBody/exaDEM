#pragma once

namespace exaDEM
{

	/**
	 * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
	 */
	struct Interaction
	{
		exanb::Vec3d friction= {0,0,0};  /**< Friction vector associated with the interaction. */
		exanb::Vec3d moment = {0,0,0};   /**< Moment vector associated with the interaction. */
		uint64_t id_i;         /**< Id of the first particle */
		uint64_t id_j;         /**< Id of the second particle */
		size_t cell_i;         /**< Index of the cell of the first particle involved in the interaction. */
		size_t cell_j;         /**< Index of the cell of the second particle involved in the interaction. */
		uint16_t p_i;            /**< Index of the particle within its cell for the first particle involved in the interaction. */
		uint16_t p_j;            /**< Index of the particle within its cell for the second particle involved in the interaction. */
		uint16_t sub_i;          /**< Sub-particle index for the first particle involved in the interaction. */
		uint16_t sub_j;          /**< Sub-particle index for the second particle involved in the interaction. */
		uint16_t type;          /**< Type of the interaction (e.g., contact type). */

		/**
		 * @brief Resets the Interaction structure by setting friction and moment vectors to zero.
		 */
		void reset()
		{
			friction = {0, 0, 0};
			moment = {0, 0, 0};
		}

		/**
		 * @brief Displays the Interaction data.
		 */
		void print()
		{
			std::cout << "Interaction(type = " << int(type) << 
					" [cell: "<< cell_i << ", idx " << p_i << ", particle id: " << id_i << "] and" <<
					" [cell: "<< cell_j << ", idx " << p_j << ", particle id: " << id_j << "] : (friction: " <<
					friction << ", moment: " << moment << ")" << std::endl;
		}

		/**
		 * @brief Displays the Interaction data.
		 */
		void print() const
		{
			std::cout << "Interaction(type = " << int(type) << 
					" [cell: "<< cell_i << ", idx " << id_i << ", particle id: " << p_i << "] and" <<
					" [cell: "<< cell_j << ", idx " << id_j << ", particle id: " << p_j << "] : (friction: " <<
					friction << ", moment: " << moment << ")" << std::endl;
		}

		/**
		 * @brief return true if particles id and particles sub id are equals.
		 */
		bool operator==(Interaction& I)
		{
			if( this->id_i == I.id_i && 
					this->id_j == I.id_j && 
					this->sub_i == I.sub_i && 
					this->sub_j == I.sub_j && 
					this->type == I.type)
			{
				return true;
			}
			else
			{
				return false;
			}
		}

		/**
		 * @brief return true if particles id and particles sub id are equals.
		 */
		bool operator==(const Interaction& I) const
		{
			if( this->id_i == I.id_i && 
					this->id_j == I.id_j && 
					this->sub_i == I.sub_i && 
					this->sub_j == I.sub_j && 
					this->type == I.type)
			{
				return true;
			}
			else
			{
				return false;
			}
		}

		bool operator<(const Interaction& I) const
		{
			if ( this->id_i < I.id_i ) { return true; }
			else if ( this->id_i == I.id_i && this->id_j < I.id_j ) { return true; }
			else if ( this->id_i == I.id_i && this->id_j == I.id_j && this->sub_i < I.sub_i ) { return true; }
			else if ( this->id_i == I.id_i && this->id_j == I.id_j && this->sub_i == I.sub_i && this->sub_j < I.sub_j) { return true; }
			else if ( this->id_i == I.id_i && this->id_j == I.id_j && this->sub_i == I.sub_i && this->sub_j == I.sub_j && this->type < I.type){ return true; }
			else return false;
		}

		void update(Interaction& I)
		{
			this->cell_i = I.cell_i;
			this->cell_j = I.cell_j;
			this->p_i = I.p_i;
			this->p_j = I.p_j;
		}

		void update_friction_and_moment(Interaction& I)
		{
			this->friction = I.friction;
			this->moment = I.moment;
		}
	};


	inline
		std::pair<bool, Interaction&> get_interaction(std::vector<Interaction>& list, Interaction& I)
		{
			auto iterator = std::find (list.begin(), list.end(), I);
			//assert(iterator == std::end(list) && "This interaction is NOT in the list"); 
			bool exist = iterator == std::end(list);
			return {exist, *iterator};
		}

	inline
		std::vector<Interaction> extract_history_omp(std::vector<Interaction>& interactions)
		{
			std::vector<Interaction> ret;
			const exanb::Vec3d null = {0,0,0};
#pragma omp parallel
			{
				std::vector<Interaction> tmp;
#pragma omp for
				for( size_t i = 0 ; i < interactions.size() ; i++ )
				{
					if(interactions[i].moment != null || interactions[i].friction != null)
					{
						tmp.push_back(interactions[i]);
					}
				}

				if(tmp.size() > 0)
				{
#pragma omp critical
					{
						ret.insert(ret.end(), tmp.begin(), tmp.end());
					}
				}
			}

			return ret;
		}

	inline 
		void update_friction_moment_omp(std::vector<Interaction>& interactions, std::vector<Interaction>& history)
		{
#pragma omp parallel for
			for(size_t it = 0 ; it < interactions.size() ; it++)
			{
				auto & item = interactions[it];
				auto lower = std::lower_bound( history.begin(), history.end(), item );
				if(lower != history.end() )
				{
					if( item == *lower )
					{
						item.update_friction_and_moment(*lower);
					}
				}
			}
		}

	inline void update_friction_moment(std::vector<Interaction>& interactions, std::vector<Interaction>& history)
	{
		[[maybe_unused]] int number_of_active_interactions = history.size();
		[[maybe_unused]] int count_number_of_update = 0;
		for(size_t it = 0 ; it < interactions.size() ; it++)
		{
			auto & item = interactions[it];
			auto lower = std::lower_bound( history.begin(), history.end(), item );
			if(lower != history.end() )
			{
				if( item == *lower )
				{
					item.update_friction_and_moment(*lower);
					count_number_of_update++;
				}
			}
		}
		/*
		// not always true so I comment it. (ghost areas)
#ifndef NDEBUG
		if ( count_number_of_update != number_of_active_interactions)
		{
			std::cout << "count_number_of_update: " <<count_number_of_update << 
				" should be equal to " << number_of_active_interactions << std::endl;
		}

		std::vector<bool> markers(history.size());
		markers.assign(history.size(), false);

    for(size_t it = 0 ; it < interactions.size() ; it++)
    {
      auto & item = interactions[it];
      auto lower = std::lower_bound( history.begin(), history.end(), item );
      if(lower != history.end() )
      {
        if( item == *lower )
        {
					markers[std::distance(history.begin(), lower)] = true;
        }
      }
    }
		for(size_t it = 0 ; it < history.size() ; it++)
		{
			if( markers[it] == false ) history[it].print();
		}
#endif
		assert ( count_number_of_update == number_of_active_interactions );
		*/
	}

	// sequential
	inline void extract_history(std::vector<Interaction>& local, const Interaction * data, const unsigned int size)
	{
		const exanb::Vec3d null = {0,0,0};
		local.clear();
		for(size_t i = 0 ; i < size ; i++)
		{
			const auto& item = data[i];
			if(item.moment != null || item.friction != null)
			{
				local.push_back(item);
			}
		}			
	}

}
