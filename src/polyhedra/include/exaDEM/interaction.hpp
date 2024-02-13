#pragma once

namespace exaDEM
{

	/**
	 * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
	 */
	struct Interaction
	{
		exanb::Vec3d friction; /**< Friction vector associated with the interaction. */
		exanb::Vec3d moment;   /**< Moment vector associated with the interaction. */
		uint64_t id_i;         /**< Id of the first particle */
		uint64_t id_j;         /**< Id of the second particle */
		size_t cell_i;         /**< Index of the cell of the first particle involved in the interaction. */
		size_t cell_j;         /**< Index of the cell of the second particle involved in the interaction. */
		size_t p_i;            /**< Index of the particle within its cell for the first particle involved in the interaction. */
		size_t p_j;            /**< Index of the particle within its cell for the second particle involved in the interaction. */
		size_t sub_i;          /**< Sub-particle index for the first particle involved in the interaction. */
		size_t sub_j;          /**< Sub-particle index for the second particle involved in the interaction. */
		uint8_t type;          /**< Type of the interaction (e.g., contact type). */
		bool prev;             /**< prevent if the interaction can be active */

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
					" [cell: "<< cell_i << ", idx " << id_i << ", particle id: " << p_i << "] and" <<
					" [cell: "<< cell_j << ", idx " << id_j << ", particle id: " << p_j << "] : (friction: " <<
					friction << ", moment: " << moment << ")" << std::endl;
		}

		/**
		 * @brief return true if particles id and particles sub id are equals.
		 */
		bool operator==(Interaction& I)
		{
			if ( this->id_i == I.id_i && this->id_j == I.id_j && this->sub_i == I.sub_i && this->sub_j == I.sub_j && this->type == I.type)
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
			if ( this->id_i == I.id_i && this->id_j == I.id_j && this->sub_i == I.sub_i && this->sub_j == I.sub_j && this->type == I.type)
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
		std::vector<Interaction> extract_history(std::vector<Interaction>& interactions)
		{
			std::vector<Interaction> ret;
			const exanb::Vec3d null = {0,0,0};
#ifndef SERIAL
#pragma omp parallel
			{
				std::vector<Interaction> local;
#pragma omp for
				for( size_t i = 0 ; i < interactions.size() ; i++ )
				{
					if(interactions[i].moment == null && interactions[i].friction == null)
					{
						local.push_back(interactions[i]);
					}
				}

				if(local.size() > 0)
				{
#pragma omp critical
					{
						ret.insert(ret.end(), local.begin(), local.end());
					}
				}
			}
#else
			int last = interactions.size() - 1;
			for(int i = last ; i >= 0 ; i--)
			{
				if(interactions[i].moment == null && interactions[i].friction == null)
				{
					interactions[i] = interactions[last--];
				}
			}
			interactions.resize(last+1);
			ret = interactions;
#endif
			return ret;
		}

	inline 
		void update_friction_moment_old(std::vector<Interaction>& interactions, std::vector<Interaction>& history)
		{
			// stupid way
			size_t compt_interaction_copied = 0;

			for(size_t i = 0 ; i < history.size() ; i++)
			{
				auto & old_item = history[i];
				for (size_t j = 0 ; j < interactions.size() ; j++)
				{
					auto & item = interactions[j];
					if ( old_item == item ) 
					{
						item.update_friction_and_moment(old_item);
						compt_interaction_copied++;
						break;
					}
				}
			}
#ifdef DEBUG_INTERACTION
			std::cout << " Number of interaction copied: " << compt_interaction_copied << ". Number of old interactions: " << history.size() << ". Number of new interactions: " << interactions.size() << "." << std::endl;
#endif
			history.clear();
		}
	inline 
		void update_friction_moment(std::vector<Interaction>& interactions, std::vector<Interaction>& history)
		{

			if (history.size() == 0) return;
			if (interactions.size() == 0) return;



			// interactions and history are sorted
			size_t id = 0;
			auto& old_item    = history[0];
			auto& first_item  = interactions[0];
			// get first elem
			while( old_item < first_item && id < history.size()) old_item = history[++id];

			for(size_t it = 0 ; it < interactions.size() ; it++)
			{
				auto & item = interactions[it];
				if ( item < old_item  ) continue;

				id++; // incr

				if ( old_item == item ) 
				{
					item.update_friction_and_moment(old_item);
					if(id < history.size())
					{
						old_item = history[id];
					}
					else
					{
						return;
					}						
				}
				else
				{
					if(id < history.size())
					{
						old_item = history[id];
						it--;
					}
					else
					{
						return;
					}						
				}
			}
		}
}
