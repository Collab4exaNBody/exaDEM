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
#pragma once

#include <iostream>
//#include <ostream>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_stream.h>
#include <exaDEM/interaction/interaction_pair.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/inner_bond_interaction.hpp>
#include <exaDEM/color_log.hpp>

namespace exaDEM
{
  enum InteractionType
  { 
    ParticleParticle,
    ParticleDriver,
    InnerBond 
  };

  struct InteractionTypeId
  {
    static constexpr int NTypesParticleParticle = 13;
    static constexpr int VertexVertex = 0;
    static constexpr int VertexEdge = 1;
    static constexpr int VertexFace = 2;
    static constexpr int EdgeEdge = 3;
    static constexpr int VertexCylinder = 4;
    static constexpr int VertexSurface = 5;
    static constexpr int VertexBall = 6;
    static constexpr int NTypesStickecParticles = 1;
    static constexpr int InnerBond = 13;
    static constexpr int NTypes = NTypesParticleParticle + NTypesStickecParticles;
  };


	template <typename GridT> 
		inline bool filter_duplicates(
				const GridT &G,
				const ParticleSubLocation& owner,
				const ParticleSubLocation& partner,
				int type)
		{
			if (type < 4) // polyhedron - polyhedron or sphere - sphere
			{
				if (G.is_ghost_cell(owner.cell))
				{
					return false;
				}
			}
			return true;
		}

	template <typename GridT, typename InteractionT> 
		inline bool filter_duplicates(
				const GridT &G, 
				const InteractionT &I)
		{
      return filter_duplicates(G, I.pair.owner(), I.pair.partner(), I.type());
		}

	static constexpr size_t constexpr_max(std::size_t a, std::size_t b) 
	{
		return a > b ? a : b;
	}

	template <typename T, typename... Ts>
		static constexpr size_t max_sizeof() {
			if constexpr (sizeof...(Ts) == 0) {
				return sizeof(T);
			} else {
				return constexpr_max(sizeof(T), max_sizeof<Ts...>());
			}
		}


	/**
	 * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
	 */
	struct PlaceholderInteraction
	{
		static constexpr int PlaceholderInteractionPairSize = sizeof(InteractionPair);
		static constexpr int PlaceholderInteractionSize = max_sizeof<exaDEM::Interaction, exaDEM::InnerBondInteraction, exaDEM::InnerBondInteraction>() - PlaceholderInteractionPairSize;

		static_assert(PlaceholderInteractionSize > 0);

		static constexpr size_t MaxAlign =
			std::max({alignof(Interaction), alignof(InnerBondInteraction)});

		static_assert(PlaceholderInteractionSize > 0);

    // members
		InteractionPair pair;
		alignas(MaxAlign) uint8_t data[PlaceholderInteractionSize];

		ParticleSubLocation& i() { return pair.pi; }
		ParticleSubLocation& j() { return pair.pj; }
		ParticleSubLocation& driver() { return j(); }
		uint16_t type() { return pair.type; } 
		uint16_t type() const { return pair.type; } 
		uint32_t cell() { return pair.owner().cell; } // associate cell -> cell_i
    InteractionPair& pair_info() { return pair; }
    const InteractionPair& pair_info() const { return pair; }

		/**
		 * @brief Displays the Interaction data.
		 */
		void print()
		{
			if( type() < InteractionTypeId::NTypesParticleParticle ) 
			{
				return this->as<Interaction>().print();
			}
			else if( type() == InteractionTypeId::InnerBond )
			{
				return this->as<InnerBondInteraction>().print();
			}
		}

		/**
		 * @brief Displays the Interaction data.
		 */
		void print() const
		{
			if( type() < InteractionTypeId::NTypesParticleParticle ) 
			{
				return this->as<Interaction>().print();
			}
			else if( type() == InteractionTypeId::InnerBond )
			{
				return this->as<InnerBondInteraction>().print();
			}
		}

		void update(PlaceholderInteraction& in)
		{
			std::memcpy(data, in.data, PlaceholderInteractionSize * sizeof(uint8_t)); 
		}

		ONIKA_HOST_DEVICE_FUNC void clear_placeholder()
		{
			memset( data, 0, PlaceholderInteractionSize);
		}

		ONIKA_HOST_DEVICE_FUNC bool active() const 
		{ 
			if( type() < InteractionTypeId::NTypesParticleParticle ) 
			{
				return this->as<Interaction>().active();
			}
			else if( type() == InteractionTypeId::InnerBond )
			{
				return this->as<InnerBondInteraction>().active();
			}
      pair.print();
			color_log::mpi_error("PlaceholderInteraction::active", 
					"The type value of this interaction is invalid: "  + std::to_string(type()));
			std::exit(EXIT_FAILURE);
		}

    // Defines whether an interaction will be reconstructed or not.
		ONIKA_HOST_DEVICE_FUNC bool persistent()
		{ 
			if( type() < InteractionTypeId::NTypesParticleParticle ) 
			{
				return this->as<Interaction>().persistent();
			}
			else if( type() == InteractionTypeId::InnerBond )
			{
				return this->as<InnerBondInteraction>().persistent();
			}
			color_log::mpi_error("PlaceholderInteraction::persistent", 
					"The type value of this interaction is invalid: " + std::to_string(type()));
			std::exit(EXIT_FAILURE);
		}

		ONIKA_HOST_DEVICE_FUNC bool ignore_other_interactions() 
		{ 
			if( type() < InteractionTypeId::NTypesParticleParticle ) 
			{
				return this->as<Interaction>().ignore_other_interactions();
			}
			else if( type() == InteractionTypeId::InnerBond )
			{
				return this->as<InnerBondInteraction>().ignore_other_interactions();
			}
			color_log::mpi_error("PlaceholderInteraction::ignore_other_interactions", 
					"The type value of this interaction is invalid");
			std::exit(EXIT_FAILURE);
		}

		ONIKA_HOST_DEVICE_FUNC void reset()
		{ 
			if( type() < InteractionTypeId::NTypesParticleParticle ) 
			{
				this->as<Interaction>().reset();
			}
			else if( type() == InteractionTypeId::InnerBond )
			{
				this->as<InnerBondInteraction>().reset();
			}
      else  
      {
			  color_log::mpi_error("PlaceholderInteraction::reset", 
					"The type value of this interaction is invalid: " + std::to_string(type()));
      }
		}

		template <typename T>
			T& as()
			{
				static_assert(std::is_standard_layout_v<T>,
						"as<T>() requires a standard-layout type");
				static_assert(sizeof(T) >= sizeof(InteractionPair),
						"Type T must contain at least InteractionPair");
				static_assert(alignof(T) <= alignof(PlaceholderInteraction),
						"Type T alignment exceeds PlaceholderInteraction alignment");
				return *reinterpret_cast<T*>(this);
			}

		template <typename T>
			const T& as() const
			{
				static_assert(std::is_standard_layout_v<T>,
						"as<T>() requires a standard-layout type");
				static_assert(sizeof(T) >= sizeof(InteractionPair),
						"Type T must contain at least InteractionPair");
				static_assert(alignof(T) <= alignof(PlaceholderInteraction),
						"Type T alignment exceeds PlaceholderInteraction alignment");

				return *reinterpret_cast<const T*>(this);
			}

		template <InteractionType IT>
			auto& convert()
			{
				if constexpr( IT == ParticleParticle ) return as<Interaction>();
				if constexpr( IT == InnerBond ) return as<InnerBondInteraction>();
				color_log::mpi_error("PlaceholderInteraction::as<InteractionType>", 
						"Error, no Interaction type is defined for this value of InteractionType");
			}


		ONIKA_HOST_DEVICE_FUNC bool operator==(PlaceholderInteraction& I)
		{
			return (pair == I.pair);
		}

		ONIKA_HOST_DEVICE_FUNC bool operator==(const PlaceholderInteraction& I) const
		{
			return (pair == I.pair);
		}

		ONIKA_HOST_DEVICE_FUNC bool operator<(PlaceholderInteraction& I)
		{
			return (pair < I.pair);
		}

		ONIKA_HOST_DEVICE_FUNC bool operator<(const PlaceholderInteraction& I) const
		{
			return (pair < I.pair);
		}
	};

	inline std::pair<bool, PlaceholderInteraction&> get_interaction(std::vector<PlaceholderInteraction> &list, PlaceholderInteraction &I)
	{
		auto iterator = std::find(list.begin(), list.end(), I);
		// assert(iterator == std::end(list) && "This interaction is NOT in the list");
		bool exist = iterator == std::end(list);
		return {exist, *iterator};
	}

	inline std::vector<PlaceholderInteraction> extract_history_omp(std::vector<PlaceholderInteraction> &interactions)
	{
		std::vector<PlaceholderInteraction> res;
#   pragma omp parallel
		{
			std::vector<PlaceholderInteraction> tmp;
#     pragma omp for
			for (size_t i = 0; i < interactions.size(); i++)
			{
				auto& I = interactions[i];
				if( I.active() ) tmp.push_back(I);
			}

			if (tmp.size() > 0)
			{
#       pragma omp critical
				{
					res.insert(res.end(), tmp.begin(), tmp.end());
				}
			}
		}

		return res;
	}

	inline void update_omp(
			std::vector<PlaceholderInteraction> &interactions, 
			std::vector<PlaceholderInteraction> &history)
	{
#   pragma omp parallel for
		for (size_t it = 0; it < interactions.size(); it++)
		{
			auto &item = interactions[it];
			auto lower = std::lower_bound(history.begin(), history.end(), item);
			if (lower != history.end())
			{
				if (item == *lower)
				{
					item.update(*lower);
				}
			}
		}
	}

	inline void update(
			std::vector<PlaceholderInteraction> &interactions, 
			std::vector<PlaceholderInteraction> &history)
	{
		for (size_t it = 0; it < interactions.size(); it++)
		{
			auto &item = interactions[it];
			auto lower = std::lower_bound(history.begin(), history.end(), item);
			if (lower != history.end())
			{
				if (item == *lower)
				{
					item.update(*lower);
				}
			}
		}
	}
} // namespace exaDEM
