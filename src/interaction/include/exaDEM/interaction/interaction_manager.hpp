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

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>

namespace exaDEM
{
  struct InteractionManager
  {
    std::vector<exaDEM::PlaceholderInteraction> hist;              // history
    std::vector<std::vector<exaDEM::PlaceholderInteraction>> list; //
    std::vector<std::vector<uint64_t>> ignore;
    size_t current_cell_id;
    size_t current_cell_particles;

    void reset(const size_t size)
    {
      list.resize(size);
      ignore.resize(size);
      for (size_t p = 0; p < size; p++)
      {
        list[p].clear();
        ignore[p].clear();
      }
    }

    void add_item(const size_t p, exaDEM::PlaceholderInteraction &I)
    {
      assert(p < list.size());
      //if(I.type() != InteractionTypeId::VertexVertex 
      //    || (I.type() == InteractionTypeId::VertexVertex && !skip_ignored_interactions(p, I)))
      if(!skip_ignored_interactions(p, I))
      {
        list[p].push_back(I);
      }
    }

    void add(std::vector<exaDEM::PlaceholderInteraction>& vec)
    {
      for(auto& it : vec)
      {
        add_item(it.pair.pi.p, it);
      }
    }

    size_t get_size()
    {
      size_t count(0);
      for (auto &it : list)
        count += it.size();
      return count;
    }

    template <bool use_history> 
      void update_extra_storage(CellExtraDynamicDataStorageT<PlaceholderInteraction> &storage)
      {
        const size_t total_size = this->get_size();
        size_t offset = 0;
        auto &info = storage.m_info;
        auto &data = storage.m_data;

        data.resize(total_size);

        for (size_t p = 0; p < list.size(); p++)
        {
          info[p].offset = offset;
          if constexpr (use_history)
            update(list[p], hist);
          std::copy(list[p].begin(), list[p].end(), data.data() + offset);
          info[p].size = list[p].size();
          offset += list[p].size();
        }
      }

    // call it after update persistent interaction
    void update_ignore_interaction()
    {
      size_t n = list.size();
      ignore.resize(n);
      for(size_t p=0 ; p<n ; p++)
      {
        auto& interactions = list[p];
        for(size_t i=0 ; i<interactions.size() ; i++) 
        {
          if( interactions[i].ignore_other_interactions())
          {
            auto& partner = interactions[i].j();
            auto& ignore_ids = ignore[p];
            bool add_info = true;
            for(size_t j=0 ; j < ignore_ids.size() ; j++)
            {
              if(partner.id == ignore_ids[j])
              {
                add_info = false;
                break;
              }
            }
            if(add_info) ignore_ids.push_back(partner.id);
          }           
        }
      }
    }

    bool skip_ignored_interactions(
        size_t p, 
        exaDEM::PlaceholderInteraction &I)
    {
      auto& partner = I.pair.partner();
      auto& ignore_ids = ignore[p];
      for(size_t j=0 ; j<ignore_ids.size() ; j++)
      {
        //if(owner.id == ignore_ids[j]) return true;
        if(partner.id == ignore_ids[j]) return true;
      }
      return false;
    }
  };

  inline void update_persistent_interactions(
      InteractionManager& manager, 
      CellExtraDynamicDataStorageT<PlaceholderInteraction> &storage)
  {
    size_t n_interactions = storage.m_data.size();
    for(size_t i = 0; i <n_interactions ; i++)
    {
			PlaceholderInteraction& I = storage.m_data[i];
      if( I.pair.type > 13 ) continue; //color_log::mpi_error("update_persistent_interactions", "Interaction type > 13 at position " + std::to_string(i) + " / " + std::to_string(size));
			if( I.persistent() )
			{
				if( I.pair.owner().cell != manager.current_cell_id ) 
					color_log::mpi_error("update_persistent_interactions", "This interaction is illformed, owner.cell should be: " 
							+ std::to_string(manager.current_cell_id) 
							+ " cell: " + std::to_string(I.pair.owner().cell)
							+ " p; " + std::to_string(I.pair.owner().p) 
							+ " id: " + std::to_string(I.pair.owner().id)); 
				if( I.pair.owner().p >= manager.current_cell_particles ) 
					color_log::mpi_error("update_persistent_interactions", "This interaction is illformed, owner.p should be inferior to: " 
							+ std::to_string(manager.current_cell_particles)
							+ " cell: " + std::to_string(I.pair.owner().cell)
							+ " p; " + std::to_string(I.pair.owner().p) 
							+ " id: " + std::to_string(I.pair.owner().id)); 
				manager.list[I.pair.owner().p].push_back(I);
			}
		}

    for(size_t p=0; p<manager.list.size() ; p++)
    {
       std::stable_sort(manager.list[p].begin(), manager.list[p].end());
    }
	}

	// sequential
		inline void extract_history(
				std::vector<PlaceholderInteraction> &local, 
				const PlaceholderInteraction *data, 
				const unsigned int size)
		{
			local.clear();
			for (size_t i = 0; i < size; i++)
			{
				const auto &item = data[i];
        if( item.pair.type > 13 ) continue; 
				if (item.active())
				{
					local.push_back(item);
				}
			}
		}

	template<typename ParticleVertexViewT>
		bool check_stiked_face(
				std::vector<exaDEM::PlaceholderInteraction>& interactions, 
				size_t p, 
				ParticleVertexViewT& pvv)
		{
			std::vector<int> vertex_id = {};
			std::vector<Vec3d> vertices;
			// identify vertices
			for(size_t i=0 ; i<interactions.size() ; i++)
			{
				if(interactions[i].type() == InteractionTypeId::InnerBond)
				{
					vertex_id.push_back(interactions[i].pair.pi.sub);
				} 
			}

			if( vertex_id.size() == 0 ) return true; // true cause no stiked face 
			vertices.resize(vertex_id.size());

			for(size_t i=0 ; i<vertex_id.size() ; i++)
			{
				vertices[i] = pvv[vertex_id[i]];
			}

			if(vertices.size() <= 2)
			{
				color_log::warning("interaction_manager::check_stiked_face", "sticked face is illformed (n_vertices should be >= 3) with only " + std::to_string(vertices.size()) + " vertices");
				return false;
			}

			Vec3d va = vertices[1] - vertices[0];
			Vec3d vb = vertices[2] - vertices[0];

			bool res = true;

			Vec3d normal = exanb::cross(va, vb);
			constexpr double tol = 1.e-10;

			// Check that remaining vertices belong to the same plane
			for (size_t j=3; j<vertices.size(); j++)
			{
				Vec3d vj = vertices[j] - vertices[0];
				double distance_to_plane = std::abs(exanb::dot(vj, normal));

				if (distance_to_plane > tol)
				{
					//color_log::error(
					color_log::warning(
							"interaction_manager::check_sticked_face",
							"Sticked face is not coplanar: vertex " + std::to_string(j) +
							" is out of the plane defined by the first three vertices. " +
							"Distance to plane = " + std::to_string(distance_to_plane)
							);
					return false;
				}
			}
			return res;
		}
} // namespace exaDEM
