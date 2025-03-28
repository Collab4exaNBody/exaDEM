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
  struct interaction_manager
  {
    std::vector<exaDEM::Interaction> hist;              // history
    std::vector<std::vector<exaDEM::Interaction>> list; //

    void reset(const size_t size)
    {
      list.resize(size);
      for (size_t p = 0; p < size; p++)
      {
        list[p].clear();
      }
    }

    void add_item(const size_t p, exaDEM::Interaction &I)
    {
      assert(p < list.size());
      list[p].push_back(I);
    }

    size_t get_size()
    {
      size_t count(0);
      for (auto &it : list)
        count += it.size();
      return count;
    }

    template <bool use_history> void update_extra_storage(CellExtraDynamicDataStorageT<Interaction> &storage)
    {
      const size_t total_size = this->get_size();
      size_t offset = 0;
      auto &info = storage.m_info;
      auto &data = storage.m_data;

      data.resize(total_size);

      for (size_t p = 0; p < list.size(); p++)
      {
        info[p].offset = offset;
        //if constexpr (use_history)
          //update_friction_moment(list[p], hist);
        std::copy(list[p].begin(), list[p].end(), data.data() + offset);
        info[p].size = list[p].size();
        offset += list[p].size();
      }
    }
  };
} // namespace exaDEM
