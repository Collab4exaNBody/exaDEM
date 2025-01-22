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
        if constexpr (use_history)
          update_friction_moment(list[p], hist);
        std::copy(list[p].begin(), list[p].end(), data.data() + offset);
        info[p].size = list[p].size();
        offset += list[p].size();
      }
    }
  };
} // namespace exaDEM
