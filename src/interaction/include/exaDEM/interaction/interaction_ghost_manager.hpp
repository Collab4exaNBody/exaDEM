#include <exanb/mpi/ghosts_comm_scheme.h>

#pragma once

namespace exaDEM {
struct CellGhostDetails {
  uint32_t m_partner_cell_i = 0;
  uint32_t m_size = 0;
  uint32_t m_shift = 0;
};

struct InteractionGhostManager {
  std::vector<std::vector<exaDEM::PlaceholderInteraction>> rbuf;
  std::vector<std::vector<exaDEM::PlaceholderInteraction>> sbuf;
  std::vector<std::vector<CellGhostDetails>> send_cell_config;  // send details
  std::vector<std::vector<CellGhostDetails>> recv_cell_config;  // recv details
};
}  // namespace exaDEM
