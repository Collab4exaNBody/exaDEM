#include <exanb/mpi/ghosts_comm_scheme.h>

#pragma once

namespace exaDEM {
struct CellGhostDetails {
  uint32_t partner_cell_i_ = 0;
  uint32_t size_ = 0;
  uint32_t shift_ = 0;
};

struct InteractionGhostManager {
  std::vector<std::vector<exaDEM::PlaceholderInteraction>> rbuf_;
  std::vector<std::vector<exaDEM::PlaceholderInteraction>> sbuf_;
  std::vector<std::vector<CellGhostDetails>> send_cell_config_;  // send details
  std::vector<std::vector<CellGhostDetails>> recv_cell_config_;  // recv details
};
}  // namespace exaDEM
