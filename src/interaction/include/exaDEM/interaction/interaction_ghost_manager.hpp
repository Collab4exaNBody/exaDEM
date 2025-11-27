#include <exanb/mpi/ghosts_comm_scheme.h>

#pragma once

//#define EXACOLD_PRINT_COMM

namespace exaDEM
{
  using namespace exanb;

  struct CellGhostDetails
  {
    uint32_t m_cell_i = 0;
    uint32_t m_partner_cell_i = 0;
    uint32_t m_size = 0;
    uint32_t m_shift = 0;
  };

  struct InteractionGhostManager
  {
    std::vector<std::vector<exaDEM::PlaceholderInteraction>> rbuf;
    std::vector<std::vector<exaDEM::PlaceholderInteraction>> sbuf;
    std::vector<std::vector<CellGhostDetails>> send_cell_config; // send details
    std::vector<std::vector<CellGhostDetails>> recv_cell_config; // recv details
    size_t n_procs = 0;

    template<typename InteractionT, typename GridT>
      void setup(exanb::GhostCommunicationScheme& ghost_scheme, MPI_Comm& comm, InteractionT& ges, GridT& grid)
      {
        // tags constants
        const int TAG_SIZE    = 100;
        const int TAG_CONFIG  = 101;
        const int TAG_PAYLOAD = 102;

        const auto & partners = ghost_scheme.m_partner;
        n_procs = partners.size();
        int my_rank, mpi_size;
        MPI_Comm_rank(comm, &my_rank);
        MPI_Comm_size(comm, &mpi_size);

        // resize
        rbuf.resize(n_procs);
        sbuf.resize(n_procs);
        send_cell_config.resize(n_procs);
        recv_cell_config.resize(n_procs);

        std::vector<int> config_size;
        config_size.resize(n_procs);

        for( size_t proc = 0 ; proc < n_procs ; proc++)
        {
          sbuf[proc].clear();
          auto& send_config = send_cell_config[proc];
          const auto& partner = partners[proc];

          send_config.clear();

          // build send buffers and send_config correctly
          uint32_t shift = 0;
          for (auto &it : partner.m_sends)
          {
            auto &interactions = ges[it.m_cell_i].m_data;
            if (interactions.empty()) {
              continue;
            }
						uint32_t n_sent = 0;

						for (size_t j = 0 ; j < interactions.size() ; j++) {
							exaDEM::PlaceholderInteraction& I = interactions[j]; //storage.get_particle_item(p, j);
							if (I.pair.ghost == InteractionPair::OwnerGhost) {
								sbuf[proc].push_back(I);
								++n_sent;
							}
						} 

						/*
							 auto& particle_p = it.m_particle_i;
							 auto& storage = ges[it.m_cell_i];
							 for (size_t i = 0; i < particle_p.size(); ++i)
							 {
							 size_t p = particle_p[i];
							 size_t n = storage.particle_number_of_items(p);

							 for (size_t j = 0 ; j < n ; j++)
							 {
							 exaDEM::PlaceholderInteraction& I = storage.get_particle_item(p, j);
							 if (I.pair.ghost == InteractionPair::OwnerGhost) {
							 if(I.pair.type > 13  || I.pair.type < 0) color_log::mpi_error("debug", "ERROOOOOOOOOOOOOOOOOOOOR");
							 sbuf[proc].push_back(I);
							 ++n_sent;
							 }

							 } 
							 }
						 */
						// push the *number actually sent* and the current shift
						if( n_sent > 0)
						{
							send_config.push_back({static_cast<uint32_t>(it.m_cell_i), static_cast<uint32_t>(it.m_partner_cell_i),
									n_sent, shift});
							shift += n_sent;
						}
					}

					// send cell configs
					if (partner.m_particles_to_send > 0)
					{
						MPI_Request req1 = MPI_REQUEST_NULL, req2 = MPI_REQUEST_NULL, req3 = MPI_REQUEST_NULL;
						config_size[proc] = static_cast<int>(send_config.size());
						MPI_Isend(&config_size[proc], 1, MPI_INT, proc, TAG_SIZE, comm, &req1);
						if (config_size[proc] > 0)
						{
							// send_config: send bytes (count = number_of_bytes)
							MPI_Isend(send_config.data(),
									static_cast<int>(send_config.size() * sizeof(CellGhostDetails)),
									MPI_BYTE,
									proc,
									TAG_CONFIG,
									comm,
									&req2);

							// payload: sbuf in bytes
							long long count = static_cast<long long>(sbuf[proc].size() ) * sizeof(exaDEM::PlaceholderInteraction);
							MPI_Isend(sbuf[proc].data(),
									count,
									MPI_BYTE,
									proc,
									TAG_PAYLOAD,
									comm,
									&req3);
						}
					}
				}

				for( size_t proc = 0 ; proc < n_procs ; proc++)
				{
					rbuf[proc].clear();
					auto& recv_config = recv_cell_config[proc];
					const auto& partner = partners[proc];
					recv_config.clear();

					// recv cell configs
					if (partner.m_particles_to_receive > 0)
					{
						MPI_Request rreq1 = MPI_REQUEST_NULL, rreq2 = MPI_REQUEST_NULL, rreq3 = MPI_REQUEST_NULL;
						MPI_Status  s1, s2, s3;
						int isize = 0;

						MPI_Irecv(&isize, 1, MPI_INT, proc, TAG_SIZE, comm, &rreq1);
						MPI_Wait(&rreq1, &s1);

						if (isize > 0)
						{
							recv_config.resize(isize);
							MPI_Irecv(recv_config.data(),
									static_cast<int>(recv_config.size() * sizeof(CellGhostDetails)),
									MPI_BYTE,
									proc,
									TAG_CONFIG,
									comm,
									&rreq2);
							MPI_Wait(&rreq2, &s2);

							// determine overall buffer size from recv_config: use the last entry
							auto &last = recv_config.back();
							size_t needed = last.m_shift + last.m_size;
							rbuf[proc].resize(needed);

							long long count = rbuf[proc].size() * sizeof(exaDEM::PlaceholderInteraction);
							MPI_Irecv(rbuf[proc].data(),
									count,
									MPI_BYTE,
									proc,
									TAG_PAYLOAD,
									comm,
									&rreq3);
							MPI_Wait(&rreq3, &s3);
						}
					}
				}
			}

		template<typename GridT, typename GridExtraDynamicStorage>
			void copy_interaction(GridT& grid, GridExtraDynamicStorage& ges)
			{
				IJK dims = grid.dimension();
				const auto cells = grid.cells();
#pragma omp parallel for schedule(dynamic)
				for(size_t proc=0 ; proc<n_procs ; proc++)
				{
					auto& buffer = rbuf[proc];
					for(auto& config : recv_cell_config[proc])
					{
						assert(config.m_size > 0);
						IJK loc = grid_index_to_ijk(dims, config.m_partner_cell_i);
						assert(config.m_partner_cell_i < ges.size());
						auto& storage = ges[config.m_partner_cell_i].m_data;
						if( !grid.is_ghost_cell(config.m_partner_cell_i) ) color_log::mpi_error("copy_interaction","This cell is not a ghost");

						uint64_t partner_id;
						uint32_t partner_cell;
						uint16_t partner_p;
						bool found = false;

						for(size_t i=config.m_shift ; i<config.m_shift+config.m_size ; i++)
						{
							assert(i<buffer.size());
							exaDEM::PlaceholderInteraction& item = buffer[i];
							// update info
							auto& owner = item.pair.owner(); 
							owner.cell = config.m_partner_cell_i; // defined by another mpi process (or himself if periodic boundary conditions)
							item.pair.ghost = InteractionPair::PartnerGhost; // identify this kind of interaction

							// check
							{
								bool is_idx_exist = false;
								assert(owner.p < cells[owner.cell].size());
								if( owner.p < cells[owner.cell].size() )
								{
									if(owner.id == cells[owner.cell][field::id][owner.p]) is_idx_exist = true;
								}
								if(!is_idx_exist)
								{
									const uint64_t* const __restrict__ ids = cells[owner.cell][field::id];
									for(size_t p=0 ; p<cells[owner.cell].size() ; p++) 
									{
										if(owner.id == ids[p])
										{
											owner.p = p;
											is_idx_exist = true;
										}
									} 
								}
								/*
									 if(!is_idx_exist)
									 {
									 const uint64_t* const __restrict__ ids = cells[owner.cell][field::id];
									 int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
									 std::cout << "Error in Proc: " << rank << " from the mpi process ID: " << proc << std::endl;
									 for(size_t p=0 ; p<cells[owner.cell].size() ; p++) std::cout << "ids are[" << int(p) << "]: " << ids[p] << std::endl;
									 item.print();
									 color_log::mpi_error("copy_interaction", "owner.id: " + std::to_string(owner.id) 
									 + " is != of cells[owner.cell][field::id][owner.p]: "  
									 + std::to_string(cells[owner.cell][field::id][owner.p]));
									 }
								 */
							}

							auto& partner = item.pair.partner();

							if(found && partner.id == partner_id)
							{
								partner.cell = partner_cell;
								partner.p = partner_p;
								found = true; // still true
							}
							else 
							{ 
								found = false; 
							}

							for(int z = loc.k-1 ; (z <= (loc.k+1)) && !found ; z++)
								for(int y = loc.j-1 ; (y <= (loc.j+1)) && !found ; y++)
									for(int x = loc.i-1 ; (x <= (loc.i+1)) && !found ; x++)
									{
										IJK loc_neigh = IJK{x,y,z};
										if( grid.contains(loc_neigh))
										{
											uint32_t neigh = grid_ijk_to_index(dims, loc_neigh);

											//if(neigh >= cells->size()) std::cout << "cell " << neigh << "/" << grid.number_of_cells() << " loc " << loc_neigh << " dims " << grid.dimension() << std::endl;
											assert(neigh < grid.number_of_cells());
											if( grid.is_ghost_cell(neigh) ) continue;
											const uint64_t* const __restrict__ ids = cells[neigh][field::id];
											for(size_t p=0 ; p<cells[neigh].size() ; p++)
											{
												if(partner.id == ids[p]) 
												{
													partner.cell = neigh;
													partner.p = p;
													partner_id = partner.id;
													partner_cell = neigh;
													partner_p = p;
													found = true;
												}
											}
										}
									}
							if(found) 
							{
								// check
								assert(partner.cell<grid.number_of_cells());
								assert(partner.p<cells[partner.cell].size());
								if(partner.id != cells[partner.cell][field::id][partner.p])
								{
									item.print();
									color_log::mpi_error("copy_interaction", "partner.id: " + std::to_string(partner.id) 
											+ " is != of cells[partner.cell][field::id][partner.p]: "  
											+ std::to_string(cells[partner.cell][field::id][partner.p]));
								}
								storage.push_back(item);
							}
						} 
					}
				}
			}
	};
}
