namespace exaDEM
{
  using namespace exanb;
  using VertexArray = ::onika::oarray_t<::exanb::Vec3d, EXADEM_MAX_VERTICES>;
  using NumberOfInteractionPerTypes = ::onika::oarray_t<int, NumberOfInteractionTypes>;
/** reuse functions defined into the particle strategy*/

/***************************/
/*  Global Block functions */
/***************************/

  template<int BLOCKX, int BLOCKY, typename TMPLC>
    __global__ void get_number_of_interations_pair(
        TMPLC cells,
        IJK dims,
        GridChunkNeighborsData nbh,
        shapes shps,
        double rVerlet,
        NumberOfInteractionPerTypes * count_data,
        size_t* cell_idx)
    {
      using BlockReduce = cub::BlockReduce<int, BLOCKX, cub::BLOCK_REDUCE_RAKING, BLOCKY>; // 8*8 blockDimXY>;
      const size_t cell_a = cell_idx[blockIdx.x];
      IJK loc_a = grid_index_to_ijk( dims, cell_a);

      assert(blockDim.x == BLOCKX);
      // cub stuff
      __shared__ typename BlockReduce::TempStorage temp_storage;

      // Struct to fill count_data at the enf
      int count[NumberOfInteractionTypes];
      for(size_t i = 0; i < NumberOfInteractionTypes ; i++)
      {
        count[i] = 0;
      }

      const unsigned int cell_a_particles = cells[cell_a].size();
      const auto stream_info = chunknbh_stream_info( nbh[cell_a] , cell_a_particles );
      const uint16_t* stream_base = stream_info.stream;
      const uint16_t* stream = stream_base;
      const uint32_t* __restrict__ particle_offset = stream_info.offset;

      const int32_t poffshift = stream_info.shift;

      for(unsigned int p_a= threadIdx.x; p_a<cell_a_particles ; p_a+= blockDim.x)
      {
        if( particle_offset!=nullptr ) stream = stream_base + particle_offset[p_a] + poffshift;

        unsigned int cell_groups = *(stream++); // number of cell groups for this neighbor list

        /** load data */
        particle_info p(cells, shps, cell_a, p_a);

        /** compute obb */
        OBB obb_i = p.shp->obb;
        quat conv_orient_i = p.get_quat();
        obb_i.rotate(conv_orient_i);
        obb_i.translate(vec3r{p.r.x, p.r.y, p.r.z});
        obb_i.enlarge(rVerlet);

        /** Count the number of interactions per thread */
        for(unsigned int cg=0; cg<cell_groups ;cg++)
        {
          header_nbh nbh_cg = decode_stream_header_nbh(loc_a, dims, stream);
          unsigned int nbh_cell_particles = cells[nbh_cg.cell_b].size();
          for(unsigned int chunk= threadIdx.y ; chunk<nbh_cg.nchunks ; chunk+= blockDim.y)
          {
            unsigned int p_b = nbh_cg.chunk_idx[chunk];
            if( p_b<nbh_cell_particles && (nbh_cg.cell_b!=cell_a || p_b!=p_a) )
            {
              particle_info p_nbh(cells, shps, nbh_cg.cell_b, p_b);
              if( intersect(rVerlet, obb_i, p_nbh))
              {
                count_interactions( rVerlet, count, !nbh_cg.is_ghost_b, p, p_nbh);
              }
            }
          }
        }
      }
      for(int i = 0; i < NumberOfInteractionTypes ; i++)
      {
        int aggregate = BlockReduce(temp_storage).Sum(count[i]);
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0) count_data[blockIdx.x][i] = aggregate;
      }
    }



  template<int BLOCKX, int BLOCKY, typename TMPLC>
    __global__ void fill_classifier_pair(
        InteractionSOA* data,
        TMPLC cells,
        IJK dims,
        GridChunkNeighborsData nbh,
        shapes shps,
        double rVerlet,
        NumberOfInteractionPerTypes * shift_data,
        size_t* cell_idx)
    {
      assert(blockDim.x == BLOCKX);
      using BlockScan = cub::BlockScan<int, BLOCKX, cub::BLOCK_SCAN_RAKING, BLOCKY>;
      const size_t cell_a = cell_idx[blockIdx.x];
      IJK loc_a = grid_index_to_ijk( dims, cell_a);

      // cub stuff
      __shared__ typename BlockScan::TempStorage temp_storage;

      // Struct to fill count_data at the enf
      int count[NumberOfInteractionTypes];
      int prefix[NumberOfInteractionTypes];
      for(size_t i = 0; i < NumberOfInteractionTypes ; i++)
      {
        count[i] = 0;
        prefix[i] = 0;
      }

      /** Get stream info containing neighbors data */
      const unsigned int cell_a_particles = cells[cell_a].size();
      const auto stream_info = chunknbh_stream_info( nbh[cell_a] , cell_a_particles );
      const uint16_t* stream_base = stream_info.stream;
      const uint16_t* stream = stream_base;
      const uint32_t* __restrict__ particle_offset = stream_info.offset;
      const int32_t poffshift = stream_info.shift;
      for(unsigned int p_a=threadIdx.x; p_a< cell_a_particles ; p_a+=blockDim.x)
      {
        if( particle_offset!=nullptr ) stream = stream_base + particle_offset[p_a] + poffshift;
        unsigned int cell_groups = *(stream++); // number of cell groups for this neighbor list

        /** load data */
        particle_info p(cells, shps, cell_a, p_a);

        /** compute obb */
        OBB obb_i = p.shp->obb;
        quat conv_orient_i = p.get_quat();
        obb_i.rotate(conv_orient_i);
        obb_i.translate(vec3r{p.r.x, p.r.y, p.r.z});
        obb_i.enlarge(rVerlet);

        /** Count the number of interactions per thread */
        for(unsigned int cg=0; cg<cell_groups ;cg++)
        {
          header_nbh nbh_cg = decode_stream_header_nbh(loc_a, dims, stream);
          unsigned int nbh_cell_particles = cells[nbh_cg.cell_b].size();
          for(unsigned int chunk= threadIdx.y ; chunk<nbh_cg.nchunks ; chunk+= blockDim.y)
          {
            unsigned int p_b = nbh_cg.chunk_idx[chunk];
            if( p_b<nbh_cell_particles && (nbh_cg.cell_b!=cell_a || p_b!=p_a) )
            {
              particle_info p_nbh(cells, shps, nbh_cg.cell_b, p_b);
              if( intersect(rVerlet, obb_i, p_nbh) )
              {
                count_interactions( rVerlet, count, !nbh_cg.is_ghost_b, p, p_nbh);
              }
            }
          }
        }
      }
      __syncthreads();

      NumberOfInteractionPerTypes sdata = shift_data[blockIdx.x];
      for(int type = 0 ; type < NumberOfInteractionTypes ; type++)
      {
        BlockScan(temp_storage).ExclusiveSum(count[type], prefix[type]);
        __syncthreads();
        prefix[type] += sdata[type];
      }
      Interaction item;
      for(unsigned int p_a=threadIdx.x; p_a< cell_a_particles ; p_a+=blockDim.x)
      {
        if( particle_offset!=nullptr ) stream = stream_base + particle_offset[p_a] + poffshift;
        unsigned int cell_groups = *(stream++); // number of cell groups for this neighbor list

        /** Get current particle info */
        particle_info p(cells, shps, cell_a, p_a);

        /** compute obb */
        OBB obb_i = p.shp->obb;
        quat conv_orient_i = p.get_quat();
        obb_i.rotate(conv_orient_i);
        obb_i.translate(vec3r{p.r.x, p.r.y, p.r.z});
        obb_i.enlarge(rVerlet);

        for(unsigned int cg=0; cg<cell_groups ;cg++)
        {
          header_nbh nbh_cg = decode_stream_header_nbh(loc_a, dims, stream);
          unsigned int nbh_cell_particles = cells[nbh_cg.cell_b].size();
          for(unsigned int chunk= threadIdx.y ; chunk<nbh_cg.nchunks ; chunk+= blockDim.y)
          {
            unsigned int p_b = nbh_cg.chunk_idx[chunk];
            if( p_b<nbh_cell_particles && (nbh_cg.cell_b!=cell_a || p_b!=p_a) )
            {
              /** Get nbh particle info */
              particle_info p_nbh(cells, shps, nbh_cg.cell_b, p_b);
              if( intersect(rVerlet, obb_i, p_nbh) )
              {
                /** Define interaction (section particle i) */
                item.id_i = p.id;
                item.cell_i = cell_a;
                item.p_i = p_a;
                /** Define interaction (section particle j) */
                item.cell_j = nbh_cg.cell_b;
                item.id_j = p_nbh.id;
                item.p_j = p_b;

                /** some basic checks */
                assert( cells[cell_a].size() == cell_a_particles);
                assert( p_a == item.p_i );
                assert(item.p_j < cells[cell_b].size());
                assert(item.p_i < cells[cell_a].size());

                /** here, we fill directly the interactionSOA data storage */
                fill_interactions( data, item, rVerlet, prefix, !nbh_cg.is_ghost_b, p, p_nbh);
              } // check
            } // p_b
          } // chunk
        } // cg
      } // p_a 
    } // fill 
} // namespace exaDEM

