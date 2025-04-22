
namespace exaDEM
{
  using namespace exanb;
  using VertexArray = ::onika::oarray_t<::exanb::Vec3d, EXADEM_MAX_VERTICES>;
  using NumberOfInteractionPerTypes = ::onika::oarray_t<int, NumberOfInteractionTypes>;

/***************************/
/*  Device Block functions */
/***************************/

  ONIKA_HOST_DEVICE_FUNC void count_interaction_block(
      double rVerlet,
      int count[],
      bool is_not_ghost_b,
      particle_info& p,
      particle_info& p_nbh)
  {
    // default value of the interaction studied (A or i -> B or j)
    if (p.id >= p_nbh.id)
    {
      if (is_not_ghost_b)
        return;
    }

    /** some renames */
    auto& shp = p.shp;
    auto& vertices_a = p.vertices;
    auto& shp_nbh = p_nbh.shp;
    auto& vertices_b = p_nbh.vertices;

    // get particle j data.
    const int nv = shp->get_number_of_vertices();
    const int ne = shp->get_number_of_edges();
    const int nf = shp->get_number_of_faces();
    const int nv_nbh = shp_nbh->get_number_of_vertices();
    const int ne_nbh = shp_nbh->get_number_of_edges();
    const int nf_nbh = shp_nbh->get_number_of_faces();

    ONIKA_CU_BLOCK_Y_SIMD_FOR(int, i, 0, nv)
    {
      ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, nv_nbh)
      {
        if (exaDEM::filter_vertex_vertex(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh))
        {
          count[VERTEX_VERTEX]++; // vertex-vertex
        }
      }
      ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, ne_nbh)
      {
        bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
        count[VERTEX_EDGE] += contact * 1; // vertex - edge
      }
      ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, nf_nbh)
      {
        bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
        count[VERTEX_FACE] += contact * 1; // vertex - face
      }
    }

    ONIKA_CU_BLOCK_Y_SIMD_FOR(int, i, 0, ne)
    {
      ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, ne_nbh)
      {
        bool contact = exaDEM::filter_edge_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
        count[EDGE_EDGE] += contact * 1; // edge - edge
      }
    }

    // interaction of from particle j to particle i
    ONIKA_CU_BLOCK_Y_SIMD_FOR(int, j, 0, nv_nbh)
    {
      ONIKA_CU_BLOCK_SIMD_FOR(int, i, 0, ne)
      {
        bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
        count[VERTEX_EDGE] += contact * 1; // edge - vertex
      }

      ONIKA_CU_BLOCK_SIMD_FOR(int, i, 0, nf)
      {
        bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
        count[VERTEX_FACE] += contact * 1; // face - vertex
      }
    }
  }


  ONIKA_HOST_DEVICE_FUNC void fill_interaction_block(
      InteractionSOA* data,
      exaDEM::Interaction& item,
      double rVerlet,
      int prefix[],
      bool is_not_ghost_b,
      particle_info& p,
      particle_info& p_nbh)
  {
    // default value of the interaction studied (A or i -> B or j)
    if (p.id >= p_nbh.id)
    {
      if (is_not_ghost_b)
        return;
    }

    /** some renames */
    auto& shp        = p.shp;
    auto& vertices_a = p.vertices;
    auto& shp_nbh    = p_nbh.shp;
    auto& vertices_b = p_nbh.vertices;

    // get particle j data.
    const int nv = shp->get_number_of_vertices();
    const int ne = shp->get_number_of_edges();
    const int nf = shp->get_number_of_faces();
    const int nv_nbh = shp_nbh->get_number_of_vertices();
    const int ne_nbh = shp_nbh->get_number_of_edges();
    const int nf_nbh = shp_nbh->get_number_of_faces();

    ONIKA_CU_BLOCK_Y_SIMD_FOR(int, i, 0, nv)
    {
      item.sub_i = i;
      item.type = 0;
      ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, nv_nbh)
      {
        if (exaDEM::filter_vertex_vertex(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh))
        {
          item.sub_j = j;
          data[item.type].set(prefix[item.type]++, item);
        }
      }

      item.type = 1;
      // vertex - edge
      ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, ne_nbh)
      {
        bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
        if (contact)
        {
          item.sub_j = j;
          data[item.type].set(prefix[item.type]++, item);
        }
      }
      item.type = 2;
      // vertex - face
      ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, nf_nbh)
      {
        bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
        if (contact)
        {
          item.sub_j = j;
          data[item.type].set(prefix[item.type]++, item);
        }
      }
    }
    item.type = 3;
    ONIKA_CU_BLOCK_Y_SIMD_FOR(int, i, 0, ne)
    {
      item.sub_i = i;
      // edge - edge
      ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, ne_nbh)
      {
        bool contact = exaDEM::filter_edge_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
        if (contact)
        {
          item.sub_j = j;
          data[item.type].set(prefix[item.type]++, item);
        }
      }
    }

    // interaction of from particle j to particle i
    ONIKA_CU_BLOCK_Y_SIMD_FOR(int, j, 0, nv_nbh)
    {
      item.type = 1;
      item.sub_i = j;
      // edge - vertex
      ONIKA_CU_BLOCK_SIMD_FOR(int, i, 0, ne)
      {
        bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
        if (contact)
        {
          item.sub_j = i;
          data[item.type].set(prefix[item.type]++, item);
        }
      }

      item.type = 2;
      // face - vertex
      ONIKA_CU_BLOCK_SIMD_FOR(int, i, 0, nf)
      {
        bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
        if (contact)
        {
          item.sub_j = i;
          data[item.type].set(prefix[item.type]++, item);
        }
      }
    }
  }

/***************************/
/*  Global Block functions */
/***************************/

  template<int BLOCKX, int BLOCKY, typename TMPLC>
    ONIKA_DEVICE_KERNEL_FUNC void get_number_of_interations_block(
        TMPLC cells,
        IJK dims,
        GridChunkNeighborsData nbh,
        shapes shps,
        double rVerlet,
        NumberOfInteractionPerTypes * count_data,
        size_t* cell_idx)
    {
      using BlockReduce = cub::BlockReduce<int, BLOCKX, cub::BLOCK_REDUCE_RAKING, BLOCKY>; // 8*8 blockDimXY>;
      const size_t cell_a = cell_idx[ONIKA_CU_BLOCK_IDX];
      IJK loc_a = grid_index_to_ijk( dims, cell_a);

      // cub stuff
      ONIKA_CU_BLOCK_SHARED typename BlockReduce::TempStorage temp_storage;

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

      // get fields
      cell_accessors cellA(cells[cell_a]);

      const int32_t poffshift = stream_info.shift;

      for(unsigned int p_a=0; p_a<cell_a_particles ; p_a++)
      {
        if( particle_offset!=nullptr ) stream = stream_base + particle_offset[p_a] + poffshift;

        unsigned int cell_groups = *(stream++); // number of cell groups for this neighbor list

        /** load data */
        particle_info p(shps, p_a, cellA);

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
          cell_accessors cellB(cells[nbh_cg.cell_b]);
          for(unsigned int chunk=0 ; chunk<nbh_cg.nchunks ; chunk++)
          {
            unsigned int p_b = nbh_cg.chunk_idx[chunk];
            if( p_b<nbh_cell_particles && (nbh_cg.cell_b!=cell_a || p_b!=p_a) )
            {
              particle_info p_nbh(shps, p_b, cellB);
              if( intersect(rVerlet, obb_i, p_nbh))
              {
                count_interaction_block( rVerlet, count, !nbh_cg.is_ghost_b, p, p_nbh);
              }
            }
          }
        }
      }
      for(int i = 0; i < NumberOfInteractionTypes ; i++)
      {
        int aggregate = BlockReduce(temp_storage).Sum(count[i]);
        ONIKA_CU_BLOCK_SYNC();
        if(ONIKA_CU_THREAD_IDX == 0 && threadIdx.y == 0 && threadIdx.z == 0) count_data[ONIKA_CU_BLOCK_IDX][i] = aggregate;
      }
    }



  template<int BLOCKX, int BLOCKY, typename TMPLC>
    ONIKA_DEVICE_KERNEL_FUNC void fill_classifier_block(
        InteractionSOA* data,
        TMPLC cells,
        IJK dims,
        GridChunkNeighborsData nbh,
        shapes shps,
        double rVerlet,
        NumberOfInteractionPerTypes * shift_data,
        size_t* cell_idx)
    {
      using BlockScan = cub::BlockScan<int, BLOCKX, cub::BLOCK_SCAN_RAKING, BLOCKY>;
      const size_t cell_a = cell_idx[ONIKA_CU_BLOCK_IDX];
      IJK loc_a = grid_index_to_ijk( dims, cell_a);

      // cub stuff
      ONIKA_CU_BLOCK_SHARED typename BlockScan::TempStorage temp_storage;

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
 
      /** get fields from cell a */
      cell_accessors cellA(cells[cell_a]);

      for(unsigned int p_a=0; p_a< cell_a_particles ; p_a++)
      {
        if( particle_offset!=nullptr ) stream = stream_base + particle_offset[p_a] + poffshift;
        unsigned int cell_groups = *(stream++); // number of cell groups for this neighbor list

        /** load data */
        particle_info p(shps, p_a, cellA);

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
          
          /** get fields from cell b */
          cell_accessors cellB(cells[nbh_cg.cell_b]);

          for(unsigned int chunk=0 ; chunk<nbh_cg.nchunks ; chunk++)
          {
            unsigned int p_b = nbh_cg.chunk_idx[chunk];
            if( p_b<nbh_cell_particles && (nbh_cg.cell_b!=cell_a || p_b!=p_a) )
            {
              particle_info p_nbh(shps, p_b, cellB);
              if( intersect(rVerlet, obb_i, p_nbh) )
              {
                count_interaction_block( rVerlet, count, !nbh_cg.is_ghost_b, p, p_nbh);
              }
            }
          }
        }
      }
      ONIKA_CU_BLOCK_SYNC();

      NumberOfInteractionPerTypes sdata = shift_data[ONIKA_CU_BLOCK_IDX];
      for(int type = 0 ; type < NumberOfInteractionTypes ; type++)
      {
        BlockScan(temp_storage).ExclusiveSum(count[type], prefix[type]);
        ONIKA_CU_BLOCK_SYNC();
        prefix[type] += sdata[type];
      }
      Interaction item;
      for(unsigned int p_a=0 ; p_a< cell_a_particles ; p_a++)
      {
        if( particle_offset!=nullptr ) stream = stream_base + particle_offset[p_a] + poffshift;
        unsigned int cell_groups = *(stream++); // number of cell groups for this neighbor list

        /** Get current particle info */
        particle_info p(shps, p_a, cellA);

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

          /** Get fields from cell B */
          cell_accessors cellB(cells[nbh_cg.cell_b]);
          for(unsigned int chunk=0 ; chunk<nbh_cg.nchunks ;chunk++)
          {
            unsigned int p_b = nbh_cg.chunk_idx[chunk];
            if( p_b<nbh_cell_particles && (nbh_cg.cell_b!=cell_a || p_b!=p_a) )
            {
              /** Get nbh particle info */
              particle_info p_nbh(shps, p_b, cellB);
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
                fill_interaction_block( data, item, rVerlet, prefix, !nbh_cg.is_ghost_b, p, p_nbh);
              } // check
            } // p_b
          } // chunk
        } // cg
      } // p_a 
    } // fill 
} // namespace exaDEM

