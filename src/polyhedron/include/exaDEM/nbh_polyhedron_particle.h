namespace exaDEM
{
  using namespace exanb;
  using VertexArray = ::onika::oarray_t<::exanb::Vec3d, EXADEM_MAX_VERTICES>;
  using NumberOfPolyhedronInteractionPerTypes = ::onika::oarray_t<int, NumberOfPolyhedronInteractionTypes>;

/***************************/
/*  Device Block functions */
/***************************/

  ONIKA_HOST_DEVICE_FUNC void count_interactions(
      double rVerlet,
      int count[],
      bool is_not_ghost_b,
      particle_info& p,
      particle_info& p_nbh,
      OBB& obb_i,
      OBB& obb_j)
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

/*
    obb_j.enlarge(shp->m_radius);
    obb_i.enlarge(shp_nbh->m_radius);
*/

		for (int i = 0; i < nv; i++)
		{
			vec3r vi = conv_to_vec3r(vertices_a[i]);
			if(obb_j.intersect(vi))
			{
				for (int j = 0 ; j < nv_nbh; j++)
				{
					if (exaDEM::filter_vertex_vertex(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh))
					{
						count[0]++; // vertex-vertex
					}
				}
				for (int j = 0; j < ne_nbh; j++)
				{
					bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
					count[1] += contact * 1; // vertex - edge
				}
				for (int j = 0; j < nf_nbh; j++)
				{
					bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
					count[2] += contact * 1; // vertex - face
				}
			}
		}


		// edge edge
		for (int i = 0; i < ne; i++)
		{
			for (int j = 0; j < ne_nbh; j++)
			{
				bool contact = exaDEM::filter_edge_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
				count[3] += contact * 1; // edge - edge
			}
		}

		// interaction of from particle j to particle i
		for (int j = 0; j < nv_nbh; j++)
		{
			vec3r vj = conv_to_vec3r(vertices_b[j]);
			if( obb_i.intersect(vj))
			{

				for (int i = 0; i < ne; i++)
				{
					bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
					count[1] += contact * 1; // edge - vertex
				}

				for (int i = 0; i < nf; i++)
				{
					bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
					count[2] += contact * 1; // face - vertex
				}
			}
		}

/*
		obb_j.enlarge(-shp->m_radius);
		obb_i.enlarge(-shp_nbh->m_radius);
*/
	}


	ONIKA_HOST_DEVICE_FUNC void fill_interactions(
			InteractionSOA* data,
			exaDEM::Interaction& item,
			double rVerlet,
			int prefix[],
			bool is_not_ghost_b,
			particle_info& p,
			particle_info& p_nbh,
      OBB& obb_i,
      OBB& obb_j)
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

/*
		obb_j.enlarge(shp->m_radius);
		obb_i.enlarge(shp_nbh->m_radius);
*/
		for (int i = 0; i < nv; i++)
		{
			vec3r vi = conv_to_vec3r(vertices_a[i]);
			if(obb_j.intersect(vi))
			{
				item.sub_i = i;
				item.type = 0;
				for (int j = 0 ; j < nv_nbh; j++)
				{
					if (exaDEM::filter_vertex_vertex(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh))
					{
						item.sub_j = j;
						data[item.type].set(prefix[item.type]++, item);
					}
				}

				item.type = 1;
				// vertex - edge
				for (int j = 0; j < ne_nbh; j++)
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
				for (int j = 0; j < nf_nbh; j++)
				{
					bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
					if (contact)
					{
						item.sub_j = j;
						data[item.type].set(prefix[item.type]++, item);
					}
				}
			}
		}

		item.type = 3;
		for (int i = 0; i < ne; i++)
		{
			item.sub_i = i;
			// edge - edge
			for (int j = 0; j < ne_nbh; j++)
			{
				bool contact = exaDEM::filter_edge_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
				if (contact)
				{
					item.sub_j = j;
					data[item.type].set(prefix[item.type]++, item);
				}
			}
		}


    std::swap(item.cell_j, item.cell_i);
    std::swap(item.id_j, item.id_i);
    std::swap(item.p_j, item.p_i);

		// interaction of from particle j to particle i
		for (int j = 0; j < nv_nbh; j++)
		{
			vec3r vj = conv_to_vec3r(vertices_b[j]);
			if( obb_i.intersect(vj))
			{
				item.type = 1;
				item.sub_i = j;
				// edge - vertex
				for (int i = 0; i < ne; i++)
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
				for (int i = 0; i < nf; i++)
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
/*
		obb_j.enlarge(-shp->m_radius);
		obb_i.enlarge(-shp_nbh->m_radius);
*/
	}

	/***************************/
	/*  Global Block functions */
	/***************************/

	template<int BLOCKX, typename TMPLC>
		ONIKA_DEVICE_KERNEL_FUNC void get_number_of_interations(
				TMPLC cells,
				IJK dims,
				GridChunkNeighborsData nbh,
				shapes shps,
				double rVerlet,
				NumberOfPolyhedronInteractionPerTypes * count_data,
				size_t* cell_idx)
		{
			using BlockReduce = cub::BlockReduce<int, BLOCKX, cub::BLOCK_REDUCE_RAKING>; // 8*8 blockDimXY>;
			const size_t cell_a = cell_idx[ONIKA_CU_BLOCK_IDX];
			IJK loc_a = grid_index_to_ijk( dims, cell_a);

			assert(ONIKA_CU_BLOCK_SIZE == BLOCKX);
			// cub stuff
			ONIKA_CU_BLOCK_SHARED typename BlockReduce::TempStorage temp_storage;

			// Struct to fill count_data at the enf
			int count[NumberOfPolyhedronInteractionTypes];
			for(size_t i = 0; i < NumberOfPolyhedronInteractionTypes ; i++)
			{
				count[i] = 0;
			}

			const unsigned int cell_a_particles = cells[cell_a].size();
			const auto stream_info = chunknbh_stream_info( nbh[cell_a] , cell_a_particles );
			const uint16_t* stream_base = stream_info.stream;
			const uint16_t* stream = stream_base;
			const uint32_t* __restrict__ particle_offset = stream_info.offset;

			const int32_t poffshift = stream_info.shift;

			for(unsigned int p_a= ONIKA_CU_THREAD_IDX; p_a<cell_a_particles ; p_a+= ONIKA_CU_BLOCK_SIZE)
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
					for(unsigned int chunk=0 ; chunk<nbh_cg.nchunks ; chunk++)
					{
						unsigned int p_b = nbh_cg.chunk_idx[chunk];
						if( p_b<nbh_cell_particles && (nbh_cg.cell_b!=cell_a || p_b!=p_a) )
						{
							particle_info p_nbh(cells, shps, nbh_cg.cell_b, p_b);
              OBB obb_j = p_nbh.shp->obb;
              obb_j.rotate(p_nbh.get_quat());
              obb_j.translate(vec3r{p_nbh.r.x, p_nbh.r.y, p_nbh.r.z});
              if( obb_i.intersect(obb_j) )
              {
                obb_j.enlarge(rVerlet);
								count_interactions( rVerlet, count, !nbh_cg.is_ghost_b, p, p_nbh, obb_i, obb_j);
							}
						}
					}
				}
			}
			for(int i = 0; i < NumberOfPolyhedronInteractionTypes ; i++)
			{
				int aggregate = BlockReduce(temp_storage).Sum(count[i]);
				ONIKA_CU_BLOCK_SYNC();
				if(ONIKA_CU_THREAD_IDX == 0) count_data[ONIKA_CU_BLOCK_IDX][i] = aggregate;
			}
		}



	template<int BLOCKX, typename TMPLC>
		ONIKA_DEVICE_KERNEL_FUNC void fill_classifier(
				InteractionSOA* data,
				TMPLC cells,
				IJK dims,
				GridChunkNeighborsData nbh,
				shapes shps,
				double rVerlet,
				NumberOfPolyhedronInteractionPerTypes * shift_data,
				size_t* cell_idx)
		{
			assert(ONIKA_CU_BLOCK_SIZE == BLOCKX);
			using BlockScan = cub::BlockScan<int, BLOCKX, cub::BLOCK_SCAN_RAKING>;
			const size_t cell_a = cell_idx[ONIKA_CU_BLOCK_IDX];
			IJK loc_a = grid_index_to_ijk( dims, cell_a);

			// cub stuff
			ONIKA_CU_BLOCK_SHARED typename BlockScan::TempStorage temp_storage;

			// Struct to fill count_data at the enf
			int count[NumberOfPolyhedronInteractionTypes];
			int prefix[NumberOfPolyhedronInteractionTypes];
			for(size_t i = 0; i < NumberOfPolyhedronInteractionTypes ; i++)
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
			for(unsigned int p_a=ONIKA_CU_THREAD_IDX; p_a< cell_a_particles ; p_a+=ONIKA_CU_BLOCK_SIZE)
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
					for(unsigned int chunk=0 ; chunk<nbh_cg.nchunks ; chunk++)
					{
						unsigned int p_b = nbh_cg.chunk_idx[chunk];
						if( p_b<nbh_cell_particles && (nbh_cg.cell_b!=cell_a || p_b!=p_a) )
						{
							particle_info p_nbh(cells, shps, nbh_cg.cell_b, p_b);
              OBB obb_j = p_nbh.shp->obb;
              obb_j.rotate(p_nbh.get_quat());
              obb_j.translate(vec3r{p_nbh.r.x, p_nbh.r.y, p_nbh.r.z});
              if( obb_i.intersect(obb_j) )
              {
                obb_j.enlarge(rVerlet);
								count_interactions( rVerlet, count, !nbh_cg.is_ghost_b, p, p_nbh, obb_i, obb_j);
							}
						}
					}
				}
			}
			ONIKA_CU_BLOCK_SYNC();

			NumberOfPolyhedronInteractionPerTypes& sdata = shift_data[ONIKA_CU_BLOCK_IDX];
			for(int type = 0 ; type < NumberOfPolyhedronInteractionTypes ; type++)
			{
				BlockScan(temp_storage).ExclusiveSum(count[type], prefix[type]);
				ONIKA_CU_BLOCK_SYNC();
				prefix[type] += sdata[type];
			}
			Interaction item;
			for(unsigned int p_a=ONIKA_CU_THREAD_IDX; p_a< cell_a_particles ; p_a+=ONIKA_CU_BLOCK_SIZE)
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
					for(unsigned int chunk=0 ; chunk<nbh_cg.nchunks ;chunk++)
					{
						unsigned int p_b = nbh_cg.chunk_idx[chunk];
						if( p_b<nbh_cell_particles && (nbh_cg.cell_b!=cell_a || p_b!=p_a) )
						{
							/** Get nbh particle info */
							particle_info p_nbh(cells, shps, nbh_cg.cell_b, p_b);
              OBB obb_j = p_nbh.shp->obb;
              obb_j.rotate(p_nbh.get_quat());
              obb_j.translate(vec3r{p_nbh.r.x, p_nbh.r.y, p_nbh.r.z});
              if( obb_i.intersect(obb_j) )
              {
                obb_j.enlarge(rVerlet);

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
								fill_interactions( data, item, rVerlet, prefix, !nbh_cg.is_ghost_b, p, p_nbh, obb_i, obb_j);
							} // check
						} // p_b
					} // chunk
				} // cg
			} // p_a 
		} // fill 
} // namespace exaDEM

