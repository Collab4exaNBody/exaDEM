namespace exaDEM
{
	using namespace exanb;
	using VertexArray = ::onika::oarray_t<::exanb::Vec3d, EXADEM_MAX_VERTICES>;
	using NumberOfInteractionPerTypes = ::onika::oarray_t<int, NumberOfInteractionTypes>;

	struct particle_info
	{
		uint64_t id;
		Vec3d r;
		Quaternion quaternion;
		VerticesType vertices; 
		const shape *shp;

		template<typename Cells>
			__host__ __device__ particle_info(Cells& cells, shapes& shps, int cell_id, int p_id)
			{
				auto& cell = cells[cell_id];
				const uint32_t type = cell[field::type][p_id];
				double rx =  cell[field::rx][p_id];
				double ry =  cell[field::ry][p_id];
				double rz =  cell[field::rz][p_id];
				id =  cell[field::id][p_id];
				r = {rx, ry, rz};
				quaternion = cell[field::orient][p_id];
				vertices = cell[field::vertices][p_id];
				shp = shps[type];
			}

		__host__ __device__ quat get_quat()
		{
			return quat{vec3r{quaternion.x, quaternion.y, quaternion.z}, quaternion.w};
		} 
	};	

	__host__ __device__ bool filter_obb(double rVerlet, OBB& obb_i, particle_info& p_nbh)
	{
		// Get particle pointers for the particle b.
		OBB obb_j = p_nbh.shp->obb;
		quat conv_orient_j = p_nbh.get_quat();
		obb_j.rotate(conv_orient_j);
		obb_j.translate(vec3r{p_nbh.r.x, p_nbh.r.y, p_nbh.r.z});
		obb_j.enlarge(rVerlet);

		return !obb_i.intersect(obb_j);
	}

	__device__ void count_interaction(
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

		for (int i = threadIdx.x; i < nv; i+= blockDim.x)
		{
			for (int j = threadIdx.y ; j < nv_nbh; j+= blockDim.y)
			{
				if (exaDEM::filter_vertex_vertex(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh))
				{
					count[0]++; // vertex-vertex
				}
			}

			for (int j = threadIdx.y; j < ne_nbh; j+= blockDim.y)
			{
				bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
				if (contact) count[1]++; // vertex - edge
			}
			for (int j = threadIdx.y; j < nf_nbh; j+=blockDim.y)
			{
				bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
				if (contact) count[2]++; // vertex - face
			}
		}
		for (int i = threadIdx.x; i < ne; i+= blockDim.x)
		{
			for (int j = threadIdx.y; j < ne_nbh; j+= blockDim.y)
			{
				bool contact = exaDEM::filter_edge_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
				if (contact) count[3]++; // edge - edge
			}
		}

		// interaction of from particle j to particle i
		for (int j = threadIdx.x; j < nv_nbh; j+= blockDim.x)
		{
			auto& vj = vertices_b[j];//shp->get_vertex(j, r_nbh, orient_nbh);
			for (int i = threadIdx.y; i < ne; i+=blockDim.y)
			{
				bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
				if (contact) count[1]++; // edge - vertex
			}

			for (int i = threadIdx.y; i < nf; i+=blockDim.y)
			{
				bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
				if (contact) count[2]++; // face - vertex
			}
		}
	}

	__device__ void fill_interaction(
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

		for (int i = threadIdx.x; i < nv; i+= blockDim.x)
		{
			item.sub_i = i;
			item.type = 0;
			for (int j = threadIdx.y ; j < nv_nbh; j+= blockDim.y)
			{
				if (exaDEM::filter_vertex_vertex(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh))
				{
					item.sub_j = j;
					data[0].set(prefix[0]++, item);
				}
			}

			item.type = 1;
			for (int j = threadIdx.y; j < ne_nbh; j+= blockDim.y)
			{
				bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
				if (contact) 
				{ 
					item.sub_j = j;
					data[1].set(prefix[1]++, item); // vertex - edge
				}
			}
			item.type = 2;
			for (int j = threadIdx.y; j < nf_nbh; j+=blockDim.y)
			{
				bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
				if (contact) 
				{ 
					item.sub_j = j;
					data[2].set(prefix[2]++, item); // vertex - face
				}
			}
		}

		item.type = 3;
		for (int i = threadIdx.x; i < ne; i+= blockDim.x)
		{
			item.sub_i = i;
			for (int j = threadIdx.y; j < ne_nbh; j+= blockDim.y)
			{
				bool contact = exaDEM::filter_edge_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
				if (contact)
				{
					item.sub_j = j;
					data[3].set(prefix[3]++, item); // edge - edge
				}
			}
		}

		swap(item.cell_j, item.cell_i);
		swap(item.p_j, item.p_i);
		swap(item.id_j, item.id_i);

		// interaction of from particle j to particle i
		for (int j = threadIdx.x; j < nv_nbh; j+= blockDim.x)
		{
			item.type = 1;
			item.sub_i = j;
			auto& vj = vertices_b[j];//shp->get_vertex(j, r_nbh, orient_nbh);
			for (int i = threadIdx.y; i < ne; i+=blockDim.y)
			{
				bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
				if (contact)
				{
					item.sub_j = i;
					data[1].set(prefix[1]++, item); // edge - vertex
				}
			}

			item.type = 2;
			for (int i = threadIdx.y; i < nf; i+=blockDim.y)
			{
				bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
				if (contact)
				{
					item.sub_j = i;
					data[2].set(prefix[2]++, item);// face - vertex
				}
			}
		}
	}

	template<typename TMPLC>
		__global__ void fill_classifier_gpu(
				InteractionSOA* data,
				TMPLC cells,
				IJK dims,
				GridChunkNeighborsData nbh,
				shapes shps,
				double rVerlet,
				NumberOfInteractionPerTypes * shift_data,
				NumberOfInteractionPerTypes * count_data,
				size_t* cell_idx)
		{
			constexpr int blockDimXY = 64;//blockDim.x * blockDim.y; 
			constexpr int CS = 1; // chunk size
			using BlockScan = cub::BlockScan<int, 64>;
			const int threadId = blockDim.x * threadIdx.x + threadIdx.y;
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

			// for obbs
			__shared__ int check_obb[64]; // 8*8
			check_obb[threadId] = false;

			const unsigned int cell_a_particles = cells[cell_a].size();
			const auto stream_info = chunknbh_stream_info( nbh[cell_a] , cell_a_particles );
			const uint16_t* stream_base = stream_info.stream;
			const uint16_t* __restrict__ stream = stream_base;
			const uint32_t* __restrict__ particle_offset = stream_info.offset;

			if( particle_offset == nullptr ) return;

			const int32_t poffshift = stream_info.shift;

			for(unsigned int p_a=0; p_a<cell_a_particles ; p_a++)
			{
				size_t p_nbh_index = 0;
				unsigned int cell_groups = *(stream++); // number of cell groups for this neighbor list
				size_t cell_b = cell_a;
				unsigned int chunk = 0;
				unsigned int nchunks = 0;
				unsigned int cg = 0; // cell group index.
				bool symcont = false;

				/** load data */
				particle_info p(cells, shps, cell_a, p_a);

				/** compute obb */
				OBB obb_i = p.shp->obb;
				quat conv_orient_i = p.get_quat();
				obb_i.rotate(conv_orient_i);
				obb_i.translate(vec3r{p.r.x, p.r.y, p.r.z});
				obb_i.enlarge(rVerlet);

				auto stream_checkpoint = stream;

				for(cg=0; cg<cell_groups && symcont ;cg++)
				{
					uint16_t cell_b_enc = *(stream++);
					IJK loc_b = loc_a + decode_cell_index(cell_b_enc);
					cell_b = grid_ijk_to_index( dims , loc_b );
					unsigned int nbh_cell_particles = cells[cell_b].size();
					nchunks = *(stream++); // should be 1
					for(chunk=threadId;chunk<nchunks && symcont;chunk+= blockDim.x * blockDim.y)
					{
						unsigned int chunk_start = static_cast<unsigned int>( *(stream++) ) * CS;
						for(unsigned int i=0;i<CS && symcont;i++)
						{
							unsigned int p_b = chunk_start + i;
							if( p_b<nbh_cell_particles && (cell_b!=cell_a || p_b!=p_a) )
							{
								particle_info p_nbh(cells, shps, cell_b, p_b);
								check_obb[chunk] = check_obb[chunk] || filter_obb(rVerlet, obb_i, p_nbh);
							}
						}
					}
				}
				__syncthreads();
				stream = stream_checkpoint;

				for(cg=0; cg<cell_groups && symcont ;cg++)
				{
					uint16_t cell_b_enc = *(stream++);
					IJK loc_b = loc_a + decode_cell_index(cell_b_enc);
					cell_b = grid_ijk_to_index( dims , loc_b );
					unsigned int nbh_cell_particles = cells[cell_b].size();
					nchunks = *(stream++); // should be 1

					bool is_ghost_b = inside_grid_shell(dims, 0, 1, loc_b); //grid.is_ghost_cell(cell_b);
					for(chunk=0;chunk<nchunks && symcont;chunk++)
					{
						unsigned int chunk_start = static_cast<unsigned int>( *(stream++) ) * CS;
						for(unsigned int i=0;i<CS && symcont;i++)
						{
							unsigned int p_b = chunk_start + i;
							if( p_b<nbh_cell_particles && (cell_b!=cell_a || p_b!=p_a) )
							{
								if( check_obb[p_b] );
								{
									particle_info p_nbh(cells, shps, cell_b, p_b);
									count_interaction( rVerlet, count, !is_ghost_b, p, p_nbh);
								}
							}
						}
					}
				}
				__syncthreads();
				BlockScan(temp_storage).ExclusiveSum(count, prefix);
				auto& sdata = shift_data[blockIdx.x];
				for(int type = 0 ; type < NumberOfInteractionTypes ; type++)
				{
					prefix[type] += sdata[type];
				}

				stream = stream_checkpoint;
				Interaction item;
				item.id_i = p.id;
				item.cell_i = cell_a;

				for(cg=0; cg<cell_groups && symcont ;cg++)
				{
					uint16_t cell_b_enc = *(stream++);
					IJK loc_b = loc_a + decode_cell_index(cell_b_enc);
					cell_b = grid_ijk_to_index( dims , loc_b );
					unsigned int nbh_cell_particles = cells[cell_b].size();
					nchunks = *(stream++); // should be 1

					item.cell_j = cell_b;
					bool is_ghost_b = inside_grid_shell(dims, 0, 1, loc_b); //grid.is_ghost_cell(cell_b);
					for(chunk=0;chunk<nchunks && symcont;chunk++)
					{
						unsigned int chunk_start = static_cast<unsigned int>( *(stream++) ) * CS;
						for(unsigned int i=0;i<CS && symcont;i++)
						{
							unsigned int p_b = chunk_start + i;
							if( p_b<nbh_cell_particles && (cell_b!=cell_a || p_b!=p_a) )
							{
								if( check_obb[p_b] );
								{
									particle_info p_nbh(cells, shps, cell_b, p_b);
									item.id_j = p_b;
									fill_interaction( data, item, rVerlet, prefix, !is_ghost_b, p, p_nbh);
								} // check
							} // p_b
						} // CS
					} // chunk
				} // cg
			} // p_a 
		} // fill 
} // namespace exaDEM
