namespace exaDEM
{
	using namespace exanb;
	using VertexArray = ::onika::oarray_t<::exanb::Vec3d, EXADEM_MAX_VERTICES>;
	using NumberOfInteractionPerTypes = ::onika::oarray_t<int, NumberOfInteractionTypes>;

	struct particle_info
	{
		uint64_t id;
		Vec3d r;
		Quaternion& quaternion;
		VerticesType& vertices; 
		const shape *shp;

		template<typename Cells>
			__host__ __device__ particle_info(Cells& cells, shapes& shps, int cell_id, int p_id) : 
        quaternion(cells[cell_id][field::orient][p_id]),
        vertices(cells[cell_id][field::vertices][p_id])
			{
				auto& cell = cells[cell_id];
				const uint32_t type = cell[field::type][p_id];
				double rx =  cell[field::rx][p_id];
				double ry =  cell[field::ry][p_id];
				double rz =  cell[field::rz][p_id];
				id =  cell[field::id][p_id];
				r = {rx, ry, rz};
				shp = shps[type];
			}

		__host__ __device__ quat get_quat()
		{
			return quat{vec3r{quaternion.x, quaternion.y, quaternion.z}, quaternion.w};
		} 
	};	

	__host__ __device__ bool intersect(double rVerlet, OBB& obb_i, particle_info& p_nbh)
	{
		// Get particle pointers for the particle b.
		OBB obb_j = p_nbh.shp->obb;
		quat conv_orient_j = p_nbh.get_quat();
		obb_j.rotate(conv_orient_j);
		obb_j.translate(vec3r{p_nbh.r.x, p_nbh.r.y, p_nbh.r.z});
		obb_j.enlarge(rVerlet);

		return obb_i.intersect(obb_j);
	}

	struct header_nbh
	{
		uint16_t nchunks;
		int cell_b;
		const uint16_t* chunk_idx;
		bool is_ghost_b;
	};

	// stream is shifted
	__host__ __device__ 
		header_nbh decode_stream_header_nbh(const IJK& loc_a, const IJK dims, const uint16_t*& stream)
		{
			header_nbh res;
			uint16_t cell_b_enc = *(stream++);
			IJK loc_b = loc_a + decode_cell_index(cell_b_enc);
			res.cell_b =  grid_ijk_to_index( dims , loc_b );
			res.nchunks = *(stream++);
			res.chunk_idx = stream;
			res.is_ghost_b = inside_grid_shell(dims, 0, 1, loc_b);
			stream += res.nchunks; /*do not forget to shift stream*/
			return res; 
		}

  // one block
  __global__ void stupid_prefix_sum(size_t size, NumberOfInteractionPerTypes * count_data, NumberOfInteractionPerTypes * prefix_data)
  {
    if(threadIdx.x < NumberOfInteractionTypes)
    {
      prefix_data[0][threadIdx.x] = 0;
      for(int i = 1 ; i < size ; i++)
      {
        prefix_data[i][threadIdx.x] = prefix_data[i-1][threadIdx.x] + count_data[i-1][threadIdx.x];
      }

    }
  }
} // namespace exaDEM
