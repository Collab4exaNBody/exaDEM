
namespace exaDEM
{
	using namespace exanb;
	//using VertexArray = ::onika::oarray_t<::exanb::Vec3d, EXADEM_MAX_VERTICES>;
	using NumberOfPolyhedronInteractionPerTypes = ::onika::oarray_t<int, NumberOfPolyhedronInteractionTypes>;
	

using TypeCountArray = NumberOfPolyhedronInteractionPerTypes; // alias, ex: int[4]
constexpr int NumTypes = NumberOfPolyhedronInteractionTypes;

__global__ void extract_column(const TypeCountArray* src, int* dest, size_t size, int col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) dest[i] = src[i][col];
}

__global__ void insert_column(TypeCountArray* dst, const int* src, size_t size, int col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) dst[i][col] = src[i];
}

void scan_per_type_with_cub(size_t size,
                            const TypeCountArray* d_input,
                            TypeCountArray* d_output,
                            void* d_temp_storage,
                            size_t& temp_storage_bytes)
{
    for (int t = 0; t < NumTypes; ++t)
    {
        // Créer des vues vers chaque colonne (type t)
        int* in_col  = nullptr;
        int* out_col = nullptr;

        // Allouer des buffers temporaires d'index
        cudaMalloc(&in_col,  size * sizeof(int));
        cudaMalloc(&out_col, size * sizeof(int));

        // Copier la colonne t dans in_col
        extract_column<<<(size+255)/256, 256>>>(d_input, in_col, size, t);

        // Calculer la taille mémoire si besoin
        if (d_temp_storage == nullptr)
        {
            cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, in_col, out_col, size);
            continue;
        }

        // Faire le scan CUB
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in_col, out_col, size);

        // Copier le résultat dans la colonne t de d_output
        insert_column<<<(size+255)/256, 256>>>(d_output, out_col, size, t);

        // Nettoyage
        cudaFree(in_col); 	
        cudaFree(out_col);
    }
}

	/***************************/
	/*  Device Block functions */
	/***************************/
	
	
	template<typename VecI, typename VecJ> ONIKA_HOST_DEVICE_FUNC void count_interaction_block_pair(
			double rVerlet,
			int count[],
			particle_info& p,
			const VecI& vertices_a,
			particle_info& p_nbh,
			const VecJ& vertices_b
			)
	{
		// default value of the interaction studied (A or i -> B or j)
		//if (p.id >= p_nbh.id)
		//{
		//	if (!is_not_ghost_b)
		//		return;
		//}

		/** some renames */
		auto& shp = p.shp;
		//auto& vertices_a = p.vertices;
		auto& shp_nbh = p_nbh.shp;
		//auto& vertices_b = p_nbh.vertices;

		// get particle j data.
		const int nv = shp->get_number_of_vertices();
		const int ne = shp->get_number_of_edges();
		const int nf = shp->get_number_of_faces();
		const int nv_nbh = shp_nbh->get_number_of_vertices();
		const int ne_nbh = shp_nbh->get_number_of_edges();
		const int nf_nbh = shp_nbh->get_number_of_faces();

		//ONIKA_CU_BLOCK_Y_SIMD_FOR(int, i, 0, nv)
		for(int i = threadIdx.y; i < nv; i+= blockDim.y)
		{
			vec3r vi = conv_to_vec3r(vertices_a[i]);

				//ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, nv_nbh)
				for(int j = threadIdx.x; j < nv_nbh; j+= blockDim.x)
				{
					if (exaDEM::filter_vertex_vertex(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh))
					{
						count[VERTEX_VERTEX]++; // vertex-vertex
					}
				}
				//ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, ne_nbh)
				for(int j = threadIdx.x; j < ne_nbh; j+= blockDim.x)
				{
					bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
					count[VERTEX_EDGE] += contact * 1; // vertex - edge
				}
				//ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, nf_nbh)
				for(int j = threadIdx.x; j < nf_nbh; j+= blockDim.x)
				{
					bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
					count[VERTEX_FACE] += contact * 1; // vertex - face
				}
		}

		//ONIKA_CU_BLOCK_Y_SIMD_FOR(int, i, 0, ne)
		for(int i = threadIdx.y; i < ne; i+= blockDim.y)
		{
			//ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, ne_nbh)
			for(int j = threadIdx.x; j < ne_nbh; j+= blockDim.x)
			{
				bool contact = exaDEM::filter_edge_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
				count[EDGE_EDGE] += contact * 1; // edge - edge
			}
		}

		// interaction of from particle j to particle i
		//ONIKA_CU_BLOCK_Y_SIMD_FOR(int, j, 0, nv_nbh)
		for(int j = threadIdx.y; j < nv_nbh; j+= blockDim.y)
		{
			vec3r vj = conv_to_vec3r(vertices_b[j]);

				//ONIKA_CU_BLOCK_SIMD_FOR(int, i, 0, ne)
				for(int i = threadIdx.x; i < ne; i+= blockDim.x)
				{
					bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
					count[VERTEX_EDGE] += contact * 1; // edge - vertex
				}

				//ONIKA_CU_BLOCK_SIMD_FOR(int, i, 0, nf)
				for(int i = threadIdx.x; i < nf; i+= blockDim.x)
				{
					bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
					count[VERTEX_FACE] += contact * 1; // face - vertex
				}
		}
	}
	
	template<typename VecI, typename VecJ> ONIKA_HOST_DEVICE_FUNC void count_interaction_block_pair2(
			double rVerlet,
			int count[],
			int& count1,
			int& count2,
			int& count3,
			int& count4,
			int& count5,
			int& count6,
			particle_info& p,
			const VecI& vertices_a,
			particle_info& p_nbh,
			const VecJ& vertices_b
			)
	{
		// default value of the interaction studied (A or i -> B or j)
		//if (p.id >= p_nbh.id)
		//{
		//	if (!is_not_ghost_b)
		//		return;
		//}

		/** some renames */
		auto& shp = p.shp;
		//auto& vertices_a = p.vertices;
		auto& shp_nbh = p_nbh.shp;
		//auto& vertices_b = p_nbh.vertices;

		// get particle j data.
		const int nv = shp->get_number_of_vertices();
		const int ne = shp->get_number_of_edges();
		const int nf = shp->get_number_of_faces();
		const int nv_nbh = shp_nbh->get_number_of_vertices();
		const int ne_nbh = shp_nbh->get_number_of_edges();
		const int nf_nbh = shp_nbh->get_number_of_faces();

		//ONIKA_CU_BLOCK_Y_SIMD_FOR(int, i, 0, nv)
		for(int i = threadIdx.y; i < nv; i+= blockDim.y)
		{
			vec3r vi = conv_to_vec3r(vertices_a[i]);

				//ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, nv_nbh)
				for(int j = threadIdx.x; j < nv_nbh; j+= blockDim.x)
				{
					if (exaDEM::filter_vertex_vertex(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh))
					{
						count[VERTEX_VERTEX]++; // vertex-vertex
						count1++;
					}
				}
				//ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, ne_nbh)
				for(int j = threadIdx.x; j < ne_nbh; j+= blockDim.x)
				{
					bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
					count[VERTEX_EDGE] += contact * 1; // vertex - edge
					count2+= contact * 1;
				}
				//ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, nf_nbh)
				for(int j = threadIdx.x; j < nf_nbh; j+= blockDim.x)
				{
					bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
					count[VERTEX_FACE] += contact * 1; // vertex - face
					count3+= contact * 1;
				}
		}

		//ONIKA_CU_BLOCK_Y_SIMD_FOR(int, i, 0, ne)
		for(int i = threadIdx.y; i < ne; i+= blockDim.y)
		{
			//ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, ne_nbh)
			for(int j = threadIdx.x; j < ne_nbh; j+= blockDim.x)
			{
				bool contact = exaDEM::filter_edge_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
				count[EDGE_EDGE] += contact * 1; // edge - edge
				count4+= contact * 1;
			}
		}

		// interaction of from particle j to particle i
		//ONIKA_CU_BLOCK_Y_SIMD_FOR(int, j, 0, nv_nbh)
		for(int j = threadIdx.y; j < nv_nbh; j+= blockDim.y)
		{
			vec3r vj = conv_to_vec3r(vertices_b[j]);

				//ONIKA_CU_BLOCK_SIMD_FOR(int, i, 0, ne)
				for(int i = threadIdx.x; i < ne; i+= blockDim.x)
				{
					bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
					count[VERTEX_EDGE] += contact * 1; // edge - vertex
					count5+= contact * 1;
				}

				//ONIKA_CU_BLOCK_SIMD_FOR(int, i, 0, nf)
				for(int i = threadIdx.x; i < nf; i+= blockDim.x)
				{
					bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
					count[VERTEX_FACE] += contact * 1; // face - vertex
					count6+= contact * 1;
				}
		}
	}



	template<typename VecI, typename VecJ> ONIKA_HOST_DEVICE_FUNC void fill_interaction_block(
			InteractionSOA2* data,
			//InteractionSOA* data,
			exaDEM::Interaction& item,
			double rVerlet,
			int prefix[],
			int& count1,
			int& count2,
			int& count3,
			int& count4,
			int& count5,
			int& count6,
			//bool is_not_ghost_b,
			particle_info& p,
			const VecI& vertices_a,
			particle_info& p_nbh,
			const VecJ& vertices_b//,
			//OBB& obb_i,
			//OBB& obb_j
			)
	{
		// default value of the interaction studied (A or i -> B or j)
		/*if (p.id >= p_nbh.id)
		{
			if (!is_not_ghost_b)
				return;
		}*/

		/** some renames */
		auto& shp        = p.shp;
		//auto& vertices_a = p.vertices;
		auto& shp_nbh    = p_nbh.shp;
		//auto& vertices_b = p_nbh.vertices;

		// get particle j data.
		const int nv = shp->get_number_of_vertices();
		const int ne = shp->get_number_of_edges();
		const int nf = shp->get_number_of_faces();
		const int nv_nbh = shp_nbh->get_number_of_vertices();
		const int ne_nbh = shp_nbh->get_number_of_edges();
		const int nf_nbh = shp_nbh->get_number_of_faces();

		//obb_j.enlarge(shp->m_radius);
		//obb_i.enlarge(shp_nbh->m_radius);

		//ONIKA_CU_BLOCK_Y_SIMD_FOR(int, i, 0, nv)
		if(count1 > 0 || count2 > 0 || count3 > 0)
		{
		for(int i = threadIdx.y; i < nv; i+= blockDim.y)
		{
			vec3r vi = conv_to_vec3r(vertices_a[i]);
			//if(obb_j.intersect(vi))
			//{
				item.sub_i = i;
				item.type = 0;
				//ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, nv_nbh)
				if(count1 > 0)
				{
				for(int j = threadIdx.x; j < nv_nbh; j+= blockDim.x)
				{
					if (exaDEM::filter_vertex_vertex(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh))
					{
						item.sub_j = j;
						data[item.type].set(prefix[item.type]++, item);
					}
				}
				}

				item.type = 1;
				// vertex - edge
				//ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, ne_nbh)
				if(count2 > 0)
				{
				for(int j = threadIdx.x; j < ne_nbh; j+= blockDim.x)
				{
					bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
					if (contact)
					{
						item.sub_j = j;
						data[item.type].set(prefix[item.type]++, item);
					}
				}
				}
				item.type = 2;
				// vertex - face
				//ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, nf_nbh)
				if(count3 > 0)
				{
				for(int j = threadIdx.x; j < nf_nbh; j+= blockDim.x)
				{
					bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
					if (contact)
					{
						item.sub_j = j;
						data[item.type].set(prefix[item.type]++, item);
					}
				}
				}
			//}
		}
		}
		item.type = 3;
		//ONIKA_CU_BLOCK_Y_SIMD_FOR(int, i, 0, ne)
		if(count4 > 0)
		{
		for(int i = threadIdx.y; i < ne; i+= blockDim.y)
		{
			item.sub_i = i;
			// edge - edge
			//ONIKA_CU_BLOCK_SIMD_FOR(int, j, 0, ne_nbh)
			for(int j = threadIdx.x; j < ne_nbh; j+= blockDim.x)
			{
				bool contact = exaDEM::filter_edge_edge(rVerlet, vertices_a, i, shp, vertices_b, j, shp_nbh);
				if (contact)
				{
					item.sub_j = j;
					data[item.type].set(prefix[item.type]++, item);
				}
			}
		}
		}

		std::swap(item.cell_j, item.cell_i);
		std::swap(item.id_j, item.id_i);
		std::swap(item.p_j, item.p_i);

		// interaction of from particle j to particle i
		//ONIKA_CU_BLOCK_Y_SIMD_FOR(int, j, 0, nv_nbh)
		if(count5 > 0 || count6 > 0)
		{
		for(int j = threadIdx.y; j < nv_nbh; j+= blockDim.y)
		{
			vec3r vj = conv_to_vec3r(vertices_b[j]);
			//if( obb_i.intersect(vj))
			//{
				item.type = 1;
				item.sub_i = j;
				// edge - vertex
				//ONIKA_CU_BLOCK_SIMD_FOR(int, i, 0, ne)
				if(count5 > 0)
				{
				for(int i = threadIdx.x; i < ne; i+= blockDim.x)
				{
					bool contact = exaDEM::filter_vertex_edge(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
					if (contact)
					{
						item.sub_j = i;
						data[item.type].set(prefix[item.type]++, item);
					}
				}
				}
				item.type = 2;
				// face - vertex
				//ONIKA_CU_BLOCK_SIMD_FOR(int, i, 0, nf)
				if(count6 > 0)
				{
				for(int i = threadIdx.x; i < nf; i+= blockDim.x)
				{
					bool contact = exaDEM::filter_vertex_face(rVerlet, vertices_b, j, shp_nbh, vertices_a, i, shp);
					if (contact)
					{
						item.sub_j = i;
						data[item.type].set(prefix[item.type]++, item);
					}
				}
				}
			//}
		}
		}
		//obb_j.enlarge(-shp->m_radius);
		//obb_i.enlarge(-shp_nbh->m_radius);
	}


	/***************************/
	/*  Global Block functions */
	/***************************/


	template<int BLOCKX, int BLOCKY, typename TMPLC, typename TMPLV>
		ONIKA_DEVICE_KERNEL_FUNC void get_number_of_interactions_block_pair(
				TMPLC cells,
				TMPLV* const __restrict__ gv,
				IJK dims,
				shapes shps,
				double rVerlet,
				NumberOfPolyhedronInteractionPerTypes * count_data,
                                uint32_t* cell_i,
                                uint32_t* cell_j,
                                uint16_t* p_i,
                                uint16_t* p_j)			
		{
			using BlockReduce = cub::BlockReduce<int, BLOCKX, cub::BLOCK_REDUCE_RAKING, BLOCKY>; // 8*8 blockDimXY>;

			// cub stuff
			ONIKA_CU_BLOCK_SHARED typename BlockReduce::TempStorage temp_storage;

			// Struct to fill count_data at the enf
			int count[NumberOfPolyhedronInteractionTypes];
			for(size_t i = 0; i < NumberOfPolyhedronInteractionTypes ; i++)
			{
				count[i] = 0;
			}
			
			auto cell_a = cell_i[ONIKA_CU_BLOCK_IDX];

			cell_accessors cellA(cells[cell_a]);

			auto p_a = p_i[ONIKA_CU_BLOCK_IDX];
			particle_info p(shps, p_a, cellA);

			auto cell_b = cell_j[ONIKA_CU_BLOCK_IDX];
			cell_accessors cellB(cells[cell_b]);

			auto p_b = p_j[ONIKA_CU_BLOCK_IDX];
			particle_info p_nbh(shps, p_b, cellB);
			
            		const ParticleVertexView vertices_a = { p_a, gv[cell_a] };
            		const ParticleVertexView vertices_b = { p_b, gv[cell_b] };			

			count_interaction_block_pair( rVerlet, count, p, vertices_a, p_nbh, vertices_b );

			for(int i = 0; i < NumberOfPolyhedronInteractionTypes ; i++)
			{
				int aggregate = BlockReduce(temp_storage).Sum(count[i]);
				ONIKA_CU_BLOCK_SYNC();
				if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) count_data[ONIKA_CU_BLOCK_IDX][i] = aggregate;
			}
		}

		
	template<int BLOCKX, int BLOCKY, typename TMPLC, typename TMPLV>
		ONIKA_DEVICE_KERNEL_FUNC void fill_classifier_block_pair(
				InteractionSOA2* data,
				//InteractionSOA* data,
				//int* counts,
				TMPLC cells,
				TMPLV* const __restrict__ gv,
				IJK dims,
				shapes shps,
				double rVerlet,
				NumberOfPolyhedronInteractionPerTypes * shift_data,
                                uint32_t* cell_i,
                                uint32_t* cell_j,
                                uint16_t* p_i,
                                uint16_t* p_j)	
		{
			using BlockScan = cub::BlockScan<int, BLOCKX, cub::BLOCK_SCAN_RAKING, BLOCKY>;

			// cub stuff
			ONIKA_CU_BLOCK_SHARED typename BlockScan::TempStorage temp_storage;
			
			int count1 = 0;
			int count2 = 0;
			int count3 = 0;
			int count4 = 0;
			int count5 = 0;
			int count6 = 0;

			// Struct to fill count_data at the enf
			int count[NumberOfPolyhedronInteractionTypes];
			int prefix[NumberOfPolyhedronInteractionTypes];
			for(size_t i = 0; i < NumberOfPolyhedronInteractionTypes ; i++)
			{
				count[i] = 0;
				prefix[i] = 0;
			}

			auto cell_a = cell_i[ONIKA_CU_BLOCK_IDX];

			cell_accessors cellA(cells[cell_a]);

			auto p_a = p_i[ONIKA_CU_BLOCK_IDX];
			particle_info p(shps, p_a, cellA);


			auto cell_b = cell_j[ONIKA_CU_BLOCK_IDX];
			cell_accessors cellB(cells[cell_b]);

			auto p_b = p_j[ONIKA_CU_BLOCK_IDX];
			particle_info p_nbh(shps, p_b, cellB);
			
            		const ParticleVertexView vertices_a = { p_a, gv[cell_a] };
            		const ParticleVertexView vertices_b = { p_b, gv[cell_b] };

			count_interaction_block_pair2( rVerlet, count, count1, count2, count3, count4, count5, count6, p, vertices_a, p_nbh, vertices_b );
			//count_interaction_block_pair( rVerlet, count, p, vertices_a, p_nbh, vertices_b );

			ONIKA_CU_BLOCK_SYNC();

			NumberOfPolyhedronInteractionPerTypes sdata = shift_data[ONIKA_CU_BLOCK_IDX];
			for(int type = 0 ; type < NumberOfPolyhedronInteractionTypes ; type++)
			{
				BlockScan(temp_storage).ExclusiveSum(count[type], prefix[type]);
				ONIKA_CU_BLOCK_SYNC();
				prefix[type] += sdata[type];
			}
			Interaction item;

			item.id_i = p.id;
			item.cell_i = cell_a;
			item.p_i = p_a;
			/** Define interaction (section particle j) */
			item.cell_j = cell_b;
			item.id_j = p_nbh.id;
			item.p_j = p_b;

			fill_interaction_block( data, item, rVerlet, prefix, count1, count2, count3, count4, count5, count6, p, vertices_a, p_nbh, vertices_b );
			//fill_interaction_block( data, item, rVerlet, prefix, p, vertices_a, p_nbh, vertices_b/*, obb_i, obb_j*/);
			
			//atomicAdd(&counts[0], count1);
			//atomicAdd(&counts[1], count5);
			//atomicAdd(&counts[2], count3);
			//atomicAdd(&counts[3], count4);

		} // fill 

} // namespace exaDEM

