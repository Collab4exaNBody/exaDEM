#pragma once

#include <exaDEM/drivers.h>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/shape_detection_driver.hpp>

namespace exaDEM
{
	/**
	 * @brief Add interactions between particles and a driver defined by an STL mesh.
	 *
	 * This function detects contacts between all particles (vertices, edges, faces)
	 * and the primitives (vertices, edges, faces) of a driver mesh. It relies on
	 * Oriented Bounding Boxes (OBB) for broad-phase filtering and then applies 
	 * fine-grained filters for vertex-vertex, vertex-edge, vertex-face, edge-edge,
	 * edge-vertex, and face-vertex interactions.
	 *
	 * @tparam Func   Functor type used to register contacts 
	 *                (signature: void(size_t pid, Interaction&, int sub_i, int sub_j)).
	 *
	 * @param mesh        STL mesh driver containing geometry and precomputed OBBs.
	 * @param cell_a      Index of the mesh grid cell to process (must be < mesh.grid_indexes.size()).
	 * @param add_contact Functor used to register detected contacts.
	 * @param item        Reusable interaction object (fields are updated during processing).
	 * @param n_particles Number of particles to process.
	 * @param rVerlet     Verlet radius (distance threshold for contact detection).
	 * @param type        Array mapping particle index -> type id.
	 * @param id          Array of unique particle identifiers.
	 * @param rx, ry, rz  Arrays of particle positions.
	 * @param vertices    Vertex field storing per-particle vertex positions.
	 * @param orient      Array of particle orientations (as quaternions).
	 * @param shps        Shape container indexed by particle type.
	 *
	 * @note The interaction type is encoded in `item.type`:
	 *       - 7 : particle-vertex vs driver-vertex
	 *       - 8 : particle-vertex vs driver-edge
	 *       - 9 : particle-vertex vs driver-face
	 *       - 10: particle-edge vs driver-edge
	 *       - 11: particle-edge vs driver-vertex
	 *       - 12: particle-face vs driver-vertex
	 */
	template <typename Func> 
		ONIKA_HOST_DEVICE_FUNC inline void add_driver_interaction(
				Stl_mesh &mesh, 
				size_t cell_a, 
				Func &add_contact, 
				PlaceholderInteraction &item, 
				const size_t n_particles, 
				const double rVerlet, 
				const ParticleTypeInt *__restrict__ type, 
				const uint64_t *__restrict__ id, 
				const double *__restrict__ rx, 
				const double *__restrict__ ry, 
				const double *__restrict__ rz, 
				VertexField& vertices,
				const exanb::Quaternion *__restrict__ orient, 
				shapes &shps)
		{
#define __particle__ vertices_i, i, shpi
#define __driver__ mesh.vertices.data(), idx, &mesh.shp
			assert(cell_a < mesh.grid_indexes.size());
			auto &list = mesh.grid_indexes[cell_a];
			const size_t stl_nv = list.vertices.size();
			const size_t stl_ne = list.edges.size();
			const size_t stl_nf = list.faces.size();

			if (stl_nv == 0 && stl_ne == 0 && stl_nf == 0)
				return;

			// Get OBBs from stl mesh
			auto &stl_shp = mesh.shp;
			OBB *__restrict__ stl_obb_vertices = onika::cuda::vector_data(stl_shp.m_obb_vertices);
			[[maybe_unused]] OBB *__restrict__ stl_obb_edges = onika::cuda::vector_data(stl_shp.m_obb_edges);
			[[maybe_unused]] OBB *__restrict__ stl_obb_faces = onika::cuda::vector_data(stl_shp.m_obb_faces);

      auto& pi = item.i(); // particle i (id, cell id, particle position, sub vertex)
      //auto& pd = item.driver(); // particle driver (id, cell id, particle position, sub vertex)

			for (size_t p = 0; p < n_particles; p++)
			{
				Vec3d r = {rx[p], ry[p], rz[p]}; // position
				ParticleVertexView vertices_i = {p, vertices};
				const Quaternion& orient_i = orient[p];
				pi.p = p;
				pi.id = id[p];
				auto ti = type[p];
				const shape *shpi = shps[ti];
				const size_t nv = shpi->get_number_of_vertices();
				const size_t ne = shpi->get_number_of_edges();
				const size_t nf = shpi->get_number_of_faces();

				// Compute particle OBB
				OBB obb_i = shpi->obb;
				quat conv_orient_i = quat{vec3r{orient_i.x, orient_i.y, orient_i.z}, orient_i.w};
				obb_i.rotate(conv_orient_i);
				obb_i.translate(vec3r{r.x, r.y, r.z});
				obb_i.enlarge(rVerlet);

				// Note:
				// loop i = particle p
				// loop j = stl mesh
				for (size_t i = 0; i < nv; i++)
				{
					vec3r v = conv_to_vec3r(vertices_i[i]);
					OBB obb_v_i;
					obb_v_i.center = v; 
					obb_v_i.enlarge(rVerlet + shpi->m_radius);

					// vertex - vertex
					item.pair.type = 7;
					pi.sub = i;
					for (size_t j = 0; j < stl_nv; j++)
					{
						size_t idx = list.vertices[j];
						if(filter_vertex_vertex_v2(rVerlet, __particle__, __driver__))
							//if(filter_vertex_vertex(rVerlet, __particle__, __driver__))
						{
							add_contact(p, item, i, idx);
						} 
					}
					// vertex - edge
					item.pair.type = 8;
					for (size_t j = 0; j < stl_ne; j++)
					{
						size_t idx = list.edges[j];
						if(filter_vertex_edge(rVerlet, __particle__, __driver__))
						{
							add_contact(p, item, i, idx);
						}
					}
					// vertex - face
					item.pair.type = 9;
					for (size_t j = 0; j < stl_nf; j++)
					{
						size_t idx = list.faces[j];
						const OBB& obb_f_stl_j = stl_obb_faces[idx];
						if( obb_f_stl_j.intersect(obb_v_i) )
						{
							if(filter_vertex_face(rVerlet, __particle__, __driver__))
							{
								add_contact(p, item, i, idx);
							}
						}
					}
				}

				for (size_t i = 0; i < ne; i++)
				{
					item.pair.type = 10;
					pi.sub = i;
					// edge - edge
					for (size_t j = 0; j < stl_ne; j++)
					{
						const size_t idx = list.edges[j];
						if(filter_edge_edge(rVerlet, __particle__, __driver__))
						{
							add_contact(p, item, i, idx);
						}
					}
				}

				for (size_t j = 0; j < stl_nv; j++)
				{
					const size_t idx = list.vertices[j];

					// rejects vertices that are too far from the stl mesh.
					const OBB& obb_v_stl_j = stl_obb_vertices[idx];
					if( !obb_v_stl_j.intersect(obb_i)) continue;

					item.pair.type = 11;
					// edge - vertex
					for (size_t i = 0; i < ne; i++)
					{
						if(filter_vertex_edge(rVerlet, __driver__, __particle__)) 
						{
							add_contact(p, item, i, idx);
						}
					}
					// face vertex
					item.pair.type = 12;
					for (size_t i = 0; i < nf; i++)
					{
						if(filter_vertex_face(rVerlet, __driver__, __particle__))
						{
							add_contact(p, item, i, idx);
						}
					}
				}
			} // end loop p
#undef __particle__
#undef __driver__
		} // end funcion

	/**
	 * @brief Add interactions between particles and a driver (boundary or external object).
	 *
	 * This function loops over all particles, fetches their associated shape,
	 * and tests each vertex against the given driver to determine if a contact 
	 * should be created. If a contact is detected, the provided functor 
	 * `add_contact` is invoked with the corresponding interaction data.
	 *
	 * @tparam DriverT   Type of the driver (e.g., surface, ball, cylinder).
	 * @tparam Func      Functor type used to register a contact (signature: void(size_t, Interaction&, int, int)).
	 *
	 * @param driver      Reference to the driver instance.
	 * @param add_contact Functor used to register new contacts.
	 * @param item        Reusable interaction object (its fields p_i and id_i are updated).
	 * @param n_particles Total number of particles.
	 * @param rVerlet     Verlet radius (tolerance distance for contact detection).
	 * @param type        Pointer to array mapping particle index -> type id.
	 * @param id          Pointer to array of unique particle ids.
	 * @param vertices    Vertex field storing particle vertex positions.
	 * @param shps        Shape container indexed by particle type.
	 */
	template <typename DriverT, typename Func> 
		ONIKA_HOST_DEVICE_FUNC inline void add_driver_interaction(
				DriverT &driver, 
				Func &add_contact, 
				PlaceholderInteraction &item, 
				const size_t n_particles, 
				const double rVerlet, 
				const ParticleTypeInt *__restrict__ type, 
				const uint64_t *__restrict__ id, 
				VertexField& vertices, 
				shapes &shps)
		{
			constexpr int DRIVER_VERTEX_SUB_IDX = -1; // Convention

      auto& pi = item.i(); // particle i (id, cell id, particle position, sub vertex)
      //auto& pd = item.driver(); // particle driver (id, cell id, particle position, sub vertex)

			for (size_t pid = 0; pid < n_particles; pid++)
			{
				pi.p = pid;
				pi.id = id[pid];
				ParticleVertexView vertex_view = {pid, vertices};

				const shape *shp = shps[type[pid]];
				assert(shp != nullptr);
				int num_vertices = shp->get_number_of_vertices();
				for (int vertex_index = 0; vertex_index < num_vertices; vertex_index++)
				{
					if (filter_vertex_driver(driver, rVerlet, vertex_view, vertex_index, shp))
					{
						add_contact(pid, item, vertex_index, DRIVER_VERTEX_SUB_IDX);
					}
				}
			}
		}
}
