#pragma once

#include <vector>
#include <exaDEM/stl_mesh.h>
#include <exaDEM/shape/shape.hpp>

namespace exaDEM
{
	using namespace exanb;
	shape build_shape(stl_mesh& mesh, std::string name)
	{
		shape shp;
		const int n_faces = mesh.m_data.size();
		std::vector<int> idx;
		auto& shp_v = shp.m_vertices;
		auto& shp_e = shp.m_edges;
		auto& shp_f = shp.m_faces;

		// tmp edges
		std::vector<std::pair<size_t, size_t>> tmp_e;

		shp_f.resize(1);
		shp_f[0] = n_faces;

#pragma omp parallel
		{
			// first get vertices
			std::vector<Vec3d> v;
#pragma omp for
			for (int i = 0 ; i < n_faces ; i++ )
			{
				Face& face = mesh.get_data( i );
				v.insert( std::end(v),  std::begin( face.vertices ), std::end ( face.vertices ) );
			}

			auto last = std::unique(v.begin(), v.end());
			v.erase ( last, v.end() );
			std::sort(v.begin() , v.end());

#pragma omp critical
			{
				shp_v.insert( std::end(shp_v), std::begin(v), std::end(v));
			}
		}

		{
			auto last = std::unique(shp_v.begin(), shp_v.end());
			shp_v.erase ( last, shp_v.end() );
			std::sort(shp_v.begin() , shp_v.end());
		}

#pragma omp parallel
		{
			// first get vertices
			std::vector<size_t> idxs;
			std::vector<std::pair<size_t, size_t>> edges;
			std::vector<int> faces;
#pragma omp for
			for (int i = 0 ; i < n_faces ; i++ )
			{
				Face& face = mesh.get_data( i );
				auto& v = face.vertices;
				idxs.resize ( v.size() );
				for( size_t idx = 0 ; idx < v.size() ; idx++ )
				{
					auto it = std::lower_bound ( std::begin(shp_v), std::end(shp_v) , v[idx]);
					idxs[idx] = std::distance ( shp_v.begin() , it );
					assert ( v[idx] == shp_v[idxs[idx]]);
				}
				// add edges and faces
				faces.push_back( v.size() );
				for( size_t idx = 0 ; idx < v.size() ; idx++ )
				{
					faces.push_back( idxs[idx] );
					size_t second ;
					if( idx + 1 == v.size() ) second = idxs[0];
					else second = idxs[idx+1];
					edges.push_back( {idxs[idx], second} );
					}
				}
#pragma omp critical
				{
					tmp_e.insert( std::end(tmp_e), std::begin(edges), std::end(edges));
				}
#pragma omp critical
				{
					shp_f.insert( std::end(shp_f), std::begin(faces), std::end(faces));
				}
			}

			{
				auto last = std::unique(tmp_e.begin(), tmp_e.end());
				tmp_e.erase ( last, tmp_e.end() );
				shp_e.resize( tmp_e.size() * 2 );
#pragma omp parallel for schedule(static)
				for(size_t i = 0 ; i< tmp_e.size() ; i++)
				{
					shp_e[2*i]   = tmp_e[i].first;
					shp_e[2*i+1] = tmp_e[i].second;
				} 
			}

			shp.m_name = name;
			shp.pre_compute_obb_vertices();
			shp.pre_compute_obb_edges();
			shp.pre_compute_obb_faces();
			shp.write_paraview();
			return shp;
	}
}
