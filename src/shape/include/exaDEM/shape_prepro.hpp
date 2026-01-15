/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 */
#pragma once

#include <exaDEM/shape.hpp>
#include <exaDEM/shape_detection.hpp>

namespace exaDEM
{
  using namespace exanb;

//  template<typename Vec>
    inline OBB build_OBB(const std::span<vec3r> vec, double radius)
    {
      OBB obb;
      vec3r mu;
      mat9r C;
      for (size_t i = 0; i < vec.size(); i++)
      {
        mu += vec[i];
      }
      mu /= (double)vec.size();

      // loop over the points again to build the
      // covariance matrix.  Note that we only have
      // to build terms for the upper trianglular
      // portion since the matrix is symmetric
      double cxx = 0.0, cxy = 0.0, cxz = 0.0, cyy = 0.0, cyz = 0.0, czz = 0.0;
      for (size_t i = 0; i < vec.size(); i++)
      {
        vec3r p = vec[i];
        cxx += p.x * p.x - mu.x * mu.x;
        cxy += p.x * p.y - mu.x * mu.y;
        cxz += p.x * p.z - mu.x * mu.z;
        cyy += p.y * p.y - mu.y * mu.y;
        cyz += p.y * p.z - mu.y * mu.z;
        czz += p.z * p.z - mu.z * mu.z;
      }

      // now build the covariance matrix
      C.xx = cxx;
      C.xy = cxy;
      C.xz = cxz;
      C.yx = cxy;
      C.yy = cyy;
      C.yz = cyz;
      C.zx = cxz;
      C.zy = cyz;
      C.zz = czz;

      // ==== set the OBB parameters from the covariance matrix
      // extract the eigenvalues and eigenvectors from C
      mat9r eigvec;
      vec3r eigval;
      C.sym_eigen(eigvec, eigval);

      // find the right, up and forward vectors from the eigenvectors
      vec3r r(eigvec.xx, eigvec.yx, eigvec.zx);
      vec3r u(eigvec.xy, eigvec.yy, eigvec.zy);
      vec3r f(eigvec.xz, eigvec.yz, eigvec.zz);
      r.normalize();
      u.normalize(), f.normalize();

      // now build the bounding box extents in the rotated frame
      vec3r minim(1e20, 1e20, 1e20), maxim(-1e20, -1e20, -1e20);
      for (size_t i = 0; i < vec.size(); i++)
      {
        vec3r p_prime(r * vec[i], u * vec[i], f * vec[i]);
        if (minim.x > p_prime.x)
          minim.x = p_prime.x;
        if (minim.y > p_prime.y)
          minim.y = p_prime.y;
        if (minim.z > p_prime.z)
          minim.z = p_prime.z;
        if (maxim.x < p_prime.x)
          maxim.x = p_prime.x;
        if (maxim.y < p_prime.y)
          maxim.y = p_prime.y;
        if (maxim.z < p_prime.z)
          maxim.z = p_prime.z;
      }

      // set the center of the OBB to be the average of the
      // minimum and maximum, and the extents be half of the
      // difference between the minimum and maximum
      obb.center = eigvec * (0.5 * (maxim + minim));
      obb.e1 = r;
      obb.e2 = u;
      obb.e3 = f;
      obb.extent = 0.5 * (maxim - minim);

      obb.enlarge(radius); // Add the Minskowski radius
      return obb;
    }

  inline OBB build_obb_from_shape(const shape& shp)
  {
    size_t nv = shp.get_number_of_vertices();
    
    const double ext = shp.minskowski(1.0);
    std::vector<vec3r> vbuf;
    vbuf.resize(nv);
    for (size_t i = 0; i < nv; i++)
    {
      vbuf[i] = conv_to_vec3r(shp.get_vertex(i));
    }
    OBB res = build_OBB(vbuf, ext);
    return res;
  }
  //#define OLD_VERSION;

  // general functon;
  inline OBB build_obb_vertex(const int index, const shape *shp, const Vec3d * v)
  {
    const double ext = shp->m_radius;
    const vec3r vertex = conv_to_vec3r(v[index]);
    std::array<vec3r, 1> vbuf = {vertex};
    OBB res = build_OBB(vbuf, ext);
    return res;
  }

  inline OBB build_obb_edge(const Vec3d &position, const int index, const shape *shp, const exanb::Quaternion &orientation)
  {
    const double ext = shp->m_radius;
    auto [first, second] = shp->get_edge(index);
    const Vec3d vf = shp->get_vertex(first, position, 1.0, orientation);
    const Vec3d vs = shp->get_vertex(second, position, 1.0, orientation);
    std::array<vec3r, 2> v = {conv_to_vec3r(vf), conv_to_vec3r(vs)};
    OBB res = build_OBB(v, ext);
    return res;
  }

  inline OBB build_obb_edge(const int index, const shape *shp, const Vec3d* v)
  {
    const double ext = shp->m_radius;
    auto [first, second] = shp->get_edge(index);
    const Vec3d& vf = v[first];
    const Vec3d& vs = v[second];
    std::array<vec3r,2> vbuf = {conv_to_vec3r(vf), conv_to_vec3r(vs)};
    OBB res = build_OBB(vbuf, ext);
    return res;
  }

  inline OBB build_obb_face(const Vec3d &position, const int index, const shape *shp, const exanb::Quaternion &orientation)
  {
    const double ext = shp->m_radius;
    const auto [data, nf] = shp->get_face(index);
    std::vector<vec3r> v(nf);
    for (int i = 0; i < nf; i++)
    {
      v[i] = conv_to_vec3r(shp->get_vertex(data[i], position, 1.0, orientation));
    }
    OBB res = build_OBB(v, ext);
    return res;
  }

  inline OBB build_obb_face(const int index, const shape *shp, const Vec3d * const v, std::vector<vec3r>& vbuf)
  {

    const double ext = shp->m_radius;
    const auto [data, nf] = shp->get_face(index);
    vbuf.resize(nf);
    for (int i = 0; i < nf; i++)
    {
      vbuf[i] = conv_to_vec3r(v[data[i]]);
    }
    OBB res = build_OBB(vbuf, ext);
    return res;
  }

  inline void shape::pre_compute_obb_vertices(const Vec3d * const v)
  {
    // This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_face`
    const size_t size = this->get_number_of_vertices();
    m_obb_vertices.resize(size);

    ldbg << "obb [vertices] = " << size << std::endl;

#   pragma omp parallel for schedule (static)
    for (size_t i = 0; i < size; i++)
    {
      m_obb_vertices[i] = build_obb_vertex(i, this, v);
    }
  }

  inline void shape::pre_compute_obb_edges(const Vec3d &particle_center, const exanb::Quaternion &particle_quat)
  {
    // This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_edge`
    const size_t size = this->get_number_of_edges();
    m_obb_edges.resize(size);

    // const exanb::Vec3d vnull      = {0,0,0};
    const exanb::Vec3d center = particle_center;//conv_to_Vec3d(this->obb.center) + particle_center;
    ldbg << "obb [edges]    = " << size << std::endl;
#   pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++)
    {
      m_obb_edges[i] = build_obb_edge(center, i, this, particle_quat);
    }
  }

  inline void shape::pre_compute_obb_edges(const Vec3d* v)
  {
    // This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_edge`
    const size_t size = this->get_number_of_edges();
    m_obb_edges.resize(size);
    ldbg << "obb [edges]    = " << size << std::endl;
#   pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++)
    {
      m_obb_edges[i] = build_obb_edge(i, this, v);
    }
  }


  inline void shape::pre_compute_obb_faces(const Vec3d &particle_center, const exanb::Quaternion &particle_quat)
  {
    // This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_face`
    const size_t size = this->get_number_of_faces();
    m_obb_faces.resize(size);
    const exanb::Vec3d center = particle_center; //conv_to_Vec3d(this->obb.center) + particle_center;
    ldbg << "obb [faces]    = " << size << std::endl;

#   pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++)
    {
      m_obb_faces[i] = build_obb_face(center, i, this, particle_quat);
    }
  }

  inline void shape::pre_compute_obb_faces(const Vec3d * v)
  {
    const size_t size = this->get_number_of_faces();
    m_obb_faces.resize(size);
    ldbg << "obb [faces]    = " << size << std::endl;

#   pragma omp parallel
    {
      std::vector<vec3r> vbuf; // buffer that will contain tmp vertex positions
#     pragma omp for schedule(static)
      for (size_t i = 0; i < size; i++)
      {
        m_obb_faces[i] = build_obb_face(i, this, v, vbuf);
      }
    }
  }

  inline void shape::increase_obb(const double value)
  {
#   pragma omp parallel
    {
#     pragma omp for schedule(static) nowait
      for (size_t i = 0; i < m_obb_vertices.size(); i++)
      {
        m_obb_vertices[i].enlarge(value);
      }
#     pragma omp for schedule(static) nowait
      for (size_t i = 0; i < m_obb_edges.size(); i++)
      {
        m_obb_edges[i].enlarge(value);
      }
#     pragma omp for schedule(static)
      for (size_t i = 0; i < m_obb_faces.size(); i++)
      {
        m_obb_faces[i].enlarge(value);
      }
    }
  }
  /*
     inline ONIKA_HOST_DEVICE OBB build_OBB(const vec3r * vec, const int * idx, int size)
     {
     double radius = 0;
     OBB obb;
     vec3r mu;
     mat9r C;
     for (size_t i = 0; i < size; i++)
     {
     mu += vec[idx[i]];
     }
     mu /= (double)size;

  // loop over the points again to build the
  // covariance matrix.  Note that we only have
  // to build terms for the upper trianglular
  // portion since the matrix is symmetric
  double cxx = 0.0, cxy = 0.0, cxz = 0.0, cyy = 0.0, cyz = 0.0, czz = 0.0;
  for (size_t i = 0; i < size; i++)
  {
  vec3r p = vec[idx[i]];
  cxx += p.x * p.x - mu.x * mu.x;
  cxy += p.x * p.y - mu.x * mu.y;
  cxz += p.x * p.z - mu.x * mu.z;
  cyy += p.y * p.y - mu.y * mu.y;
  cyz += p.y * p.z - mu.y * mu.z;
  czz += p.z * p.z - mu.z * mu.z;
  }

  // now build the covariance matrix
  C.xx = cxx;
  C.xy = cxy;
  C.xz = cxz;
  C.yx = cxy;
  C.yy = cyy;
  C.yz = cyz;
  C.zx = cxz;
  C.zy = cyz;
  C.zz = czz;

  // ==== set the OBB parameters from the covariance matrix
  // extract the eigenvalues and eigenvectors from C
  mat9r eigvec;
  vec3r eigval;
  C.sym_eigen(eigvec, eigval);

  // find the right, up and forward vectors from the eigenvectors
  vec3r r(eigvec.xx, eigvec.yx, eigvec.zx);
  vec3r u(eigvec.xy, eigvec.yy, eigvec.zy);
  vec3r f(eigvec.xz, eigvec.yz, eigvec.zz);
  r.normalize();
  u.normalize(), f.normalize();

  // now build the bounding box extents in the rotated frame
  vec3r minim(1e20, 1e20, 1e20), maxim(-1e20, -1e20, -1e20);
  for (size_t i = 0; i < size; i++)
  {
  vec3r p_prime(r * vec[idx[i]], u * vec[idx[i]], f * vec[idx[i]]);
  if (minim.x > p_prime.x)
  minim.x = p_prime.x;
  if (minim.y > p_prime.y)
  minim.y = p_prime.y;
  if (minim.z > p_prime.z)
  minim.z = p_prime.z;
  if (maxim.x < p_prime.x)
  maxim.x = p_prime.x;
  if (maxim.y < p_prime.y)
  maxim.y = p_prime.y;
  if (maxim.z < p_prime.z)
  maxim.z = p_prime.z;
  }

  // set the center of the OBB to be the average of the
  // minimum and maximum, and the extents be half of the
  // difference between the minimum and maximum
  obb.center = eigvec * (0.5 * (maxim + minim));
  obb.e1 = r;
  obb.e2 = u;
  obb.e3 = f;
  obb.extent = 0.5 * (maxim - minim);

  obb.enlarge(radius); // Add the Minskowski radius
  return obb;
}

  __global__ 
void compute_vertices(
    vec3r* scratch, 
    int size, 
    Vec3d center, 
    Quaternion quat, 
    const Vec3d * v)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x
    if( id < size )
    {
      scratch[id] = conv_to_vec3r(center + quat * v[id]);
    }
}

__host__ __device__ OBB build_OBB(const Vec3d& v)
{
  OBB obb = OBB();
  obb.center = v;
  return obb;
}


  __global__ 
void compute_obb_vertices(OBB * obb, const vec3r* scratch, int size)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x
    if( id < size )
    {
      obb[id] = build_OBB(scratch[id]);
    }
}


inline void big_stl_gpu(shape& shp, vec3r* scratch, const Vec3d &particle_center, const exanb::Quaternion &particle_quat)
{
  int nv = shp.get_numver_of_vertices();
  int ne = shp.get_numver_of_edges();
  int nf = shp.get_numver_of_faces();

  shp.m_obb_vertices.resize(nv); 
  shp.m_obb_edges.resize(ne); 
  shp.m_obb_faces.resize(nf); 
  OBB* obb_vertices = onika::cuda::vector_data( shp.m_obb_vertices );
  OBB* obb_edges = onika::cuda::vector_data( shp.m_obb_edges );
  OBB* obb_faces = onika::cuda::vector_data( shp.m_obb_faces );

  int blockSize = 256;
  int blockDimNV = (nv + 255)/256;
  int blockDimNE = (ne + 255)/256;
  int blockDimNF = (nf + 255)/256;
  compute_vertices<<<blockSize, blockDimNV>>>(scratch, nv, particle_center, particle_quat);
  cudaDeviceSynchronize();
  compute_obb_vertices<<<blockSize, blockDimNV>>>(obb_vertices, scratch, nv);
  //    compute_obb_edges<<<blockSize, blockDimNE>>>(obb_edges, scratch, ne);
  compute_obb_faces<<<blockSize, blockDimNF>>>(obb_faces, scratch, shp, nf);
  cudaDeviceSynchronize();

}
  */
inline void shape::compute_prepro_obb(Vec3d* scratch, const Vec3d &particle_center, const exanb::Quaternion &particle_quat)
{
  const size_t nv = this->get_number_of_vertices();
#   pragma omp parallel for schedule (static)
  for (size_t i = 0; i < nv; i++)
  {
    scratch[i] = this->get_vertex(i, particle_center, 1.0, particle_quat);
  }

  this->pre_compute_obb_vertices(scratch);
  this->pre_compute_obb_edges(scratch);
  this->pre_compute_obb_faces(scratch);
  this->increase_obb(this->m_radius); 
}
} // namespace exaDEM
