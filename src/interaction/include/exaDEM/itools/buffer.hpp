#pragma once

namespace exaDEM
{
  namespace itools
  {
    struct interaction_buffers
    {
      template <typename T> using VectorT = onika::memory::CudaMMVector<T>;
      VectorT<double> dn;
      VectorT<Vec3d> cp;
      VectorT<Vec3d> fn;
      VectorT<Vec3d> ft;

      void resize(const size_t size)
      {
        assert(size < 1e9);
        if (size != 0)
        {
          dn.resize(size);
          cp.resize(size);
          fn.resize(size);
          ft.resize(size);
        }
        else
        {
          dn.clear();
          cp.clear();
          fn.clear();
          ft.clear();
        }
      }
    };
  } // namespace itools
} // namespace exaDEM
