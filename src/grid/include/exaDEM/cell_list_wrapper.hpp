#pragma once

struct CellListWrapper
{
  template <typename T> using VectorT = onika::memory::CudaMMVector<T>;
  VectorT<size_t> m_data;
  
  bool iterator = false;

  size_t *data() { return onika::cuda::vector_data(m_data); }

  size_t size() { return onika::cuda::vector_size(m_data); }

  std::tuple<size_t *, size_t> info()
  {
    const size_t s = this->size();
    if (s == 0)
      return {nullptr, 0};
    else
      return {this->data(), this->size()};
  }
};
