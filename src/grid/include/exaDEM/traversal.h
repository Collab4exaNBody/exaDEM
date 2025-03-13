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

struct Traversal
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
