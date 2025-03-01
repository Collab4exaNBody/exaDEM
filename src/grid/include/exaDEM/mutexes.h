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
#include <omp.h>

struct cell_mutexes : public std::vector<omp_lock_t>
{
  inline void init()
  {
    for (auto &it : *this)
      omp_init_lock(&it);
  }

  inline void destroy()
  {
    for (auto &it : *this)
      omp_destroy_lock(&it);
  }

  inline void lock(const int i) { omp_set_lock(&this->operator[](i)); }

  inline void unlock(const int i) { omp_unset_lock(&this->operator[](i)); }
};

struct mutexes : public std::vector<cell_mutexes>
{
  mutexes() {}

  inline void initialize()
  {
#   pragma omp parallel for
    for (size_t i = 0; i < this->size(); i++)
      this->operator[](i).init();
  }
  inline void destroy()
  {
#   pragma omp parallel for
    for (size_t i = 0; i < this->size(); i++)
      this->operator[](i).destroy();
  }

  inline cell_mutexes &get_mutexes(const int i) { return this->operator[](i); }

  inline void lock(const int cell, const int index) { this->operator[](cell).lock(index); }

  inline void unlock(const int cell, const int index) { this->operator[](cell).unlock(index); }
};
