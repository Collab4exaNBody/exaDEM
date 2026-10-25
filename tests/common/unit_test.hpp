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

/**
 * @file unit_test.hpp
 * @brief Minimal unit test framework (no external dependency).
 *
 * Usage:
 *   UNIT_TEST(my_test) {
 *     CHECK(a == b);
 *     CHECK_NEAR(x, 1.0, 1e-12);
 *   }
 *   UNIT_TEST_MAIN()
 *
 * The executable returns 0 if all checks pass, 1 otherwise, so it can be registered with add_test.
 */

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace exaDEM::unit_test {

struct TestCase {
  std::string name;
  std::function<void()> func;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline int& failure_counter() {
  static int failures = 0;
  return failures;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> func) { registry().push_back({name, std::move(func)}); }
};

inline void report_failure(const char* file, int line, const std::string& msg) {
  std::cerr << file << ":" << line << ": check failed: " << msg << std::endl;
  failure_counter()++;
}

inline int run_all() {
  int failed_tests = 0;
  for (auto& test : registry()) {
    const int before = failure_counter();
    test.func();
    const bool ok = (failure_counter() == before);
    if (!ok) {
      failed_tests++;
    }
    std::cout << (ok ? "[  OK  ] " : "[FAILED] ") << test.name << std::endl;
  }
  std::cout << registry().size() - failed_tests << "/" << registry().size() << " tests passed" << std::endl;
  return failed_tests == 0 ? 0 : 1;
}

}  // namespace exaDEM::unit_test

#define UNIT_TEST(name)                                                                \
  static void unit_test_##name();                                                      \
  static ::exaDEM::unit_test::Registrar unit_test_registrar_##name(#name, unit_test_##name); \
  static void unit_test_##name()

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      ::exaDEM::unit_test::report_failure(__FILE__, __LINE__, #cond);   \
    }                                                                   \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                                              \
  do {                                                                                                     \
    const double unit_test_a = (a);                                                                        \
    const double unit_test_b = (b);                                                                        \
    if (!(std::fabs(unit_test_a - unit_test_b) <= (tol))) {                                                \
      ::exaDEM::unit_test::report_failure(__FILE__, __LINE__,                                              \
                                          std::string(#a " ~= " #b " (") + std::to_string(unit_test_a) +   \
                                              " vs " + std::to_string(unit_test_b) + ")");                 \
    }                                                                                                      \
  } while (0)

#define CHECK_VEC3_NEAR(u, v, tol)     \
  do {                                 \
    CHECK_NEAR((u).x, (v).x, tol);     \
    CHECK_NEAR((u).y, (v).y, tol);     \
    CHECK_NEAR((u).z, (v).z, tol);     \
  } while (0)

#define UNIT_TEST_MAIN() \
  int main() { return ::exaDEM::unit_test::run_all(); }
