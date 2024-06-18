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
#ifndef COMMON_HPP_F4424C49
#define COMMON_HPP_F4424C49

#include <string>
#include <limits>

// possible other name
// WordSeparator

#define CommBox Common::Instance

typedef double real;
typedef unsigned int uint;

class Common
{
public:
  char sep; 
  int precision;
	
  static Common& Instance() {
    static Common inst;
    return inst;
  }
	
  void set_precision (int p) {
    if (p <= 0) precision = std::numeric_limits<double>::digits10 + 1;
    else precision = p;
  }
	
  void sepFromKeyword(std::string &  kw) {
    if (kw == "tab") sep = '\t';
    else if (kw == "semicolon") sep = ';';
    else if (kw == "space") sep = ' ';
    else sep = ' ';
  }
	
  std::string keywordFromSep() {
    if (sep == ' ') return std::string("space");
    if (sep == '\t') return std::string("tab");
    if (sep == ';') return std::string("semicolon");
    return std::string("space");
  }
	
private:
  Common():sep(' '), precision(std::numeric_limits<double>::digits10 + 1) { }
};

#endif /* end of include guard: COMMON_HPP_F4424C49 */
