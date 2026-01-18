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

namespace exaDEM {
enum InteractionType {
  ParticleParticle,
  ParticleDriver,
  InnerBond
};

struct InteractionTypeId {
  // classic
  static constexpr int NTypesParticleParticle = 13;
  static constexpr int VertexVertex = 0;
  static constexpr int VertexEdge = 1;
  static constexpr int VertexFace = 2;
  static constexpr int EdgeEdge = 3;
  static constexpr int VertexCylinder = 4;
  static constexpr int VertexSurface = 5;
  static constexpr int VertexBall = 6;
  // fragmentation
  static constexpr int NTypesStickecParticles = 1;
  static constexpr int InnerBond = 13;
  static constexpr int NTypes = NTypesParticleParticle + NTypesStickecParticles;
  // control initialization
  static constexpr int Undefined = 666;
};
}  // namespace exaDEM
