
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

#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/log.h>
#include <onika/math/basic_types_def.h>
#include <exaDEM/shapes.hpp>
#include <EGLRender/egl_render_manager.h>

namespace exaDEM
{
  using namespace EGLRender;
  using namespace onika::scg;

  class EGLShapeToShaderInclude : public OperatorNode
  {
    ADD_SLOT( shapes           , shapes_collection  , INPUT_OUTPUT, DocString{"Collection of shapes"});

  public:
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      ldbg << "egl_shape_to_buffer : nb shapes = " << shapes_collection->size() << std::endl;
      std::ostringstream oss;
      int max_face_vertices = 0;
      int max_faces = 0;
      int max_vertices = 0;
      for(size_t i=0;i<shapes_collection->size();i++)
      {
        int totalvertices = 0;
        const auto & shp = * (*shapes_collection)[i];
        ldbg << "adding shape "<< shp.m_name << " to shader code dem/shapes"<< std::endl;
        oss<< "#define DEM_SHAPE_"<< shp.m_name<<" "<<i<<"\n";

        // create vertex buffer
        const long vcount = shp.get_number_of_vertices();

        oss<< "void dem_shape_" << shp.m_name << "_emit_faces(vec4 c, mat4 P, mat4 M, mat3 Q, vec4 diffuseColor, vec3 lightPosition)\n{\n";

        oss<< "\tvec3 ambientColor = vec3(0.1,0.1,0.1);\n";
        oss<< "\tvec3 specularColor = vec3(1,1,1);\n";

        oss<< "\tconst vec4 v["<<vcount<<"] = {\n";
        for(long i=0;i<vcount;i++)
        {
          const auto v = shp.get_vertex(i);
          oss<<"\t\t"<< ( (i>0)?", ":"  " ) <<"M * ( c + vec4(Q*vec3("<<v.x<<","<<v.y<<","<<v.z<<"),1) )\n";
        }
        oss<<"\t\t};\n";
        oss<<"\tvec3 p0,p1,p2,N;\n";
        oss<<"\tfloat d,s;\n";

        auto emit_vertex_mul_p = [&](int vi) { oss << "\tgl_Position = P * v["<<vi<<"]; EmitVertex();\n"; };
        //auto emit_vertex = [&](int vi) { oss << "\tgl_Position = v["<<vi<<"]; EmitVertex();\n"; };

        // create triangle indices buffer
        const int n_faces = shp.get_number_of_faces();
        for(int j=0;j<n_faces;j++)
        {
          const auto [idx,n] = shp.get_face(j);
          oss << "\t// face #"<<j<<" has "<<n<<" vertices :";
          for(int k=0;k<n;k++) oss << " " << idx[k];
          oss<<"\n";
          oss<<"\tp0 = v["<<idx[0]<<"].xyz/v["<<idx[0]<<"].w;\n";
          oss<<"\tp1 = v["<<idx[1]<<"].xyz/v["<<idx[1]<<"].w;\n";
          oss<<"\tp2 = v["<<idx[2]<<"].xyz/v["<<idx[2]<<"].w;\n";
          oss<<"\tN = normalize( cross( p1-p0 , p2-p0 ) );\n";
          oss<<"\td = max( dot( N , normalize(lightPosition) ) , 0 );\n";
          oss<<"\tfDiffuseColor.xyz = diffuseColor.xyz * d + ambientColor;\n";
          oss<<"\tfDiffuseColor.w = diffuseColor.w;\n";
          int ev = 0;
          if( n==4 )
          {
            emit_vertex_mul_p(idx[0]);
            emit_vertex_mul_p(idx[1]);
            emit_vertex_mul_p(idx[3]);
            emit_vertex_mul_p(idx[2]);
            ev += 4;
          }
          else for(int k=1;k<(n-1);k++)
          {
            emit_vertex_mul_p(idx[0]);
            emit_vertex_mul_p(idx[k]);
            emit_vertex_mul_p(idx[k+1]);
            ev += 3;
          }
          oss << "\tEndPrimitive();\n";
          if( ev > max_face_vertices ) max_face_vertices = ev;
          totalvertices += ev;
        }
        oss<<"}\n";
        
        if( totalvertices > max_vertices ) max_vertices = totalvertices;
        if( n_faces > max_faces ) max_faces = n_faces;
      }
      oss<<"#define DEM_SHAPE_FACE_MAX_VERTICES "<<max_face_vertices<<"\n";
      oss<<"#define DEM_SHAPE_MAX_FACES "<<max_faces<<"\n";
      oss<<"#define DEM_SHAPE_MAX_VERTICES "<<max_vertices<<"\n";
      oss<<"void dem_shape_emit_faces(vec4 center, mat4 P, mat4 M, mat3 Q, int shape, vec4 diffuseColor, vec3 lightPosition)\n{\n";
      for(size_t i=0;i<shapes_collection->size();i++)
      {
        const auto & shp = * (*shapes_collection)[i];
        oss<<"\t" << ((i>0)?"else ":"") << "if(shape==DEM_SHAPE_"<<shp.m_name<<") dem_shape_" << shp.m_name << "_emit_faces(center,P,M,Q,diffuseColor,lightPosition);\n";
      }
      oss<<"}\n";

      std::string code = oss.str();
      //ldbg << "Shader code:" <<std::endl<<code;
      platform_add_named_string( "dem/shapes" , code );
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_shape_to_shader)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_shape_to_shader", make_compatible_operator< EGLShapeToShaderInclude > );
  }

}

