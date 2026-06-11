
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
      for(size_t i=0;i<shapes_collection->size();i++)
      {
        const auto & shp = * (*shapes_collection)[i];
        ldbg << "adding shape "<< shp.m_name << " to shader code dem/shapes"<< std::endl;
        oss<< "#define DEM_SHAPE_"<< shp.m_name<<" "<<i<<"\n";
        oss<< "void dem_shape_" << shp.m_name << "_emit_faces(vec4 c, mat4 M, mat3 Q)\n{\n";

        // create vertex buffer
        const long vcount = shp.get_number_of_vertices();
        oss<< "\tconst vec3 v["<<vcount<<"] = {\n";

        ldbg << "EGL : generate vertex array , nv="<< vcount <<std::endl;
        for(long i=0;i<vcount;i++)
        {
          const auto v = shp.get_vertex(i);
          oss<<"\t\t"<< ( (i>0)?", ":"  " ) <<"{"<<v.x<<","<<v.y<<","<<v.z<<"}\n";
        }
        oss<<"\t\t};\n";

        // create triangle indices buffer
        const size_t n_faces = shp.get_number_of_faces();
        ldbg << "EGL : generate triangle emit code, n_faces="<< n_faces <<std::endl;
        for(size_t j=0;j<n_faces;j++)
        {
          const auto [idx,n] = shp.get_face(j);
          oss << "\t// face #"<<j<<" has "<<n<<" vertices\n";
          int fi = 0;
          int fj = n - 1;
          bool s = false;
          int nv=0;
          while( fi <= fj )
          {
            int vi = s ? fj-- : fi++;
            s = ! s;
            ++nv;
            oss << "\tgl_Position = M * ( c + vec4(Q*v["<<vi<<"],1) ); EmitVertex();\n";
          }
          oss << "\tEndPrimitive();\n";
          if( nv > max_face_vertices ) max_face_vertices = nv;
        }
        oss<<"}\n";
      }
      oss<<"#define DEM_SHAPE_FACES_MAX_VERTICES "<<max_face_vertices<<"\n";
      oss<<"void dem_shape_emit_faces(vec4 center, mat4 M, mat3 Q, int shape)\n{\n";
      for(size_t i=0;i<shapes_collection->size();i++)
      {
        const auto & shp = * (*shapes_collection)[i];
        oss<<"\t" << ((i>0)?"else ":"") << "if(shape==DEM_SHAPE_"<<shp.m_name<<") dem_shape_" << shp.m_name << "_emit_faces(center,M,Q);\n";
      }
      oss<<"}\n";
      std::string code = oss.str();
      ldbg << "Shader code:" <<std::endl<<code;
      platform_add_named_string( "dem/shapes" , code );
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_shape_to_shader)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_shape_to_shader", make_compatible_operator< EGLShapeToShaderInclude > );
  }

}

