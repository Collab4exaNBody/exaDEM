
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

  class EGLShapeToBuffer : public OperatorNode
  {
    ADD_SLOT( shapes           , shapes_collection  , INPUT_OUTPUT, DocString{"Collection of shapes"});
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      std::abort();
      std::cout << "egl_shape_to_buffer : nb shapes = " << shapes_collection->size() << std::endl;
      for(size_t i=0;i<shapes_collection->size();i++)
      {
        const auto & shp = * (*shapes_collection)[i];
        std::cout << "converting shape "<< shp.m_name << "to OpenGL buffers" << std::endl;

        // create vertex buffer
        const std::string vertex_buffer = shp.m_name + "_vertices";
        const long vcount = shp.get_number_of_vertices();
        int vbuf_id = egl_render_manager->vertex_buffers_id( vertex_buffer );
        if( vbuf_id == -1 )
        {
          std::cout << "EGL : create vertex buffer " << vertex_buffer <<std::endl;
          std::vector<GLint> vertex_attribs = { GL_FLOAT, 3 };
          vbuf_id = egl_render_manager->create_vertex_buffers( vertex_buffer , vcount , vertex_attribs );
        }
        GLVertexBuffers & glvbos = egl_render_manager->vertex_buffers(vbuf_id);
        std::cout << "EGL : update vertex buffer " << vertex_buffer << " , nv="<< vcount << " , id="<<vbuf_id<<std::endl;
        glvbos.set_number_of_vertices( vcount );
        GLfloat * vdata = (GLfloat*) glvbos.host_map_write_only(0);
        for(long i=0;i<vcount;i++)
        {
          const auto v = shp.get_vertex(i);
          vdata[i*3+0] = v.x;
          vdata[i*3+1] = v.y;
          vdata[i*3+2] = v.z;
        }
        glvbos.host_unmap(0);

        // create triangle indices buffer
        const std::string element_buffer = shp.m_name + "_triangles";
        long tcount = 0; 
        const size_t n_faces = shp.get_number_of_faces();
        for(size_t j=0;j<n_faces;j++)
        {
          const auto [idx,n] = shp.get_face(j);
          if(n>=3) tcount += n-2; // we assume each face is a simple polygon (a loop)
        }
        int ebuf_id = egl_render_manager->element_buffer_id( element_buffer );
        if( ebuf_id == -1 )
        {
          std::cout << "EGL : create element buffer " << element_buffer <<std::endl;
          ebuf_id = egl_render_manager->create_element_buffer( element_buffer , tcount * 3 );
        }
        
        auto & ebuf = egl_render_manager->element_buffer(ebuf_id);
        std::cout << "EGL : update element buffer " << element_buffer << " , triangles="<< tcount << " , id="<<ebuf_id<<std::endl;
        auto * elptr = ebuf.map_buffer_write_only();
        long ti = 0;
        for(size_t j=0;j<n_faces;j++)
        {
          const auto [idx,n] = shp.get_face(j);
          for(int j=1;j<(n-1);j++) { elptr[ti++]=idx[0]; elptr[ti++]=idx[j]; elptr[ti++]=idx[j+1]; }
        }
        if( ti != tcount * 3 )
        {
          onika::fatal_error() << "Internal error: wrong vertex count"<<std::endl;
        }
        ebuf.unmap_buffer();
      }
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_shape_to_buffer)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_shape_to_buffer", make_compatible_operator< EGLShapeToBuffer > );
  }

}

