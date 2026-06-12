
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
#include <exaDEM/drivers.hpp>
#include <exaDEM/shape.hpp>
#include <EGLRender/egl_render_manager.h>
#include <onika/parallel/parallel_for.h>

namespace exaDEM
{
  struct GLCopyDriverVertices
  {
    const onika::math::Vec3d * __restrict__ m_shape_vertices = nullptr;
    GLfloat * __restrict__ m_vertex_buffer = nullptr;
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t i ) const
    {
      const auto v = m_shape_vertices[i];
      m_vertex_buffer[i*3+0] = v.x;
      m_vertex_buffer[i*3+1] = v.y;
      m_vertex_buffer[i*3+2] = v.z;
    }
  };
}

namespace onika
{
  namespace parallel
  {
    template<> struct ParallelForFunctorTraits<exaDEM::GLCopyDriverVertices>
    {
      static inline constexpr bool CudaCompatible = true;
    };
  }
}

namespace exaDEM
{
  using namespace EGLRender;
  using namespace onika::scg;

  class EGLDriverToBuffer : public OperatorNode
  {
    using IntVector = std::vector<int>;
    ADD_SLOT( Drivers          , drivers            , INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );
    ADD_SLOT( IntVector , egl_cuda_devices , INPUT_OUTPUT , IntVector{} );

  public:
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      const bool runs_on_gpu = ( global_cuda_ctx()!=nullptr && global_cuda_ctx()->has_devices() );
      const int n_drivers = drivers->get_size();
      ldbg << "egl_driver_to_buffer : nb drivers = " << n_drivers << std::endl;
      long vcount = 0;
      long tcount = 0;
      for(int i=0;i<n_drivers;i++)
      {
        if( drivers->type(i) == DRIVER_TYPE::RSHAPE )
        {
          const auto & drv = drivers->get_typed_driver<RShapeDriver>(i);
          const auto & shp = drv.shp;
          ldbg << "driver #"<<i<<" is rshape, shape : nv="<<shp.get_number_of_vertices()<<", nf="<<shp.get_number_of_faces()<<", ntv="<<drv.vertices.size()<<std::endl;
          vcount += shp.get_number_of_vertices();
          for(int j=0;j<shp.get_number_of_faces();j++)
          {
            const auto [idx,n] = shp.get_face(j);
            if(n>=3) tcount += n-2;
          }
        }
      }

      ldbg << "total vertices = "<< vcount << ", total triangles = "<< tcount << std::endl;

      // create vertex buffer
      const std::string vertex_buffer = "driver_vertices";
      int vbuf_id = egl_render_manager->vertex_buffers_id( vertex_buffer );
      if( vbuf_id == -1 )
      {
        ldbg << "EGL : create vertex buffer " << vertex_buffer <<std::endl;
        std::vector<GLint> vertex_attribs = { GL_FLOAT, 3, GL_FLOAT, 3, GL_FLOAT, 3};
        vbuf_id = egl_render_manager->create_vertex_buffers( vertex_buffer , vcount , vertex_attribs );
      }
      GLVertexBuffers & glvbos = egl_render_manager->vertex_buffers(vbuf_id);
      ldbg << "EGL : update vertex buffer " << vertex_buffer << " , nv="<< vcount << " , id="<<vbuf_id<<std::endl;
      glvbos.set_number_of_vertices( vcount );
      GLfloat * vdata = nullptr;
      if( runs_on_gpu ) vdata = (GLfloat*) glvbos.gpu_map_write_only(0);
      else vdata = (GLfloat*) glvbos.host_map_write_only(0);

      long vertidx = 0;
      for(int i=0;i<n_drivers;i++)
      {
        if( drivers->type(i) == DRIVER_TYPE::RSHAPE )
        {
          const auto & drv = drivers->get_typed_driver<RShapeDriver>(i);
          const auto & shp = drv.shp;
          long dnv = shp.get_number_of_vertices();
          GLCopyDriverVertices func = { drv.vertices.data(), vdata + vertidx*3 };
          onika::parallel::parallel_for( dnv, func, parallel_execution_context("glcpydrv") );
          vertidx += dnv;
        }
      }
      if( runs_on_gpu ) glvbos.gpu_unmap(0);
      else glvbos.host_unmap(0);
      if( vertidx != vcount )
      {
        onika::fatal_error() << "inconsistent number of vertices"<<std::endl;
      }

      // create triangle indices buffer
      const std::string element_buffer = "driver_triangles";
      int ebuf_id = egl_render_manager->element_buffer_id( element_buffer );
      if( ebuf_id == -1 )
      {
        ldbg << "EGL : create element buffer " << element_buffer <<std::endl;
        ebuf_id = egl_render_manager->create_element_buffer( element_buffer , tcount*3 );

        auto & ebuf = egl_render_manager->element_buffer(ebuf_id);
        ldbg << "EGL : update element buffer " << element_buffer << " , triangles="<< tcount << " , id="<<ebuf_id<<std::endl;
        auto * elptr = ebuf.map_buffer_write_only();

        long ti = 0;
        long vi = 0;
        for(int i=0;i<n_drivers;i++)
        {
          if( drivers->type(i) == DRIVER_TYPE::RSHAPE )
          {
            const auto & drv = drivers->get_typed_driver<RShapeDriver>(i);
            const auto & shp = drv.shp;
            const int n_faces = shp.get_number_of_faces();
            for(int j=0;j<n_faces;j++)
            {
              const auto [idx,n] = shp.get_face(j);
              for(int k=1;k<(n-1);k++) { elptr[ti++]=vi+idx[0]; elptr[ti++]=vi+idx[k]; elptr[ti++]=vi+idx[k+1]; }
            }
            vi += shp.get_number_of_vertices();
          }
        }
        ebuf.unmap_buffer();
        if( ti != tcount * 3 )
        {
          onika::fatal_error() << "Internal error: wrong triangle count"<<std::endl;
        }
      }
      else
      {
        ldbg << "EGL : skip element buffer " << element_buffer << " update , id="<<ebuf_id<<std::endl;
      }
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_driver_to_buffer)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_driver_to_buffer", make_compatible_operator< EGLDriverToBuffer > );
  }

}

