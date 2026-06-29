
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
#include <exaDEM/drivers.hpp>

#include <EGLRender/egl_render_manager.h>

namespace exaDEM
{

  using namespace onika;
  using namespace onika::scg;
  using namespace EGLRender;

  class EGLRenderDrawDrivers : public OperatorNode
  {
    ADD_SLOT( Drivers          , drivers            , INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT( std::string , shader_program , INPUT , "shader" );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      const int shader_id = egl_render_manager->shader_program_id(*shader_program);
      if( shader_id == -1 )
      {
        fatal_error() << "EGL Error: shader '"<< *shader_program << "' not found" <<std::endl;        
      }
      
      auto & shader = egl_render_manager->shader_program(shader_id);
      shader.use();

      const int driver_block_id = shader.uniform_id("DriverUniformObject");
      if(driver_block_id==-1)
      {
        fatal_error() << "EGL Error: uniform block 'DriverUniformObject' not found in shader #"<<shader.m_shader_program <<std::endl;
      }
      const int center_var_id = shader.uniform(driver_block_id).variable_id("DriverUniformObject.center");
      if(center_var_id==-1)
      {
        fatal_error() << "EGL Error: variable 'center' not found in uniform block 'DriverUniformObject'" <<std::endl;
      }
      const int quat_var_id = shader.uniform(driver_block_id).variable_id("DriverUniformObject.quat");
      if(quat_var_id==-1)
      {
        fatal_error() << "EGL Error: variable 'quat' not found in uniform block 'DriverUniformObject'" <<std::endl;
      }

      const int n_drivers = drivers->get_size();
      for(int drv_i=0;drv_i<n_drivers;drv_i++)
      {
        if( drivers->type(drv_i) == DRIVER_TYPE::RSHAPE )
        {
          const auto & drv = drivers->get_typed_driver<RShapeDriver>(drv_i);
          const auto c = drv.fields.center;
          const auto q = drv.fields.quat;
          GLfloat center[4] = { static_cast<GLfloat>(c.x), static_cast<GLfloat>(c.y), static_cast<GLfloat>(c.z), 1.0f };
          GLfloat quat[4] = { static_cast<GLfloat>(q.x), static_cast<GLfloat>(q.y), static_cast<GLfloat>(q.z), static_cast<GLfloat>(q.w) };
          shader.uniform(driver_block_id).variable(center_var_id).set( center, 4 );
          shader.uniform(driver_block_id).variable(quat_var_id).set( quat, 4 );
          
          const std::string vertex_buffer = "drv" + std::to_string(drv_i) + "_vertices";
          const int vbo_id = egl_render_manager->vertex_buffers_id(vertex_buffer);
          if(vbo_id==-1)
          {
            fatal_error() << "EGL Error: vertex buffer '"<<vertex_buffer <<"' not found"<<std::endl;
          }
          auto & vbo = egl_render_manager->vertex_buffers( vbo_id );

          const std::string element_buffer = "drv" + std::to_string(drv_i) + "_triangles";
          const int elbuf_id = egl_render_manager->element_buffer_id(element_buffer);
          if(elbuf_id==-1)
          {
            fatal_error() << "EGL Error: element buffer '"<<element_buffer <<"' not found"<<std::endl;
          }          
          auto & elbuf = egl_render_manager->element_buffer( elbuf_id );

          ldbg << "EGL : draw driver #"<<drv_i<<" : vert="<< vertex_buffer << ", elements="<< element_buffer << ", shader="<< *shader_program << std::endl;

          vbo.use();
          elbuf.draw(GL_TRIANGLES);
        }
      }
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_draw_drivers)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_draw_drivers", make_compatible_operator< EGLRenderDrawDrivers > );
  }

}

