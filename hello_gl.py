# based on
#
# John T. Goetz, "Learning OpenGL with Python"
# <https://www.metamost.com/opengl-with-python/>
#
# Cyrille Rossant, "2D graphics rendering tutorial with PyOpenGL"
# <https://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/>
#
# to use GLFW, you have to install both the Python bindings and the package
# itself. the Python bindings package is `glfw`. its installation command is
#
#   pip3 install glfw
#
# on Ubuntu 18.04, the package itself is in `libglfw3` and `libglfw3-dev`. its
# installation command is
#
#   sudo apt-get install libglfw3
#   sudo apt-get install libglfw3-dev

import contextlib
import sys
from OpenGL import GL as gl
import glfw

@contextlib.contextmanager
def open_window(size):
  if not glfw.init():
    sys.exit(1)
  try:
    # see GLFW window guide for more!
    # <https://www.glfw.org/docs/latest/window.html>
    
    # request
    ##glfw.window_hint(glfw.OPENGL_ES_API)
    
    # we need OpenGL >= 3.2 to use context profiles
    ## for OpenGL ES, i don't think we need this
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
    
    # drop deprecated functionality
    ## if OpenGL ES is requested, this hint is ignored
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    
    # only require core functionality
    ## if OpenGL ES is requested, this hint is ignored
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    # open window
    window = glfw.create_window(size, size, "Hello, GL!", None, None)
    if not window:
      sys.exit(2)
    glfw.make_context_current(window)
    
    # set window options
    glfw.set_input_mode(window, glfw.STICKY_KEYS, True)
    
    yield window
  finally:
    glfw.terminate()

if __name__ == '__main__' and sys.flags.interactive == 0:
  with open_window(500) as window:
    while(
      glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS
      and not glfw.window_should_close(window)
    ):
      gl.glClear(gl.GL_COLOR_BUFFER_BIT)
      glfw.swap_buffers(window)
      glfw.poll_events()
