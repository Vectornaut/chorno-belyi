# based on
#
# John David Reaver, "Mandelbrot set"
# <https://github.com/vispy/vispy/blob/master/examples/demo/gloo/mandelbrot.py>
#
# Nicolas P. Rougier, "Rotating quad"
# <http://api.vispy.org/en/v0.4.0/examples/tutorial/gloo/rotating_quad.html>
#
# Cyrille Rossant, "Featured Recipe #6: Getting started with Vispy"
# <https://github.com/ipython-books/cookbook-code/blob/master/featured/06_vispy.ipynb>
#
# Many authors, "start.py"
# <https://github.com/vispy/vispy/blob/master/examples/basics/gloo/start.py>
#
# --- installation ---
#
#   "VisPy requires at least one toolkit for opening a window and creates an
#   OpenGL context. This can be done using one Qt, GLFW, SDL2, Wx, or Pyglet. You
#   can also use a Jupyter notebook with WebGL for some visualizations although
#   some visuals may not be possible (ex. volume rendering)."
#
#   "You only need to have one of these packages, no need to install them all!"
#
#   <http://vispy.org/installation.html>
#
# the GLFW runtime and development packages are
#
#   libglfw3
#   libglfw3-dev
#
# on Ubuntu 18.04

import sys
from vispy import app, gloo

vertex = '''
attribute vec2 position;

void main() {
  gl_Position = vec4(position, 0., 1.);
}
'''

fragment = '''
uniform vec2 resolution;
uniform float shortdim;

void main() {
  vec2 u = (2.*gl_FragCoord.xy - resolution) / shortdim;
  if (abs(u.x) + abs(u.y) < 1.) {
    gl_FragColor = vec4(0.5*(u + 1.), 0.8, 1.);
  } else {
    gl_FragColor = vec4(vec3(0.2), 1.);
  }
}
'''

class HelloCanvas(app.Canvas):
  def __init__(self, *args, **kwargs):
    app.Canvas.__init__(self, *args, **kwargs)
    self.program = gloo.Program(vertex, fragment, count = 6) # we'll always send 6 vertices
    
    # draw a rectangle that covers the canvas
    self.program['position'] = [
      (-1, -1), (-1, 1), (1, 1), # northwest triangle
      (-1, -1), (1, 1), (1, -1)  # southeast triangle
    ]
    
    # set initial resolution
    self.set_resolution()
  
  def set_resolution(self):
    width, height = self.physical_size
    gloo.set_viewport(0, 0, width, height)
    self.program['resolution'] = [width, height]
    self.program['shortdim'] = min(width, height)
  
  def on_draw(self, event):
    self.program.draw()
  
  def on_resize(self, event):
    self.set_resolution()

if __name__ == '__main__' and sys.flags.interactive == 0:
  HelloCanvas(size = (500, 500), title = 'Hello, Vispy!').show()
  app.run()
