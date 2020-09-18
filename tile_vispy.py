# based on Anton Sherwood's programs for painting hyperbolic tilings

import sys
from string import Template
from vispy import app, gloo
from math import sqrt, cos, sin, pi

vertex = '''
attribute vec2 position;

void main() {
  gl_Position = vec4(position, 0., 1.);
}
'''

fragment = ('''
uniform vec2 resolution;
uniform float shortdim;
uniform vec3 mirror1;
uniform vec3 mirror2;
uniform vec3 mirror3;

// --- minkowski geometry ---

// the minkowski bilinear form
float mprod(vec3 v, vec3 w) {
  return dot(v.xy, w.xy) - v.z*w.z;
}

float msq(vec3 v) {
  return mprod(v, v);
}

// a minkowski version of the built-in `reflect`
vec3 mreflect(vec3 v, vec3 mirror) {
  return v - 2*mprod(v, mirror)*mirror;
}

// --- tiling ---

vec3 sample(vec2 coord) {
  vec2 u = 1.2*(2.*coord - resolution) / shortdim;
  vec3 color = vec3(0.25, 0.15, 0.35);
  float r_sq = dot(u, u);
  if (r_sq < 1.) {
    vec3 v = vec3(2.*u, 1.+r_sq);
    int flips = 0;
    int onsides = 0;
    while (flips < 40) {
'''
+ ''.join([Template('''
      if (mprod($mirror, v) > 0.) {
        v = mreflect(v, $mirror);
        flips += 1;
        onsides = 0;
      } else {
        onsides += 1;
        if (onsides >= 3) {
          color = vec3(mod(flips, 2));
          break;
        }
      }
''').substitute(mirror = m) for m in ['mirror1', 'mirror2', 'mirror3']])
+ '''
    }
  }
  return color;
}

void main() {
  // mix subpixels
  /*vec2 jiggle = vec2(0.25);
  vec3 color_sum = vec3(0.);
  for (int sgn_x = 0; sgn_x < 2; sgn_x++) {
    for (int sgn_y = 0; sgn_y < 2; sgn_y++) {
      color_sum += sample(gl_FragCoord.xy + jiggle);
      jiggle.y = -jiggle.y;
    }
    jiggle.x = -jiggle.x;
  }
  gl_FragColor = vec4(0.25*color_sum, 1.);*/
  vec3 color_sum = vec3(0.);
  for (int jiggle_x = -1; jiggle_x < 2; jiggle_x++) {
    for (int jiggle_y = -1; jiggle_y < 2; jiggle_y++) {
      color_sum += sample(gl_FragCoord.xy + vec2(jiggle_x, jiggle_y)/3.);
    }
  }
  gl_FragColor = vec4(color_sum / 9., 1.);
}
''')

class TilingCanvas(app.Canvas):
  def __init__(self, p, q, r, *args, **kwargs):
    app.Canvas.__init__(self, *args, **kwargs)
    self.program = gloo.Program(vertex, fragment, count = 6) # we'll always send 6 vertices
    
    # draw a rectangle that covers the canvas
    self.program['position'] = [
      (-1, -1), (-1, 1), (1, 1), # northwest triangle
      (-1, -1), (1, 1), (1, -1)  # southeast triangle
    ]
    
    # initialize settings
    self.set_resolution()
    self.set_tiling(p, q, r)
  
  def set_resolution(self):
    width, height = self.physical_size
    gloo.set_viewport(0, 0, width, height)
    self.program['resolution'] = [width, height]
    self.program['shortdim'] = min(width, height)
  
  def set_tiling(self, p, q, r):
    # get angle cosines
    sp = sin(pi/p)
    cp = cos(pi/p)
    cq = cos(pi/q)
    cr = cos(pi/r)
    
    # find the side normals of the fundamental triangle, scaled to unit norm
    self.program['mirror1'] = (1, 0, 0)
    self.program['mirror2'] = (-cp, sp, 0)
    self.program['mirror3'] = (
      -cq,
      -(cp*cq + cr) / sp,
      sqrt(-1 + (cp*cp + cq*cq + cr*cr + 2*cp*cq*cr)) / sp
    )
  
  def on_draw(self, event):
    self.program.draw()
  
  def on_resize(self, event):
    self.set_resolution()

if __name__ == '__main__' and sys.flags.interactive == 0:
  orders = [2, 3, 7]
  title = 'Tiling {} {} {}'.format(*orders)
  TilingCanvas(*orders, size = (500, 500), title = title).show()
  app.run()
