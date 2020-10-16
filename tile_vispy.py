# based on Anton Sherwood's programs for painting hyperbolic tilings

import sys
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
uniform vec3 mirrors [3];
uniform bool antialias;

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

const float VIEW = 1.2;
const float SQRT2 = 1.4142135623730951;

void main_none() {
  vec2 u = VIEW*(2.*gl_FragCoord.xy - resolution) / shortdim;
  float r_sq = dot(u, u);
  if (r_sq < 1.) {
    vec3 v = vec3(2.*u, 1.+r_sq);
    int flips = 0;
    int onsides = 0;
    while (flips < 40) {
      for (int k = 0; k < 3; k++) {
        if (mprod(mirrors[k], v) > 0.) {
          v = mreflect(v, mirrors[k]);
          flips += 1;
          onsides = 0;
        } else {
          onsides += 1;
          if (onsides >= 3) {
            gl_FragColor = vec4(vec3(mod(flips, 2)), 1.);
            return;
          }
        }
      }
    }
  }
  gl_FragColor = vec4(0.25, 0.15, 0.35, 1.);
}

const float A1 = 0.278393;
const float A2 = 0.230389;
const float A3 = 0.000972;
const float A4 = 0.078108;

// Abramowitz and Stegun, equation 7.1.27
float erfc_appx(float t) {
  float p = 1. + A1*(t + A2*(t + A3*(t + A4*t)));
  float p_sq = p*p;
  return 1. / (p_sq*p_sq);
}

void main_gauss() {
  // find screen coordinate
  vec2 u = VIEW*(2.*gl_FragCoord.xy - resolution) / shortdim;
  float r_sq = dot(u, u);
  
  // find pixel radius, for antialiasing
  float r_px_screen = VIEW / shortdim; // the inner radius of a pixel in the Euclidean metric of the screen
  float r_px = 2.*r_px_screen / (1.-r_sq); // the approximate inner radius of our pixel in the hyperbolic metric
  
  // reduce to fundamental domain
  float mirror_prod [3];
  if (r_sq < 1.) {
    vec3 v = vec3(2.*u, 1.+r_sq) / (1.-r_sq);
    int flips = 0;
    int onsides = 0; // how many times in a row we've been on the negative side of a mirror
    while (flips < 40) {
      for (int k = 0; k < 3; k++) {
        mirror_prod[k] = mprod(v, mirrors[k]);
        if (mirror_prod[k] > 0.) {
          v -= 2.*mirror_prod[k]*mirrors[k];
          flips += 1;
          onsides = 0;
        } else {
          onsides += 1;
          if (onsides >= 3) {
            // we're in the fundamental domain, on the negative side of every mirror
            
            // get the distance to the nearest mirror
            float mirror_dist = -SQRT2 * max(max(mirror_prod[0], mirror_prod[1]), mirror_prod[2]);
            
            // estimate how much of our pixel is on the negative side of the nearest mirror
            float overflow = 0.5*erfc_appx(mirror_dist / r_px);
            
            float pos_color = mod(flips, 2);
            float px_color = mix(pos_color, 1.-pos_color, overflow);
            gl_FragColor = vec4(vec3(px_color), 1.);
            return;
          }
        }
      }
    }
  }
  gl_FragColor = vec4(0.25, 0.15, 0.35, 1.);
}

void main() {
  if (antialias) main_gauss(); else main_none();
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
    self.update_resolution()
    self.program['antialias'] = False
    self.set_tiling(p, q, r)
    self.update_title()
  
  def update_title(self):
    tiling_name = 'Tiling {} {} {}'.format(*self.orders)
    aa_ind = 'antialiased' if self.program['antialias'] else 'no antialiasing'
    self.title = tiling_name + ' | ' + aa_ind
  
  def update_resolution(self):
    width, height = self.physical_size
    gloo.set_viewport(0, 0, width, height)
    self.program['resolution'] = [width, height]
    self.program['shortdim'] = min(width, height)
  
  def toggle_antialiasing(self):
    self.program['antialias'] = not self.program['antialias']
  
  def set_tiling(self, p, q, r):
    # store and display vertex orders
    self.orders = (p, q, r)
    
    # get vertex cosines
    sp = sin(pi/p)
    cp = cos(pi/p)
    cq = cos(pi/q)
    cr = cos(pi/r)
    
    # find the side normals of the fundamental triangle, scaled to unit norm
    self.program['mirrors[0]'] = (1, 0, 0)
    self.program['mirrors[1]'] = (-cp, sp, 0)
    self.program['mirrors[2]'] = (
      -cq,
      -(cp*cq + cr) / sp,
      sqrt(-1 + (cp*cp + cq*cq + cr*cr + 2*cp*cq*cr)) / sp
    )
  
  def on_draw(self, event):
    self.program.draw()
  
  def on_resize(self, event):
    self.update_resolution()
  
  def on_key_press(self, event):
    # update tiling
    p, q, r = self.orders
    if   event.text == 'j': p -= 1
    elif event.text == 'u': p += 1
    elif event.text == 'k': q -= 1
    elif event.text == 'i': q += 1
    elif event.text == 'l': r -= 1
    elif event.text == 'o': r += 1
    elif event.text == 'a': self.toggle_antialiasing()
    
    if (q*r + r*p + p*q < p*q*r):
      self.set_tiling(p, q, r)
    
    self.update_title()
    self.update()

if __name__ == '__main__' and sys.flags.interactive == 0:
  # show controls
  print('''
  uio  raise vertex orders
  jkl  lower vertex orders
  
  a    toggle antialiasing
  ''')
  
  orders = (2, 3, 7)
  TilingCanvas(*orders, size = (500, 500)).show()
  app.run()
