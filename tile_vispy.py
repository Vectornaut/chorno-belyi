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

vec3 sample(vec2 coord) {
  vec2 u = VIEW*(2.*coord - resolution) / shortdim;
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
            return vec3(mod(flips, 2));
          }
        }
      }
    }
  }
  return vec3(0.25, 0.15, 0.35);
}

void main_multi() {
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

void main_linramp() {
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
    /*if (mprod(v, v) < -0.5) {
      gl_FragColor = vec4(0.2, 0., 1., 1.);
      return;
    }*/
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
            /*if (mirror_dist > r_px) {
              gl_FragColor = vec4(0., 1., 0., 1.);
              return;
            }*/
            /*if (mprod(v, v) < -1.+1e-6) {
              gl_FragColor = vec4(1., 0., 0.2, 1.);
              return;
            }*/
            
            // estimate how much of our pixel is on the negative side of the nearest mirror
            float coverage = 0.5 + 0.5*min(mirror_dist / r_px, 1.);
            
            float pos_color = mod(flips, 2);
            float px_color = mix(1.-pos_color, pos_color, coverage);
            gl_FragColor = vec4(vec3(px_color), 1.);
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
            /*if (mirror_dist > r_px) {
              gl_FragColor = vec4(0., 1., 0., 1.);
              return;
            }*/
            
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

void main_box() {
  // find screen coordinate
  vec2 u = VIEW*(2.*gl_FragCoord.xy - resolution) / shortdim;
  float r_sq = dot(u, u);
  
  // find pixel bounds, for antialiasing
  float r_px_screen = VIEW / shortdim; // the inner radius of a pixel in the Euclidean metric of the screen
  float r_px = 2.*r_px_screen / (1.-r_sq); // the approximate inner radius of our pixel in the hyperbolic metric
  vec2 dir_px = vec2(1., 0.); // the direction toward the east edge of our pixel, represented as a vector in the x, y plane
  
  // reduce to fundamental domain
  float mirror_prod [3];
  if (r_sq < 1.) {
    vec3 v = vec3(2.*u, 1.+r_sq) / (1.-r_sq);
    /*if (mprod(v, v) < -0.5) {
      gl_FragColor = vec4(0.2, 0., 1., 1.);
      return;
    }*/
    int flips = 0;
    int onsides = 0; // how many times in a row we've been on the negative side of a mirror
    while (flips < 40) {
      for (int k = 0; k < 3; k++) {
        mirror_prod[k] = mprod(v, mirrors[k]);
        if (mirror_prod[k] > 0.) {
          v -= 2.*mirror_prod[k]*mirrors[k];
          dir_px -= 2.*dot(dir_px, mirrors[k].xy)*mirrors[k].xy;
          flips += 1;
          onsides = 0;
        } else {
          onsides += 1;
          if (onsides >= 3) {
            // we're in the fundamental domain, on the negative side of every mirror
            
            // find the nearest mirror
            int near;
            if (mirror_prod[0] >= mirror_prod[1] && mirror_prod[0] >= mirror_prod[2]) {
              near = 0;
            } else if (mirror_prod[1] >= mirror_prod[2]) {
              near = 1;
            } else {
              near = 2;
            }
            
            // find the distance to the nearest mirror, in pixel radii, and the
            // the pixel direction in the frame where the mirror normal
            // direction is (1, 0)
            float rel_dist = -SQRT2 * mirror_prod[near] / r_px;
            vec2 rel_dir = abs(vec2(
              dir_px.x * mirrors[near].x + dir_px.y * mirrors[near].y,
              -dir_px.x * mirrors[near].y + dir_px.y * mirrors[near].x
            ));
            if (rel_dir.y > rel_dir.x) rel_dir = rel_dir.yx;
            rel_dir /= length(rel_dir);
            
            // estimate how much of our pixel is on the positive side of the nearest mirror
            float overflow;
            dir_px /= length(dir_px);
            if (rel_dist >= rel_dir.x + rel_dir.y) {
              overflow = 0.;
            } else if (rel_dist >= rel_dir.x - rel_dir.y) {
              float lap = rel_dir.x + rel_dir.y - rel_dist;
              overflow = lap*lap / (8.*rel_dir.x*rel_dir.y);
            } else {
              overflow = 0.5*(1. - rel_dist/rel_dir.x);
            }
            
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
  /*main_multi();*/
  /*main_linramp();*/
  main_gauss();
  /*main_box();*/
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
    # store and display vertex orders
    self.orders = (p, q, r)
    self.title = 'Tiling {} {} {}'.format(p, q, r)
    
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
    self.set_resolution()
  
  def on_key_press(self, event):
    # update tiling
    p, q, r = self.orders
    if event.text == 'j': p -= 1
    elif event.text == 'u': p += 1
    elif event.text == 'k': q -= 1
    elif event.text == 'i': q += 1
    elif event.text == 'l': r -= 1
    elif event.text == 'o': r += 1
    if (q*r + r*p + p*q < p*q*r):
      self.set_tiling(p, q, r)
      self.update()

if __name__ == '__main__' and sys.flags.interactive == 0:
  orders = (2, 3, 7)
  TilingCanvas(*orders, size = (500, 500)).show()
  app.run()
