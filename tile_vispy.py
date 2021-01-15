# based on Anton Sherwood's programs for painting hyperbolic tilings
# <https://commons.wikimedia.org/wiki/User:Tamfang/programs>

import sys
from vispy import app, gloo
from math import sqrt, cos, sin, pi

import covering

vertex = '''
attribute vec2 position;

void main() {
  gl_Position = vec4(position, 0., 1.);
}
'''

fragment = ('''
uniform vec2 resolution;
uniform float shortdim;
uniform bool antialias;

uniform vec3 mirrors [3];
uniform mat3 shift;
uniform int p;
uniform int q;
uniform float K_a;
uniform float K_b;
uniform float cover_a [20]; /*[TEMP] should make size adjustable*/
uniform float cover_b [20]; /*[TEMP]*/
uniform int color_machine [110];

// --- complex arithmetic ---

const vec2 ZERO = vec2(0.);
const vec2 ONE  = vec2(1., 0.);
const vec2 I    = vec2(0., 1.);

//  the complex conjugate of `z`
vec2 conj(vec2 z) {
  return vec2(z.x, -z.y);
}

// the product of `z` and `w`
vec2 mul(vec2 z, vec2 w) {
  return mat2(z, conj(z).yx) * w;
}

// the reciprocal of `z`
vec2 rcp(vec2 z) {
  // 1/z = z'/(z'*z) = z'/|z|^2
  return conj(z) / dot(z, z);
}

// `z^n`
vec2 cpow(vec2 z, int n) {
  vec2 z_power = ONE;
  for (int k = 0; k < n; k++) {
    z_power = mul(z, z_power);
  }
  return z_power;
}

// the square root of `z`, from the complex arithmetic code listing in
// Appendix C of _Numerical Recipes in C_
//
// William Press, Saul Teukolsky, William Vetterling, and Brian Flannery,
// _Numerical Recipes in C_, 2nd edition. Cambridge University Press, 1992
//
vec2 csqrt(vec2 z) {
    // sqrt(0) = 0
    if (z.x == 0. && z.y == 0.) {
        return vec2(0.);
    }
    
    // calculate w
    vec2 a = abs(z);
    float w;
    if (a.x >= a.y) {
        float sl = a.y / a.x;
        w = sqrt(a.x) * sqrt(0.5*(1. + sqrt(1. + sl*sl)));
    } else {
        float sl = a.x / a.y;
        w = sqrt(a.y) * sqrt(0.5*(sl + sqrt(1. + sl*sl)));
    }
    
    // construct output
    if (z.x >= 0.) {
        return vec2(w, z.y / (2.*w));
    } else if (z.y >= 0.) {
        return vec2(z.y/(2.*w), w);
    } else {
        return -vec2(z.y/(2.*w), w);
    }
}

// --- elliptic integral of the first kind ---
//
// B. C. Carlson, "Numerical computation of real or complex elliptic integrals"
// Numerical Algorithms, vol. 10, pp. 13--26, 1995
// <doi:10.1007/BF02198293>
//
// William Press and Saul Teukolsky, "Elliptic Integrals"
// Computers in Physics, vol. 4, pp. 92--96, 1990
// <doi:10.1063/1.4822893>

const int N = 12;

const vec2 C1 = 1./24. * ONE;
const vec2 C2 = 0.1    * ONE;
const vec2 C3 = 3./44. * ONE;
const vec2 C4 = 1./14. * ONE;

vec2 RF(vec2 x, vec2 y, vec2 z) {
    for (int n = 0; n < N; n++) {
        vec2 sqrt_x = csqrt(x);
        vec2 sqrt_y = csqrt(y);
        vec2 sqrt_z = csqrt(z);
        vec2 lambda = mul(sqrt_x, sqrt_y) + mul(sqrt_y, sqrt_z) + mul(sqrt_z, sqrt_x);
        x = 0.25*(x + lambda);
        y = 0.25*(y + lambda);
        z = 0.25*(z + lambda);
    }
    vec2 avg = (x + y + z)/3.;
    vec2 off_x = x - avg;
    vec2 off_y = y - avg;
    vec2 off_z = z - avg;
    vec2 e2 = mul(off_x, off_y) - mul(off_z, off_z);
    vec2 e3 = mul(mul(off_x, off_y), off_z);
    return mul(ONE + mul(mul(C1, e2) - C2 - mul(C3, e3), e2) + mul(C4, e3), rcp(csqrt(avg)));
}

// inverse sine (Carlson 1995, equation 4.18)
vec2 casin(vec2 z) {
    vec2 z_sq = mul(z, z);
    return mul(z, RF(ONE - z_sq, ONE, ONE));
}

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

// --- covering ---

/*[TEMP] should make size adjustable*/
vec2 apply_series(float[20] series, vec2 w, int order) {
  // write cover(w) as w * deformation(w)
  vec2 deformation = vec2(0.);
  vec2 w_order = cpow(w, order);
  vec2 w_power = ONE;
  for (int n = 0; n < 20; n++) {
    deformation += series[n] * w_power;
    w_power = mul(w_order, w_power);
  }
  return mul(w, deformation);
}

vec2 cover(vec3 v) {
  vec3 v_shift = shift * v;
  if (v.z < v_shift.z) {
    // v is closer to the time axis (this comparison works because v and v_shift
    // are on the forward -1 hyperboloid)
    vec2 w = v.xy / (1. + v.z);
    vec2 s = apply_series(cover_a, w / K_a, p);
    return cpow(s, p);
  } else {
    vec2 w_shift = v_shift.xy / (1. + v_shift.z);
    vec2 s = apply_series(cover_b, w_shift / K_b, q);
    return ONE - cpow(s, q);
  }
}

// --- strip coloring ---

const float PI = 3.141592653589793;

const int NONE = 0;
const int L_HALF = 1;
const int R_HALF = 2;
const int L_WHOLE = 3;
const int R_WHOLE = 4;
const int WHOLE = 5;
const int ERROR = 6;

vec3 strip_color(vec2 z, int part, int edge) {
  // set up edge palette
  vec3 edge_palette [6];
  edge_palette[0] = vec3(1.00, 0.43, 0.00);
  edge_palette[1] = vec3(1.00, 0.87, 0.21);
  edge_palette[2] = vec3(0.92, 0.67, 0.96);
  edge_palette[3] = vec3(0.64, 0.07, 0.73);
  edge_palette[4] = vec3(0.40, 0.00, 0.29);
  edge_palette[5] = vec3(0.30, 0.11, 0.68);
  
  // get strip coordinate
  vec2 h = 8./PI * casin(z);
  h.y = abs(h.y);
  
  // draw ribbon graph
  vec3 color;
  if (h.y < 0.5) {
    // draw ribbon
    color = vec3(h.x < 0. ? 0. : 1.);
  } else {
    // draw sky
    if (
      part == L_HALF && -0.5 < h.x && h.x < 0. ||
      part == R_HALF && 0. < h.x && h.x < 0.5 ||
      part == L_WHOLE && h.x > 3.5 ||
      part == R_WHOLE && -h.x > 3.5
    ) {
      // draw edge identification tab
      color = edge_palette[edge];
    } else {
      float off = mod(h.y, 1.);
      color = (off < 0.5) ? vec3(0.267, 0.6, 0.941) : vec3(0.494, 0.698, 0.980);
    }
  }
  
  // highlight fundamental domain
  if (
    part == NONE ||
    part == L_HALF && h.x > 0. ||
    part == R_HALF && h.x < 0.
  ) {
    return mix(color, vec3(0.5), 0.8);
  } else {
    return color;
  }
}

// --- tiling ---

const float VIEW = 1.2;
const float EPS = 1e-6;
const float SQRT2 = 1.4142135623730951;

void main_none() {
  vec2 u = VIEW*(2.*gl_FragCoord.xy - resolution) / shortdim;
  float r_sq = dot(u, u);
  if (r_sq < 1.) {
    vec3 v = vec3(2.*u, 1.+r_sq) / (1.-r_sq);
    int flips = 0;
    int onsides = 0;
    int state = 1;
    while (flips < 40) {
      for (int k = 0; k < 3; k++) {
        if (mprod(mirrors[k], v) > EPS) {
          v = mreflect(v, mirrors[k]);
          flips += 1;
          onsides = 0;
          state = color_machine[5*state + k];
        } else {
          onsides += 1;
          if (onsides >= 3) {
            vec2 z = cover(v);
            vec3 color = strip_color(
              2.*z - ONE,
              color_machine[5*state + 3],
              color_machine[5*state + 4]
            );
            /*float tone = 1. / (1. + length(z - ZERO) / length(z - ONE));
            vec3 color = mix(vec3(mod(flips, 2)), vec3(1., 0.5, 0.), tone);*/
            gl_FragColor = vec4(color, 1.);
            return;
          }
        }
      }
    }
    gl_FragColor = vec4(0., 1., 0., 1.); /*[DEBUG] real axis speckles*/
    return; /*[DEBUG] real axis speckles*/
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
        if (mirror_prod[k] > EPS) {
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
    gl_FragColor = vec4(0., 1., 0., 1.); /*[DEBUG] real axis speckles*/
    return; /*[DEBUG] real axis speckles*/
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
    self.program['mirrors[0]'] = (0, 1, 0)
    self.program['mirrors[1]'] = (-sp, -cp, 0)
    self.program['mirrors[2]'] = (
      (cp*cq + cr) / sp,
      -cq,
      sqrt(-1 + (cp*cp + cq*cq + cr*cr + 2*cp*cq*cr)) / sp
    )
    
    # find the covering map to CP^1
    bel = covering.Covering(p, q, r, 20)
    self.program['shift'] = bel.shift
    self.program['p'] = bel.p
    self.program['q'] = bel.q
    self.program['K_a'] = bel.K_a;
    self.program['K_b'] = bel.K_b;
    for n in range(len(bel.cover_a)):
      self.program['cover_a[{}]'.format(n)] = bel.cover_a[n]
      self.program['cover_b[{}]'.format(n)] = bel.cover_b[n]
    
    # load the coloring machine
    NONE = 0
    L_HALF = 1
    R_HALF = 2
    L_WHOLE = 3
    R_WHOLE = 4
    WHOLE = 5
    ERROR = 6
    color_machine = [
      [ 0,  0,  0, NONE,    -1],
      [ 2, 12,  0, L_WHOLE,  0],
      [21,  3,  0, L_WHOLE,  0],
      [ 4, 21,  0, L_HALF,   1],
      [21,  5,  0, L_HALF,   1],
      [ 6, 27,  8, WHOLE,   -1],
      [21, 11,  7, WHOLE,   -1],
      [ 0,  0, 21, R_HALF,   2],
      [ 9,  0, 21, R_HALF,   1],
      [21,  0, 10, R_HALF,   1],
      [ 0,  0, 21, R_HALF,   2],
      [ 0, 21,  0, L_HALF,   2],
      [13, 21, 18, WHOLE,   -1],
      [21, 14, 17, WHOLE,   -1],
      [15, 21,  0, L_HALF,   3],
      [21, 16,  0, L_HALF,   3],
      [ 0, 21,  0, L_HALF,   2],
      [ 0,  0, 21, R_HALF,   3],
      [19,  0, 21, R_WHOLE,  4],
      [21,  0, 20, R_WHOLE,  4],
      [ 0,  0, 27, R_HALF,   3],
      [27, 27, 27, ERROR,   -1]
    ]
    for state in range(len(color_machine)):
      for input in range(5):
        self.program['color_machine[{}]'.format(5*state + input)] = color_machine[state][input];
  
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
  
  orders = (6, 3, 4)
  TilingCanvas(*orders, size = (500, 500)).show()
  app.run()
