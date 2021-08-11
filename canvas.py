# based on Anton Sherwood's programs for painting hyperbolic tilings
# <https://commons.wikimedia.org/wiki/User:Tamfang/programs>

# the step animation method is from Jonny Hyman's logistic map visualizations
# https://github.com/jonnyhyman/Chaos/blob/master/logistic_interactive.py

from PyQt5.QtCore import QTimer
from vispy import app, gloo
import vispy.util.keys as keys
from numpy import array, identity, matmul, dot, pi

from covering import Covering

vertex = '''
attribute vec2 position;

void main() {
  gl_Position = vec4(position, 0., 1.);
}
'''

fragment = ('''
// display settings
uniform vec2 resolution;
uniform float shortdim;
uniform bool show_tiling;
uniform mat3 viewpoint;

// covering map
uniform vec3 mirrors [3];
uniform mat3 shift;
uniform int p;
uniform int q;
uniform float K_a;
uniform float K_b;
uniform float cover_a [20]; /*[TEMP] should make size adjustable*/
uniform float cover_b [20]; /*[TEMP]*/

// triangle coloring
/*uniform int tri_tree [1022];*/
uniform int tri_tree [672]; /*[TEMP] Sam's MacBook can't handle large uniforms*/
uniform bool bdry_lit;

// --- complex arithmetic ---

const vec2 ZERO = vec2(0.);
const vec2 ONE  = vec2(1., 0.);
const vec2 I    = vec2(0., 1.);

//  the complex conjugate of `z`
vec2 conj(vec2 z) {
  return vec2(z.x, -z.y);
}

// multiplication by `z`
mat2 mul(vec2 z) {
    return mat2(z, conj(z).yx);
}

// the product of `z` and `w`
vec2 mul(vec2 z, vec2 w) {
    return mul(z) * w;
}

// the reciprocal of `z`
vec2 rcp(vec2 z) {
  // 1/z = z'/(z'*z) = z'/|z|^2
  return conj(z) / dot(z, z);
}

// `z^n`
vec2 cpow(vec2 z, int n) {
  vec2 z_power = ONE;
  mat2 mul_z = mul(z);
  for (int k = 0; k < n; k++) {
    z_power = mul_z * z_power;
  }
  return z_power;
}

// --- automatic differentiation ---

// a cjet is a 1-jet of a holomorphic map C --> C,
// with image point `pt` and derivative `push`

struct cjet {
  vec2 pt;
  mat2 push;
};

cjet add(cjet f, cjet g) {
  return cjet(f.pt + g.pt, f.push + g.push);
}

cjet add(cjet f, vec2 c) {
  return cjet(f.pt + c, f.push);
}

cjet scale(float a, cjet f) {
  return cjet(a*f.pt, a*f.push);
}

cjet mul(cjet f, cjet g) {
  mat2 mul_f = mul(f.pt);
  mat2 mul_g = mul(g.pt);
  return cjet(
    mul_f*g.pt,
    f.push*mul_g + mul_f*g.push
  );
}

// `f^n`
cjet cpow(cjet f, int n) {
  cjet f_power = cjet(ONE, mat2(0.));
  for (int k = 0; k < n; k++) {
    f_power = mul(f, f_power);
  }
  return f_power;
}

// --- complex square root ---
//
// from the complex arithmetic code listing in Appendix C of _Numerical Recipes_
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
cjet casin(cjet z) {
  mat2 mul_z = mul(z.pt);
  vec2 z_sq = mul_z * z.pt;
  return cjet(
    mul_z * RF(ONE - z_sq, ONE, ONE),
    mul(rcp(csqrt(ONE - z_sq))) * z.push
  );
}

// --- minkowski geometry ---

// the minkowski bilinear form
float mprod(vec3 v, vec3 w) {
  return dot(v.xy, w.xy) - v.z*w.z;
}

// --- covering ---

/*[TEMP] should make size adjustable*/
cjet apply_series(float[20] series, cjet w, int order) {
  // write cover(w) as w * deformation(w)
  cjet deformation = cjet(ZERO, mat2(0.));
  cjet w_order = cpow(w, order);
  cjet w_power = cjet(ONE, mat2(0.));
  for (int n = 0; n < 20; n++) {
    deformation = add(deformation, scale(series[n], w_power));
    w_power = mul(w_order, w_power);
  }
  return mul(w, deformation);
}

// if `v` came from somewhere else in the hyperbolic plane, we can factor in
// the conformal scale factor of the mobius transformation that brought it here
// by passing its original `r_sq` as `r_sq_orig`
cjet cover(vec3 v, float r_sq_orig) {
  vec3 v_shift = shift * v;
  if (v.z < v_shift.z) {
    // v is closer to the time axis (this comparison works because v and v_shift
    // are on the forward -1 hyperboloid)
    
    // project to the Poincare disk, and find the conformal scale factor of the
    // Mobius transformation that brought us here
    vec2 w_pt = v.xy / (1. + v.z);
    float r_sq = dot(w_pt, w_pt);
    float pre_scaling = (1-r_sq) / (1-r_sq_orig);
    
    // apply the covering map
    cjet w = cjet(w_pt, mat2(pre_scaling));
    cjet s = apply_series(cover_a, scale(1./K_a, w), p);
    return cpow(s, p);
  } else {
    // project to the Poincare disk, and find the conformal scale factor of the
    // Mobius transformation that brought us here
    vec2 w_shift_pt = v_shift.xy / (1. + v_shift.z);
    float r_sq = dot(w_shift_pt, w_shift_pt);
    float pre_scaling = (1-r_sq) / (1-r_sq_orig);
    
    // apply the covering map
    cjet w_shift = cjet(w_shift_pt, mat2(pre_scaling));
    cjet s = apply_series(cover_b, scale(1./K_b, w_shift), q);
    return add(scale(-1., cpow(s, q)), ONE);
  }
}

// --- pixel sampling ---

const float A1 = 0.278393;
const float A2 = 0.230389;
const float A3 = 0.000972;
const float A4 = 0.078108;

// Abramowitz and Stegun, equation 7.1.27
float erfc_appx(float t) {
  float r = abs(t);
  float p = 1. + A1*(r + A2*(r + A3*(r + A4*r)));
  float p_sq = p*p;
  float erfc_r = 1. / (p_sq*p_sq);
  return t < 0. ? (2. - erfc_r) : erfc_r;
}

// how much of a pixel's sampling distribution falls on the negative side of an
// edge. `disp` is the pixel's displacement from the edge in pattern space
float neg_part(float pattern_disp, float scaling, float r_px) {
  // find the displacement to the edge in the screen tangent space
  float screen_disp = pattern_disp / scaling;
  
  // integrate our pixel's sampling distribution on the screen tangent space to
  // find out how much of the pixel falls on the negative side of the edge
  return 0.5*erfc_appx(screen_disp / r_px);
}

// find the color of a pixel near an edge between two colored regions.
// `neg` and `pos` are the colors on the negative and positive sides of the
// edge. `disp` is the displacement from the edge

float edge_mix(float neg, float pos, float pattern_disp, float scaling, float r_px) {
  return mix(pos, neg, neg_part(pattern_disp, scaling, r_px));
}

vec3 edge_mix(vec3 neg, vec3 pos, float pattern_disp, float scaling, float r_px) {
  return mix(pos, neg, neg_part(pattern_disp, scaling, r_px));
}

// how much of a pixel's sampling distribution falls on a thickened line.
// `width` is the line thickness, in pixels. `pattern_disp` is the pixel's
// displacement from the line in pattern space
float line_part(float width, float pattern_disp, float scaling, float r_px) {
  // find the displacement to the edge in the screen tangent space
  float screen_disp = pattern_disp / scaling;
  float screen_disp_px = screen_disp / r_px;
  
  // integrate our pixel's sampling distribution on the screen tangent space to
  // find out how much of the pixel falls within `width/2` of the line
  float lower = erfc_appx(screen_disp_px - 0.5*width);
  float upper = erfc_appx(screen_disp_px + 0.5*width);
  return 0.5*(lower - upper);
}

vec3 line_mix(vec3 stroke, vec3 bg, float width, float pattern_disp, float scaling, float r_px) {
  return mix(bg, stroke, line_part(width, pattern_disp, scaling, r_px));
}

vec4 line_mix(vec4 stroke, vec4 bg, float width, float pattern_disp, float scaling, float r_px) {
  return mix(bg, stroke, line_part(width, pattern_disp, scaling, r_px));
}

// --- strip coloring ---

const float PI = 3.141592653589793;

vec3 strip_color(
  cjet z, float proj_scaling, float r_px,
  float[3] mirror_prod, int twin_k,
  bvec2 lit, bvec2 twin_lit,
  ivec2 outer_trim, ivec2 twin_trim, int inner_trim
) {
  // set up edge palette
  vec3 edge_palette [6];
  edge_palette[0] = vec3(231.,  48.,  44.) / 255.;
  edge_palette[1] = vec3(250., 144.,   4.) / 255.;
  edge_palette[2] = vec3(255., 242.,   0.) / 255.;
  edge_palette[3] = vec3( 40., 184., 242.) / 255.;
  edge_palette[4] = vec3(128.,  90., 244.) / 255.;
  edge_palette[5] = vec3( 58.,  39., 178.) / 255.;
  
  // get strip coordinate and side
  cjet h = scale(8./PI, casin(z));
  int side = h.pt.x < 0. ? 0 : 1;
  
  // reflect strip coordinate into non-negative quadrant
  vec2 h_pos = abs(h.pt);
  
  // find the conformal scale factor of the map from screen space to pattern
  // space
  float scaling = length(h.push[0]);
  
  // draw ribbon graph
  vec3 ribbon = vec3(edge_mix(1, 0, h.pt.x, scaling, r_px));
  vec3 sky = mix(vec3(0.8, 0.9, 1.0), vec3(0.6, 0.75, 0.9), 0.5 / max(h_pos.y, 0.5));
  
  // dim unlit half-triangles. recall that when -mirror_prod[k] is small, it
  // approximates the distance to mirror[k]
  vec3 shadow = vec3(0.4, 0.45, 0.5);
  float dimness = edge_mix(float(!lit[0]), float(!lit[1]), h.pt.x, scaling, r_px);
  dimness = edge_mix(float(!twin_lit[side]), dimness, -mirror_prod[twin_k], proj_scaling, r_px);
  ribbon = mix(ribbon, shadow, 0.8*dimness);
  sky = mix(sky, shadow, 0.8*dimness);
  
  // draw inner trim
  vec3 trimmed = sky;
  if (inner_trim > 0) {
    trimmed = line_mix(edge_palette[inner_trim-1], sky, 10, h.pt.x, scaling, r_px);
  }
  
  // draw outer trim. recall that when -mirror_prod[k] is small, it
  // approximates the distance to mirror[k]
  int active_trim = int(max(outer_trim[side], twin_trim[side]));
  if (active_trim > 0) {
    trimmed = line_mix(edge_palette[active_trim-1], trimmed, 10, -mirror_prod[1+side], proj_scaling, r_px);
  }
  
  // combine ribbon and trim
  return edge_mix(ribbon, trimmed, h_pos.y - 0.5, scaling, r_px);
}

// --- tiling ---

const float VIEW = 1.02;
const float EPS = 1e-6;
const float TWIN_EPS = 1e-5;

void main_dessin() {
  // find screen coordinate
  float r_px = VIEW / shortdim; // the inner radius of a pixel in the Euclidean metric of the screen
  vec2 u = r_px * (2.*gl_FragCoord.xy - resolution);
  float r_sq = dot(u, u);
  
  // set boundary color
  vec3 bdry_color = vec3(0.8, 0.9, 1.0);
  if (!bdry_lit) {
    bdry_color = mix(bdry_color, vec3(0.4, 0.45, 0.5), 0.8);
  }
  
  // reduce to fundamental domain
  if (r_sq < 1.) {
    vec3 v = viewpoint * vec3(2.*u, 1.+r_sq) / (1.-r_sq);
    int flips = 0;
    int onsides = 0; // how many times in a row we've been on the negative side of a mirror
    int index = 1;
    
    // for area sampling across triangle boundaries, we follow a twin point on
    // the other side of the nearest mirror. (strictly speaking, or kludgey
    // algorithm doesn't always find the nearest mirror. if two mirrors'
    // minkowski products with us are smaler than TWIN_EPS, we could get a twin
    // across either one
    int twin = 0; // our twin's triangle tree index
    
    // to identify the twin point during the final onsides check, we also save
    // the last minkowski product with each mirror. since the minkowski metric
    // induces the hyperbolic metric of curvature -1 on the forward hyperboloid,
    // a small -mirror_prod[k] approximates the distance to mirror k
    float mirror_prod [3];
    
    // the minkowski product with the nearest mirror we've flipped across so
    // far. to set the initial value, we note that the inradius of a hyperbolic
    // triangle is at most 2*log(3), so the product with the nearest mirror
    // can be at most
    //
    //   sqrt(1 - cosh(2*log(3))) = sqrt(1 - 5/9) = 2/3
    //
    float twin_prod = 0.6667;
    
    while (flips < 60) {
      for (int k = 0; k < 3; k++) {
        mirror_prod[k] = mprod(v, mirrors[k]);
        if (mirror_prod[k] > EPS) {
          if (mirror_prod[k] < twin_prod - TWIN_EPS) {
            // the twin that didn't flip here would be closer than our current
            // twin, so start following that twin instead
            twin = index;
            twin_prod = mirror_prod[k];
          } else {
            twin = tri_tree[8*twin + k];
          }
          
          // reflect across mirror k
          v -= 2.*mirror_prod[k]*mirrors[k];
          flips += 1;
          onsides = 0;
          index = tri_tree[8*index + k];
        } else {
          onsides += 1;
          
          if (-mirror_prod[k] < twin_prod - TWIN_EPS) {
            // the twin that did flip here would be closer than our current twin
            // twin, so start following that twin instead
            twin = tri_tree[8*index + k];
            twin_prod = -mirror_prod[k];
          }
          
          if (onsides >= 3) {
            // we're in the fundamental domain, on the negative side of every mirror
            
            // find the mirror between us and our twin. along the way, we'll
            // see which side of the strip we're closest to hyperbolically
            int twin_k = mirror_prod[1] > mirror_prod[0] ? 1 : 0;
            if (mirror_prod[2] > mirror_prod[twin_k]) twin_k = 2;
            
            // fetch coloring data
            bvec2 lit = bvec2(tri_tree[8*index + 3], tri_tree[8*index + 4]);
            ivec2 outer_trim = ivec2(tri_tree[8*index + 5], tri_tree[8*index + 6]);
            int inner_trim = tri_tree[8*index + 7];
            bvec2 twin_lit = bvec2(tri_tree[8*twin + 3], tri_tree[8*twin + 4]);
            ivec2 twin_trim = ivec2(tri_tree[8*twin + 5], tri_tree[8*twin + 6]);
            
            // find the conformal scale factor of the Poincare projection
            float proj_scaling = 2. / (1.-r_sq);
            
            // sample the dessin coloring
            cjet z = cover(v, r_sq);
            vec3 color = strip_color(
              add(scale(2., z), -ONE), proj_scaling, r_px,
              mirror_prod, twin_k,
              lit, twin_lit,
              outer_trim, twin_trim, inner_trim
            );
            
            // area-sample the disk boundary
            color = line_mix(bdry_color, color, 2, r_sq - 1., 2., r_px);
            gl_FragColor = vec4(color, 1.);
            return;
          }
        }
      }
    }
    //gl_FragColor = vec4(0., 1., 0., 1.); /*[DEBUG] real axis speckles*/
    //return; /*[DEBUG] real axis speckles*/
  }
  gl_FragColor = line_mix(vec4(bdry_color, 1.), vec4(0.), 2, r_sq - 1., 2., r_px);
}

void main_tiling() {
  // find screen coordinate
  vec2 u = VIEW*(2.*gl_FragCoord.xy - resolution) / shortdim;
  float r_sq = dot(u, u);
  
  // find pixel radius, for area sampling
  float r_px_screen = VIEW / shortdim; // the inner radius of a pixel in the Euclidean metric of the screen
  float r_px = 2.*r_px_screen / (1.-r_sq); // the approximate inner radius of our pixel in the hyperbolic metric
  
  // reduce to fundamental domain
  if (r_sq < 1.) {
    vec3 v = viewpoint * vec3(2.*u, 1.+r_sq) / (1.-r_sq);
    int flips = 0;
    int onsides = 0; // how many times in a row we've been on the negative side of a mirror
    
    // for area sampling, we save the last minkowski product with each mirror
    float mirror_prod [3];
    
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
            
            // estimate the hyperbolic distance to the nearest mirror, using the
            // fact that the minkowski metric induces the hyperbolic metric of
            // curvature -1 on the forward hyperboloid
            float mirror_dist = -max(max(mirror_prod[0], mirror_prod[1]), mirror_prod[2]);
            
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
  if (show_tiling) main_tiling(); else main_dessin();
}
''')

def tri_tree_key(index, attr):
  return 'tri_tree[{}]'.format(8*index + attr)

class DomainCanvas(app.Canvas):
  edge_palette = ['#e7302c', '#fa9004', '#fff200', '#28b8f2', '#805af4', '#3a27b2']
  
  def __init__(self, p, q, r, lit=True, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.program = gloo.Program(vertex, fragment, count = 6) # we'll always send 6 vertices
    
    # draw a rectangle that covers the canvas
    self.program['position'] = [
      (-1, -1), (-1, 1), (1, 1), # northwest triangle
      (-1, -1), (1, 1), (1, -1)  # southeast triangle
    ]
    
    # initialize resolution and tiling display mode
    self.update_resolution()
    self.program['show_tiling'] = False
    
    # initialize step animation
    self.step = lambda t: identity(3)
    self.step_frame_rate = 30
    self.step_frame_interval = 1000 / self.step_frame_rate
    self.step_speed = 6
    self.step_nframes = 1
    self.step_timer = QTimer()
    self.step_timer.timeout.connect(self.animate_step)
    self.step_end = 0
    
    # initialize tiling and fundamental domain
    self.set_tiling(p, q, r)
    self.domain = None
    
    # initialize triangle-coloring tree
    #for m in range(1022):
    for m in range(672): ##[TEMP] Sam's MacBook can't handle large uniforms
      self.program['tri_tree[{}]'.format(m)] = 0
    if lit:
      self.load_empty_tree(True)
    self.program['bdry_lit'] = lit
    
    # initialize work state
    self.paint_display = None
    self.selection_display = None
    self.set_working(False)
    self.set_paint_color(1)
    self.set_selection(None)
  
  def update_resolution(self, size=None):
    width, height = size if size else self.physical_size
    gloo.set_viewport(0, 0, width, height)
    self.program['resolution'] = [width, height]
    self.program['shortdim'] = min(width, height)
  
  def get_step_frame(self):
    return self._step_frame
  
  def set_step_frame(self, t):
    self._step_frame = t
    self.viewpoint = matmul(self.old_viewpoint, self.step(self.step_end * t/self.step_nframes))
    self.program['viewpoint'] = self.viewpoint.transpose()
    self.update()
  
  step_frame = property(get_step_frame, set_step_frame)
  
  def time_step(self, length):
    self.step_nframes = round(length / self.step_speed * self.step_frame_rate)
  
  def animate_step(self):
    self.step_frame += 1
    if self.step_frame >= self.step_nframes:
      self.step_timer.stop()
  
  def reset_viewpoint(self):
    # reset step animation
    self.step_timer.stop()
    self.old_viewpoint = identity(3)
    self.new_viewpoint = identity(3)
    self.step_frame = 0
    
    # check paths
    self.viewpoint_color = 0
    self.path_right = True
    self.path_left = self.covering.orders[self.viewpoint_color] % 2 == 0
  
  def walk_right(self):
    if self.path_right:
      # animate step
      self.step_timer.stop()
      self.step = self.covering.shift
      self.old_viewpoint = self.new_viewpoint
      self.new_viewpoint = matmul(self.old_viewpoint, self.covering.shift_ab)
      self.step_end = -1
      self.step_frame = 0
      self.time_step(self.covering.dist_ab)
      self.step_timer.start(self.step_frame_interval)
      
      # update paths
      self.viewpoint_color = 1 - self.viewpoint_color
      self.path_right = self.covering.orders[self.viewpoint_color] % 2 == 0
      self.path_left = True
  
  def walk_left(self):
    if self.path_left:
      # animate step
      self.step_timer.stop()
      self.step = self.covering.shift
      self.old_viewpoint = self.new_viewpoint
      self.new_viewpoint = matmul(self.old_viewpoint, self.covering.shift_ba)
      self.step_end = 1
      self.step_frame = 0
      self.time_step(self.covering.dist_ab)
      self.step_timer.start(self.step_frame_interval)
      
      # update paths
      self.viewpoint_color = 1 - self.viewpoint_color
      self.path_right = True
      self.path_left = self.covering.orders[self.viewpoint_color] % 2 == 0
  
  # `dir` should be 1 or -1 (for couterclockwise or clockwise turns)
  def turn(self, dir):
    # animate step
    self.step_timer.stop()
    self.step = lambda t: self.covering.rot(self.viewpoint_color, t)
    self.old_viewpoint = self.new_viewpoint
    self.new_viewpoint = matmul(self.old_viewpoint, self.step(dir/2))
    self.step_end = dir/2
    self.step_frame = 0
    self.time_step(pi/self.covering.orders[self.viewpoint_color])
    self.step_timer.start(self.step_frame_interval)
    
    # update paths
    self.path_right = not self.path_right
    self.path_left = not self.path_left
  
  def set_covering(self, covering):
    self.covering = covering
    for k in range(3):
      self.program['mirrors[{}]'.format(k)] = covering.mirrors[k]
    self.program['shift'] = covering.shift_ba.transpose()
    self.program['p'] = covering.p
    self.program['q'] = covering.q
    self.program['K_a'] = covering.K_a;
    self.program['K_b'] = covering.K_b;
    for n in range(len(covering.cover_a)):
      self.program['cover_a[{}]'.format(n)] = covering.cover_a[n]
      self.program['cover_b[{}]'.format(n)] = covering.cover_b[n]
    self.reset_viewpoint()
  
  def set_tiling(self, p, q, r):
    self.set_covering(Covering(p, q, r, 20)) ##[TEMP] should make size adjustable
  
  def load_empty_tree(self, lit=False):
    if lit:
      for attr in range(3):
        self.program[tri_tree_key(1, attr)] = 1
      for side in range(2):
        self.program[tri_tree_key(1, 3 + side)] = True
        self.program[tri_tree_key(1, 5 + side)] = 0
      self.program[tri_tree_key(1, 7)] = 0
    else:
      for attr in range(7):
        self.program[tri_tree_key(1, attr)] = 0
    
    self.program['bdry_lit'] = lit;
  
  def load_tri_tree(self, tree):
    for tri in tree.flatten(1):
      for k in range(3):
        if tri.children[k] != None:
          self.program[tri_tree_key(tri.index, k)] = tri.children[k].index
        else:
          self.program[tri_tree_key(tri.index, k)] = 0
      for side in range(2):
        self.program[tri_tree_key(tri.index, 3 + side)] = tri.lit[side]
        self.program[tri_tree_key(tri.index, 5 + side)] = tri.outer_trim[side]
      self.program[tri_tree_key(tri.index, 7)] = tri.inner_trim
    
    self.program['bdry_lit'] = False;
  
  def set_domain(self, domain, lit=False, working=None):
    self.domain = domain
    if domain:
      self.load_tri_tree(domain.tree)
      if domain.orders != self.covering.orders:
        self.set_tiling(*domain.orders)
    else:
      self.load_empty_tree(lit)
    self.update()
    
    if domain == None:
      self.set_working(False)
    elif working != None:
      self.set_working(working)
    self.set_selection(None)
  
  def on_draw(self, event):
    self.program.draw()
  
  def on_resize(self, event):
    self.update_resolution()
  
  def on_key_press(self, event):
    # change viewpoint
    if event.key == keys.RIGHT:
      self.walk_right()
    elif event.key == keys.LEFT:
      self.walk_left()
    elif event.key == keys.UP:
      self.turn(1)
    elif event.key == keys.DOWN:
      self.turn(-1)
    
    # update coloring
    side = None
    lit = None
    outer_trim = None
    inner_trim = None
    if event.key == keys.SPACE:
      self.program['show_tiling'] = not self.program['show_tiling']
      self.update()
    elif self.working:
      if event.key == 'c':
        lit = (True, True)
        outer_trim = (0, 0)
        inner_trim = 0
      elif event.key == 'x':
        side = self.selection_side
        lit = True
        inner_trim = self.paint_color
      elif event.key == 's':
        side = self.selection_side
        lit = True
        outer_trim = self.paint_color
      elif event.key == 'z':
        lit = False
      elif event.text.isdigit():
        self.set_paint_color(int(event.text))
      
      if lit != None:
        if lit == False:
          self.domain.tree.drop(self.selection)
        else:
          self.domain.tree.store(self.selection, side, lit, outer_trim, inner_trim)
        self.load_tri_tree(self.domain.tree)
        self.update()
  
  def on_mouse_release(self, event):
    # find screen coordinate
    VIEW = 1.02
    u = VIEW*(2*array(event.pos) - self.program['resolution']) / self.program['shortdim']
    
    # get address
    r_sq = dot(u, u)
    if r_sq <= 1:
      v = array([2*u[0], -2*u[1], 1+r_sq]) / (1-r_sq)
      self.set_selection(*self.covering.address(v))
    else:
      self.set_selection(None)
  
  def set_working(self, working):
    self.working = working
    if working: self.set_paint_color()
    elif self.paint_display: self.paint_display.setText(None)
  
  def set_paint_color(self, color=None):
    if color != None: self.paint_color = color
    if self.paint_display:
      self.paint_display.setText(str(self.paint_color))
      if self.paint_color <= len(self.edge_palette):
        textcolor = 'black' if self.paint_color == 3 else 'white'
        bgcolor = self.edge_palette[self.paint_color-1]
      else:
        textcolor = 'black'
        bgcolor = 'none'
      self.paint_display.setStyleSheet(
        'color: {}; background-color: {};'.format(textcolor, bgcolor)
      )
  
  def set_selection(self, address, side=None):
    self.selection = address
    self.selection_side = side
    if self.selection_display:
      if address == None:
        self.selection_display.setText(None)
      else:
        self.selection_display.setText(
          'Side {} of triangle {}'.format(self.selection_side, self.selection)
        )
  
  def render(self):
    self.set_current()
    size = (400, 400)
    fbo = gloo.FrameBuffer(
      color=gloo.RenderBuffer(size[::-1]),
      depth=gloo.RenderBuffer(size[::-1])
    )
    
    try:
      self.update_resolution(size)
      fbo.activate()
      self.program.draw()
      return fbo.read()
    finally:
      fbo.deactivate()
      self.update_resolution()
