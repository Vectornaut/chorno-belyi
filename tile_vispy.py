# based on Anton Sherwood's programs for painting hyperbolic tilings
# <https://commons.wikimedia.org/wiki/User:Tamfang/programs>

import sys, os, re, json
import PyQt5.QtWidgets as qt
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QValidator
from vispy import app, gloo
import vispy.util.keys as keys
from math import sqrt, cos, sin, pi, floor
from numpy import array, dot

import covering
import triangle_tree
from triangle_tree import *

app.use_app(backend_name = 'PyQt5', call_reuse = True)

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
uniform int tri_tree [1000];

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
  // find screen coordinate
  vec2 u = VIEW*(2.*gl_FragCoord.xy - resolution) / shortdim;
  float r_sq = dot(u, u);
  
  // reduce to fundamental domain
  if (r_sq < 1.) {
    vec3 v = vec3(2.*u, 1.+r_sq) / (1.-r_sq);
    int flips = 0;
    int onsides = 0;
    int index = 1;
    while (flips < 40) {
      for (int k = 0; k < 3; k++) {
        if (mprod(mirrors[k], v) > EPS) {
          v = mreflect(v, mirrors[k]);
          flips += 1;
          onsides = 0;
          index = tri_tree[5*index + k];
        } else {
          onsides += 1;
          if (onsides >= 3) {
            vec2 z = cover(v);
            vec3 color = strip_color(
              2.*z - ONE,
              tri_tree[5*index + 3],
              tri_tree[5*index + 4]
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

# the minkowski bilinear form
def mprod(v, w):
  return dot(v[:-1], w[:-1]) - v[-1]*w[-1]

def tri_tree_key(index, attr):
  return 'tri_tree[{}]'.format(5*index + attr)

class TilingCanvas(app.Canvas):
  edge_palette = [
    '#ff6e00',
    '#ffde36',
    '#ebabf5',
    '#a312ba',
    '#66004a',
    '#4c1cad'
  ]
  
  def __init__(self, p, q, r, highlight=WHOLE, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.program = gloo.Program(vertex, fragment, count = 6) # we'll always send 6 vertices
    
    # draw a rectangle that covers the canvas
    self.program['position'] = [
      (-1, -1), (-1, 1), (1, 1), # northwest triangle
      (-1, -1), (1, 1), (1, -1)  # southeast triangle
    ]
    
    # initialize resolution and antialiasing mode
    self.update_resolution()
    self.program['antialias'] = False
    
    # initialize tiling and fundamental domain
    self.set_tiling(p, q, r)
    self.domain = None
    
    # initialize triangle-coloring tree
    for m in range(1000):
      self.program['tri_tree[{}]'.format(m)] = 0
    if highlight == WHOLE:
      self.load_empty_tree(WHOLE)
    
    # initialize work state
    self.paint_display = None
    self.selection_display = None
    self.set_working(False)
    self.set_paint_color(0)
    self.set_selection(None)
  
  def update_resolution(self):
    width, height = self.physical_size
    gloo.set_viewport(0, 0, width, height)
    self.program['resolution'] = [width, height]
    self.program['shortdim'] = min(width, height)
  
  def toggle_antialiasing(self):
    self.program['antialias'] = not self.program['antialias']
  
  def set_tiling(self, p, q, r):
    # save vertex orders
    self.orders = (p, q, r)
    
    # get vertex cosines
    sp = sin(pi/p)
    cp = cos(pi/p)
    cq = cos(pi/q)
    cr = cos(pi/r)
    
    # find the side normals of the fundamental triangle, scaled to unit norm
    self.mirrors = [
      array([0, 1, 0]),
      array([-sp, -cp, 0]),
      array([
        (cp*cq + cr) / sp,
        -cq,
        sqrt(-1 + (cp*cp + cq*cq + cr*cr + 2*cp*cq*cr)) / sp
      ])
    ]
    for k in range(3):
      self.program['mirrors[{}]'.format(k)] = self.mirrors[k]
    
    # find the covering map to CP^1
    self.bel = covering.Covering(p, q, r, 20)
    self.program['shift'] = self.bel.shift
    self.program['p'] = self.bel.p
    self.program['q'] = self.bel.q
    self.program['K_a'] = self.bel.K_a;
    self.program['K_b'] = self.bel.K_b;
    for n in range(len(self.bel.cover_a)):
      self.program['cover_a[{}]'.format(n)] = self.bel.cover_a[n]
      self.program['cover_b[{}]'.format(n)] = self.bel.cover_b[n]
  
  def load_empty_tree(self, highlight=NONE):
    if highlight == WHOLE:
      for attr in range(3):
        self.program[tri_tree_key(1, attr)] = 1
      self.program[tri_tree_key(1, 3)] = WHOLE
      self.program[tri_tree_key(1, 4)] = 0
    else:
      for attr in range(5):
        self.program[tri_tree_key(1, attr)] = 0
  
  def load_tri_tree(self, tree):
    for tri in tree.flatten(1):
      for k in range(3):
        if tri.children[k] != None:
          self.program[tri_tree_key(tri.index, k)] = tri.children[k].index
        else:
          self.program[tri_tree_key(tri.index, k)] = 0
      self.program[tri_tree_key(tri.index, 3)] = tri.highlight
      self.program[tri_tree_key(tri.index, 4)] = tri.color
  
  def set_domain(self, domain, highlight=NONE, working=None):
    self.domain = domain
    if domain:
      self.load_tri_tree(domain.tree)
      if domain.orders != self.orders:
        self.set_tiling(*domain.orders)
    else:
      self.load_empty_tree(highlight)
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
    # update coloring
    highlight = None
    color = None
    if event.key == ';':
      self.toggle_antialiasing()
      self.update()
    elif self.working:
      if event.key == 'c': highlight = triangle_tree.WHOLE
      elif event.key == 'x': highlight = L_HALF + self.selection_side
      elif event.key == 's': highlight = L_WHOLE + self.selection_side
      elif event.key == 'z': highlight = NONE
      elif event.text.isdigit(): self.set_paint_color(int(event.text))
      
      if self.domain and highlight != None:
        if highlight == NONE:
          self.domain.tree.drop(self.selection)
        elif highlight == WHOLE:
          self.domain.tree.store(self.selection, highlight)
        else:
          self.domain.tree.store(self.selection, highlight, self.paint_color)
        self.load_tri_tree(self.domain.tree)
        self.update()
  
  def on_mouse_release(self, event):
    # find screen coordinate
    VIEW = 1.2
    u = VIEW*(2*array(event.pos) - self.program['resolution']) / self.program['shortdim']
    r_sq = dot(u, u)
    
    # reduce to fundamental domain
    EPS = 1e-6
    if r_sq <= 1:
      v = array([2*u[0], -2*u[1], 1+r_sq]) / (1-r_sq)
      address = []
      onsides = 0 # how many times in a row we've been in the desired half-plane
      while len(address) < 40:
        for k in range(3):
          sep = mprod(v, self.mirrors[k])
          if sep > EPS:
            v -= 2*sep*self.mirrors[k]
            address += [k]
            onsides = 0
          else:
            onsides += 1
            if onsides >= 3:
              # save the address of the selected triangle
              z = self.bel.apply(v)
              self.set_selection(address, 0 if z.real < 0.5 else 1)
              return
    self.set_selection(None)
  
  def set_working(self, working):
    self.working = working
    if working: self.set_paint_color()
    elif self.paint_display: self.paint_display.setText(None)
  
  def set_paint_color(self, color=None):
    if color != None: self.paint_color = color
    if self.paint_display:
      self.paint_display.setText(str(self.paint_color))
      if self.paint_color < len(self.edge_palette):
        textcolor = 'black' if self.paint_color < 3 else 'white'
        bgcolor = self.edge_palette[self.paint_color]
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

class DessinControlPanel(qt.QWidget):
  def __init__(self, canvas, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    # store a pointer to the TilingCanvas this panel controls
    self.canvas = canvas
  
  def showing(self):
    return (
      hasattr(self.parentWidget(), 'currentWidget')
      and self == self.parentWidget().currentWidget()
    )
  
  def change_controls(self):
    if self.showing():
      self.take_the_canvas()
  
  def take_the_canvas(self):
    pass

class TilingPanel(DessinControlPanel):
  def __init__(self, canvas, *args, **kwargs):
    super().__init__(canvas, *args, **kwargs)
    self.setLayout(qt.QHBoxLayout())
    
    # add order spinners
    self.order_spinners = []
    for n in canvas.orders:
      spinner = qt.QSpinBox()
      spinner.setValue(n)
      spinner.valueChanged.connect(self.change_controls)
      self.layout().addWidget(spinner)
      self.order_spinners.append(spinner)
    self.set_minimums()
  
  # set the spinner minimums so that any allowed change to a single spinner will
  # keep the tiling hyperbolic. the hyperbolicity condition is
  #
  #   p*q*r - q*r - r*p - p*q > 0
  #
  # for vertex orders p, q, r
  def set_minimums(self):
    # to keep the tiling hyperbolic, each vertex order has to stay above
    #
    #   m*n / (m*n - m - n),
    #
    # where `m` and `n` are the orders of the other two vertices. we set its
    # minimum to the smallest integer greater than this value. the `floor` in
    # our implementation is safe for integer ratios because IEEE floating-point
    # division is exact for all integers from 0 to 2^(# significand bits)
    #
    #   Daniel Lemire, "Fast exact integer divisions using floating-point operations"
    #   https://lemire.me/blog/2017/11/16/fast-exact-integer-divisions-using-floating-point-operations/
    #
    orders = [spinner.value() for spinner in self.order_spinners]
    for k in range(3):
      m = orders[(k+1)%3]
      n = orders[(k+2)%3]
      self.order_spinners[k].setMinimum(floor(1 + m*n / (m*n - m - n)))
  
  def change_controls(self):
    super().change_controls()
    self.set_minimums()
  
  def take_the_canvas(self):
    self.canvas.load_empty_tree(WHOLE)
    self.canvas.set_tiling(*[
      spinner.value()
      for spinner in self.order_spinners
    ])
    self.canvas.set_working(False)
    self.canvas.set_selection(None)
    self.canvas.update()

class PermutationValidator(QValidator):
  pmt_format = QRegExp(r'(\((\d+,)*\d+\))+')
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, *kwargs)
  
  def validate(self, input, pos):
    if self.pmt_format.exactMatch(input):
      return (self.Acceptable, input, pos)
    else:
      return (self.Intermediate, input, pos)

class WorkingPanel(DessinControlPanel):
  def __init__(self, canvas, *args, **kwargs):
    super().__init__(canvas, *args, **kwargs)
    self.setLayout(qt.QVBoxLayout())
    
    # start list of working domains
    self.domains = []
    
    # add domain entry bar
    entry_bar = qt.QWidget()
    entry_bar.setLayout(qt.QHBoxLayout())
    self.pmt_fields = []
    pmt_validator = PermutationValidator()
    for n in range(3):
      field = qt.QLineEdit()
      field.setValidator(pmt_validator)
      field.textChanged.connect(self.check_entry_format)
      field.returnPressed.connect(self.new_domain)
      self.pmt_fields.append(field)
      entry_bar.layout().addWidget(field)
    self.orbit_field = qt.QLineEdit()
    self.orbit_field.setMaximumWidth(30)
    self.orbit_field.textChanged.connect(self.check_entry_format)
    self.orbit_field.returnPressed.connect(self.new_domain)
    self.tag_field = qt.QLineEdit()
    self.tag_field.returnPressed.connect(self.new_domain)
    self.new_button = qt.QPushButton('New')
    self.new_button.setEnabled(False)
    self.new_button.clicked.connect(self.new_domain)
    entry_bar.layout().addWidget(self.orbit_field)
    entry_bar.layout().addWidget(self.tag_field)
    entry_bar.layout().addWidget(self.new_button)
    self.layout().addWidget(entry_bar)
    
    # add domain chooser bar
    chooser_bar = qt.QWidget()
    chooser_bar.setLayout(qt.QHBoxLayout())
    self.save_button = qt.QPushButton('Save')
    self.save_button.setEnabled(False)
    self.domain_box = qt.QComboBox()
    self.save_button.clicked.connect(self.save_domain)
    self.domain_box.currentTextChanged.connect(self.change_controls)
    self.domain_box.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)
    chooser_bar.layout().addWidget(self.save_button)
    chooser_bar.layout().addWidget(self.domain_box)
    self.layout().addWidget(chooser_bar)
    
    # create error dialog
    self.error_dialog = qt.QMessageBox()
  
  def check_entry_format(self):
    self.new_button.setEnabled(
      all([field.hasAcceptableInput() for field in self.pmt_fields])
      and self.orbit_field.text() != ''
    )
  
  def new_domain(self):
    try:
      cycle_strs = [field.text() for field in self.pmt_fields]
      orbit = self.orbit_field.text()
      tag = self.tag_field.text()
      domain = DessinDomain(cycle_strs, orbit, tag if tag else None)
    except Exception as ex:
      self.error_dialog.setText("Error computing dessin metadata.")
      self.error_dialog.setDetailedText(str(ex))
      self.error_dialog.exec()
    else:
      p, q, r = domain.orders
      if p*q*r - q*r - r*p - p*q > 0:
        # add new domain
        self.domains.append(domain)
        box = self.domain_box
        box.addItem(domain.name())
        
        # choose new domain
        box.setCurrentIndex(box.count()-1)
        self.change_controls()
      else:
        self.error_dialog.setText('Order triple {}, {}, {} not of hyperbolic type.'.format(p, q, r))
        self.error_dialog.setDetailedText(None)
        self.error_dialog.exec()
  
  def change_controls(self):
    super().change_controls()
    self.save_button.setEnabled(self.domain_box.currentText() != '')
  
  def take_the_canvas(self):
    index = self.domain_box.currentIndex()
    if index < 0:
      self.canvas.set_domain(None, working=False)
    else:
      self.canvas.set_domain(self.domains[index], working=True)
  
  def save_domain(self):
    index = self.domain_box.currentIndex()
    domain = self.domains[index]
    try:
      with open('domains/' + domain.name() + '.json', 'w') as file:
        domain.dump(file)
    except (TypeError, ValueError, OSError) as ex:
      self.error_dialog.setText('Error saving file.')
      self.error_dialog.setDetailedText(str(PicklingError))
      self.error_dialog.exec()

class SavedPanel(DessinControlPanel):
  def __init__(self, canvas, *args, **kwargs):
    super().__init__(canvas, *args, **kwargs)
    self.setLayout(qt.QHBoxLayout())
    
    # add domain chooser bar
    self.passport_box = qt.QComboBox()
    self.orbit_box = qt.QComboBox()
    self.domain_box = qt.QComboBox()
    self.passport_box.currentTextChanged.connect(self.list_orbits)
    self.orbit_box.currentTextChanged.connect(self.list_domains)
    self.domain_box.currentTextChanged.connect(self.change_controls)
    self.passport_box.setMinimumContentsLength(18)
    self.passport_box.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLength)
    self.orbit_box.setMinimumContentsLength(1)
    self.orbit_box.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLength)
    self.domain_box.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)
    for box in [self.passport_box, self.orbit_box, self.domain_box]:
      self.layout().addWidget(box)
    
    # open saved domains
    self.domains = {}
    for filename in os.listdir('domains'):
      if re.match(r'.*\.json$', filename):
        try:
          with open('domains/' + filename, 'r') as file:
            dom = DessinDomain.load(file)
        except (json.JSONDecodeError, OSError) as ex:
          print(ex)
        else:
          if not dom.passport in self.domains:
            self.domains[dom.passport] = {}
          if not dom.orbit in self.domains[dom.passport]:
            self.domains[dom.passport][dom.orbit] = []
          self.domains[dom.passport][dom.orbit].append(dom)
    
    # list passports. when we add the first one, the resulting
    # `currentTextChanged` signal will call `list_orbits`
    for passport in self.domains:
      self.passport_box.addItem(passport)
  
  def list_orbits(self, passport):
    self.orbit_box.blockSignals(True)
    self.orbit_box.clear()
    self.orbit_box.blockSignals(False)
    
    # when we add the first orbit to `orbit_box`, the resulting
    # `currentTextChanged` signal will call `list_domains`
    if passport:
      for orbit in self.domains[passport]:
        self.orbit_box.addItem(orbit)
  
  def list_domains(self, orbit):
    self.domain_box.blockSignals(True)
    self.domain_box.clear()
    self.domain_box.blockSignals(False)
    
    if orbit:
      passport = self.passport_box.currentText()
      for domain in self.domains[passport][orbit]:
        permutation_str = ','.join([s.cycle_string() for s in domain.group.gens()])
        if domain.tag == None:
          self.domain_box.addItem(permutation_str)
        else:
          self.domain_box.addItem('-'.join([permutation_str, domain.tag]))
  
  def take_the_canvas(self):
    passport = self.passport_box.currentText()
    if passport:
      orbit = self.orbit_box.currentText()
      index = self.domain_box.currentIndex()
      self.canvas.set_domain(self.domains[passport][orbit][index], working=False)
    else:
      self.canvas.set_domain(None)

class TilingWindow(qt.QMainWindow):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.setWindowTitle('Chorno-Belyi')
    self.resize(700, 900)
    
    # set up central panel
    central = qt.QWidget()
    central.setLayout(qt.QVBoxLayout())
    self.setCentralWidget(central)
    
    # add tiling canvas
    self.canvas = TilingCanvas(4, 4, 3, size=(1200, 1200))
    self.canvas.native.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
    central.layout().addWidget(self.canvas.native)
    
    # add work info bar
    work_info_bar = qt.QWidget()
    work_info_bar.setLayout(qt.QHBoxLayout())
    self.canvas.selection_display = qt.QLabel()
    self.canvas.paint_display = qt.QLabel()
    self.canvas.paint_display.setMaximumWidth(40)
    self.canvas.paint_display.setAlignment(Qt.AlignCenter)
    work_info_bar.layout().addWidget(self.canvas.selection_display)
    work_info_bar.layout().addWidget(self.canvas.paint_display)
    central.layout().addWidget(work_info_bar)
    
    # set up control panels for tilings, working domains, and saved domains
    tiling_panel = TilingPanel(self.canvas)
    working_panel = WorkingPanel(self.canvas)
    saved_panel = SavedPanel(self.canvas)
    
    # add mode tabs
    self.control_panels = qt.QTabWidget()
    self.control_panels.addTab(tiling_panel, "Tiling")
    self.control_panels.addTab(working_panel, "Working domain")
    self.control_panels.addTab(saved_panel, "Saved domain")
    self.control_panels.currentChanged.connect(self.change_mode)
    central.layout().addWidget(self.control_panels)
  
  def change_mode(self, index):
    self.control_panels.currentWidget().take_the_canvas()

if __name__ == '__main__' and sys.flags.interactive == 0:
  main_app = qt.QApplication(sys.argv)
  window = TilingWindow()
  window.show()
  main_app.exec_()
