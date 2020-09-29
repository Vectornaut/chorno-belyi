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

// --- display ---

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

const float VIEW = 2.;
const float PI = 3.141592653589793;

void main() {
  // find screen coordinate
  vec2 u = VIEW*(2.*gl_FragCoord.xy - resolution) / shortdim;
  
  // find pixel radius, for antialiasing
  float r_px_screen = VIEW / shortdim; // the inner radius of a pixel in the Euclidean metric of the screen
  
  vec3 color = vec3(1.);
  vec2 z = casin(u);
  if (mod(8./PI * z.x, 1.) < 0.5) {
    color.r *= 0.8;
    color.g *= 0.7;
  }
  if (mod(8./PI * z.y, 1.) < 0.5) {
    color.g *= 0.8;
    color.b *= 0.7;
  }
  gl_FragColor = vec4(color, 1.);
}
''')

class StripCanvas(app.Canvas):
  def __init__(self, *args, **kwargs):
    app.Canvas.__init__(self, *args, **kwargs)
    self.program = gloo.Program(vertex, fragment, count = 6) # we'll always send 6 vertices
    
    # draw a rectangle that covers the canvas
    self.program['position'] = [
      (-1, -1), (-1, 1), (1, 1), # northwest triangle
      (-1, -1), (1, 1), (1, -1)  # southeast triangle
    ]
    
    # initialize settings
    self.update_resolution()
  
  def update_resolution(self):
    width, height = self.physical_size
    gloo.set_viewport(0, 0, width, height)
    self.program['resolution'] = [width, height]
    self.program['shortdim'] = min(width, height)
  
  def on_draw(self, event):
    self.program.draw()
  
  def on_resize(self, event):
    self.update_resolution()

if __name__ == '__main__' and sys.flags.interactive == 0:
  StripCanvas(title = "Vertical strip", size = (500, 500)).show()
  app.run()
