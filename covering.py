# coding=utf-8

from sage.all import QQ, CDF, polygen, PowerSeriesRing, hypergeometric
import numpy as np
from numpy import pi, matmul, dot
from scipy.special import gamma

# the minkowski bilinear form
def mprod(v, w):
  return dot(v[:-1], w[:-1]) - v[-1]*w[-1]

def apply_series(series, w, order):
  # write cover(w) as w * deformation(w)
  deformation = 0
  w_order = w**order
  w_power = 1
  for coeff in series:
    deformation += coeff * w_power
    w_power *= w_order
  return w * deformation

# a covering map from the Poincaré disk to CP^1 which maps the vertices of the
# p, q, r triangulation to 0, 1, infinity. the order-p vertex `a` is at 0, and
# the order-q vertex `b` is on the positive real axis. we encode the covering
# map `phi` in a pair of Taylor series, `cover_a` and `cover_b`, with
#
#   phi(w) = cover_a(w / K_a)^p
#          = 1 - cover_b(shift(w) / K_b)^q,
#
# where `shift` is the Möbius transformation that translates `b` to `a` along
# the real axis
#
#   [KMSV] Michael Klug, Michael Musty, Sam Schiavone, and John Voight.
#          "Numerical calculation of three-point branched covers of the
#          projective line" <arXiv:1311.2081>
#
class Covering():
  def __init__(self, p, q, r, prec):
    self.p = p
    self.q = q
    self.r = r
    
    # --- find mirrors
    
    # get vertex cosines
    sp = sin(pi/p)
    cp = cos(pi/p)
    cq = cos(pi/q)
    sq = sin(pi/q)
    cr = cos(pi/r)
    
    # find the side normals of the fundamental triangle, scaled to unit norm
    self.mirrors = [
      np.array([0, 1, 0]),
      np.array([-sp, -cp, 0]),
      np.array([
        (cp*cq + cr) / sp,
        -cq,
        np.sqrt(-1 + (cp*cp + cq*cq + cr*cr + 2*cp*cq*cr)) / sp
      ])
    ]
    
    # --- find symmetries
    
    # `shift_ba` is the Möbius transformation that translates `b` to `a` along
    # the real axis. `cosh_dist` is cosh(d(a, b)). in [KMSV], `cosh_dist` is
    # called `lambda` (and the formula for it is off by a factor of two)
    cosh_dist = (cp*cq + cr) / (sp*sq)
    sinh_dist = np.sqrt(cosh_dist**2 - 1)
    ##self.shift_ba = np.array([
    self.shift = np.array([
      [ cosh_dist, 0, -sinh_dist],
      [         0, 1,          0],
      [-sinh_dist, 0,  cosh_dist]
    ])
    self.shift_ab = np.array([
      [cosh_dist, 0, sinh_dist],
      [        0, 1,         0],
      [sinh_dist, 0, cosh_dist]
    ])
    
    # `midpoint` is the midpoint of the edge in the fundamental domain.
    # `half_shift_ba` translates `b` to `midpoint` and `midpoint` to `a` along
    # the real axis. doing it twice gives `shift_ba`
    cosh_half_dist = np.sqrt((cosh_dist + 1) / 2)
    sinh_half_dist = np.sqrt((cosh_dist - 1) / 2)
    self.half_shift_ba = np.array([
      [ cosh_half_dist, 0, -sinh_half_dist],
      [              0, 1,               0],
      [-sinh_half_dist, 0,  cosh_half_dist]
    ])
    self.half_shift_ab = np.array([
      [cosh_half_dist, 0, sinh_half_dist],
      [             0, 1,              0],
      [sinh_half_dist, 0, cosh_half_dist]
    ])
    self.midpoint = self.half_shift_ab[:, 2]
    
    # `flip` does a half-turn rotation around `midpoint`
    bare_flip = np.array([
      [0, -1, 0],
      [1,  0, 0],
      [0,  0, 1]
    ])
    self.flip = matmul(matmul(self.half_shift_ab, bare_flip), self.half_shift_ba)
    
    # in the notation of [KMSV],
    #   delta_a = rot_ccw[0]
    #   delta_b = flip * rot_ccw[1] * flip
    self.rot_ccw = [
      np.array([
        [cos(2*pi/p), -sin(2*pi/p), 0],
        [sin(2*pi/p),  cos(2*pi/p), 0],
        [          0,            0, 1]
      ]),
      np.array([
        [cos(2*pi/q), -sin(2*pi/q), 0],
        [sin(2*pi/q),  cos(2*pi/q), 0],
        [          0,            0, 1]
      ])
    ]
    self.rot_cw = [
      np.array([
        [ cos(2*pi/p), sin(2*pi/p), 0],
        [-sin(2*pi/p), cos(2*pi/p), 0],
        [           0,           0, 1]
      ]),
      np.array([
        [ cos(2*pi/q), sin(2*pi/q), 0],
        [-sin(2*pi/q), cos(2*pi/q), 0],
        [           0,           0, 1]
      ])
    ]
    
    # in the notation of [KMSV],
    #   delta_a = rot_a_ccw
    #   delta_b = shift_ab * rot_b_ccw_shift * shift_ba
    self.rot_a_ccw = np.array([
      [cos(pi/p), -sin(pi/p), 0],
      [sin(pi/p),  cos(pi/p), 0],
      [        0,          0, 1]
    ])
    self.rot_b_ccw_shift = np.array([
      [cos(pi/q), -sin(pi/q), 0],
      [sin(pi/q),  cos(pi/q), 0],
      [        0,          0, 1]
    ])
    self.rot_a_cw = np.array([
      [ cos(pi/p), sin(pi/p), 0],
      [-sin(pi/p), cos(pi/p), 0],
      [         0,         0, 1]
    ])
    self.rot_b_cw_shift = np.array([
      [ cos(pi/q), sin(pi/q), 0],
      [-sin(pi/q), cos(pi/q), 0],
      [        0,          0, 1]
    ])
    
    # --- find scale factors
    
    # get hypergeometric parameters
    A = QQ(1)/2 * (QQ(1)/p - QQ(1)/q - QQ(1)/r + 1)
    B = QQ(1)/2 * (QQ(1)/p - QQ(1)/q + QQ(1)/r + 1)
    C = 1 + QQ(1)/p
    
    w_b = sinh_dist / (1 + cosh_dist)
    self.K_a = (
      w_b * gamma(2-C)*gamma(C-A)*gamma(C-B)
      / (gamma(1-A)*gamma(1-B)*gamma(C))
    )
    self.K_b = (
      -w_b * gamma(1+A+B-C)*gamma(C-A)*gamma(C-B)
      / (gamma(A)*gamma(B)*gamma(1-A-B+C))
    )
    
    # --- find covering series
    
    # define coordinates
    # z is the standard coordinate on CP^1
    # w is the standard coordinate on the Poincaré disk (only used in comments)
    # around 0,        t =   z,  s^p = t
    # around 1,        t = 1-z,  s^q = t
    # around infinity, t = 1/z,  s^r = t
    t = polygen(QQ, 't')
    s = polygen(QQ, 's')
    P = PowerSeriesRing(QQ, 's')
    
    # use ratios of hypergeometric functions to describe the function that lifts
    # CP^1, slit along the arc from -1 to 1 through infinity, to the fundamental
    # pair of triangles `a`, `b`, `c` and `a`, `b`, `conj(c)` in the Poincaré
    # disk. we encode this lift `psi` as a pair of Taylor series, `lift_0` and
    # `lift_1`, with
    #
    #   psi(z) = K_a * lift_0(z^(1/p))
    #          = K_b * lift_1((1-z)^(1/q))
    # find the hypergeometric series which lift neighborhoods of 0 and 1 to
    # neighborhoods of `a` and `b`. the `y` labels for the hypergeometric series
    # are from
    #
    #   Lucy Joan Slater, _Generalized Hypergeometric Functions_,
    #   Sections 1.3.1--1.3.2
    
    # find the lifting series around 0
    g_0 = hypergeometric([A, B], [C], t).series(t, prec) # y_1
    h_0 = hypergeometric([A-C+1, B-C+1], [2-C], t).series(t, prec) # y_2
    lift_0 = s * P(g_0(t = s**p), p*prec) / P(h_0(t = s**p), p*prec)
    
    # find the lifting series around 1
    h1_1 = hypergeometric([C-A, C-B], [1-A-B+C], t).series(t, prec) # y_6
    h2_1 = hypergeometric([  A,   B], [1+A+B-C], t).series(t, prec) # y_5
    lift_1 = s * P(h1_1(t = s**q), q*prec) / P(h2_1(t = s**q), q*prec)
    
    # invert lifting series to get covering series, and extract non-zero
    # coefficients. each series is of the form
    #
    #   w*(cover[0] + cover[1]*w^p + cover[2]*w^(2p) + cover[3]*w^(3p) + ...)
    #
    self.cover_a, self.cover_b = [
      np.array(lift.reverse().coefficients())
      for lift in [lift_0, lift_1]
    ]
  
  def apply(self, v):
    v_shift = np.matmul(self.shift, v)
    if v[2] < v_shift[2]:
      # v is closer to the time axis (this comparison works because v and
      # v_shift are on the forward -1 hyperboloid)
      w = (v[0] + 1j*v[1]) / (1 + v[2])
      s = apply_series(self.cover_a, w / self.K_a, self.p)
      return s**self.p
    else:
      w_shift = (v_shift[0] + 1j*v_shift[1]) / (1 + v_shift[2]);
      s = apply_series(self.cover_b, w_shift / self.K_b, self.q);
      return 1 - s**self.q
  
  # find the flip address of a point in the Poincaré disk
  def address(self, u):
    EPS = 1e-6
    TWIN_EPS = 1e-5
    
    r_sq = dot(u, u)
    if r_sq <= 1:
      v = np.array([2*u[0], -2*u[1], 1+r_sq]) / (1-r_sq)
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
              z = self.apply(v)
              return (address, 0 if z.real < 0.5 else 1)
    return (None, None)

##[TEST]
def test_covering(u):
  r_sq = dot(u, u)
  v = np.array([2*u[0], -2*u[1], 1+r_sq]) / (1-r_sq)
  print(str(u) + ' -> ' + str(bel.apply(v)))

##[TEST]
from math import cos, sin

if __name__ == '__main__':
  bel = Covering(5, 4, 3, 4)
  print(bel.shift)
  print()
  print(bel.cover_a)
  print()
  print(bel.cover_b)
  print()
  test_covering(0.1*np.array([1, 0]))
  test_covering(0.4*np.array([1, 0]))
  test_covering(0.6*np.array([1, 0]))
  test_covering(0.4*np.array([cos(pi/5), sin(pi/5)]))
