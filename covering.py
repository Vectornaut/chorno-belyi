# coding=utf-8

from sage.all import QQ, CDF, polygen, PowerSeriesRing, hypergeometric
import numpy as np
from numpy import pi

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
    
    # --- find shift
    
    # `shift` is the Möbius transformation that translates `b` to `a` along the
    # real axis. `cosh_dist` is cosh(d(a, b)). in [KMSV], `cosh_dist` is called
    # `lambda` (and the formula for it is off by a factor of two)
    cosh_dist = (np.cos(pi/p)*np.cos(pi/q) + np.cos(pi/r)) / (np.sin(pi/p)*np.sin(pi/q))
    sinh_dist = np.sqrt(cosh_dist**2 - 1)
    self.shift = np.array([
      [cosh_dist, 0, -sinh_dist],
      [        0, 1,          0],
      [-sinh_dist, 0, cosh_dist]
    ])
    
    # --- find covering series
    
    # get hypergeometric parameters
    A = QQ(1)/2 * (QQ(1)/p - QQ(1)/q - QQ(1)/r + 1)
    B = QQ(1)/2 * (QQ(1)/p - QQ(1)/q + QQ(1)/r + 1)
    C = 1 + QQ(1)/p
    
    # define coordinates
    # z is the standard coordinate on CP^1
    # w is the standard coordinate on the Poincaré disk
    # around 0,        t =   z,  s^p = t
    # around 1,        t = 1-z,  s^q = t
    # around infinity, t = 1/z,  s^r = t
    t = polygen(QQ, 't')
    s = polygen(QQ, 's')
    w = polygen(QQ, 'w')
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
    
    # invert lifting series to get covering series
    self.cover_a, self.cover_b = [
      lift.reverse()(s = w).change_ring(CDF)
      for lift in [lift_0, lift_1]
    ]

if __name__ == '__main__':
  bel = Covering(5, 4, 3, 3)
  print(bel.shift)
  print()
  print(bel.cover_a)
  print()
  print(bel.cover_b)
