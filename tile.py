# based on Anton Sherwood's programs for painting hyperbolic tilings
# <https://commons.wikimedia.org/wiki/User:Tamfang/programs>

from __future__ import division

from PIL import Image
from numpy import linspace, array, diag, dot, matmul
from numpy.linalg import norm, solve, inv
from math import sqrt, cos, sin, pi

# the minkowski bilinear form
def mprod(v, w):
  return -v[0]*w[0] + dot(v[1:], w[1:])

def msq(v):
  return mprod(v, v)

def munit(v):
  return v / sqrt(abs(msq(v)))

# reflect across the plane orthogonal to the unit vector `mirror`
def reflect(mirror, v):
  return v - 2*mprod(v, mirror)*mirror

def sample(x, y, mirrors):
  r_sq = x*x + y*y
  if r_sq >= 1: return 0xC0;
  
  v = array([1 + r_sq, 2*x, 2*y]) / (1 - r_sq)
  flips = 0
  onsides = 0 # how many times in a row we've been in the desired half-plane
  while flips < 40:
    for m in mirrors:
      if mprod(v, m) > 0:
        v = reflect(m, v)
        flips += 1
        onsides = 0
      else:
        onsides += 1
        if onsides >= 3: return 0xFF*(flips & 1)
  return 0xA0

def paint(p, q, r, size = 500):
  # get angle cosines
  sp = sin(pi/p)
  cp = cos(pi/p)
  cq = cos(pi/q)
  cr = cos(pi/r)
  
  # find the side normals of the fundamental triangle, scaled to unit norm
  raw_mirrors = [
    array([0, 1, 0]),
    array([0, -cp, sp]),
    array([
      sqrt(-1 + (cp*cp + cq*cq + cr*cr + 2*cp*cq*cr)) / sp,
      -cq,
      -(cp*cq + cr) / sp
    ])
  ]
  ##DEBUG
  ##print([msq(u) for u in raw_mirrors])
  ##return
  
  # find the center of the fundamental triangle, which has the same minkowski
  # product with each side normal
  center = solve(array(raw_mirrors), array([-1, -1, -1]))
  center[0] *= -1
  if center[0] < 0: center *= -1
  center = munit(center)
  
  # move the center onto the time axis
  tilt = norm(center[1:])
  shift = center[0] + tilt # the doppler shift factor of the boost we need
  boostframe = array([
    [     tilt,       tilt,          0],
    [center[1], -center[1],  center[2]],
    [center[2], -center[2], -center[1]]
  ])
  boost = matmul(matmul(boostframe, diag([1/shift, shift, 1])), inv(boostframe))
  mirrors = [matmul(boost, m) for m in raw_mirrors]
  
  # paint the image
  img = Image.new("L", (size, size))
  mesh = linspace(-7/6, 7/6, size)
  img.putdata([sample(x, y, mirrors) for x in mesh for y in mesh])
  img.save("tiling.png")
