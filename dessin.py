# coding=utf-8

import numpy as np
from numpy import identity, matmul, pi, cos, sin

from covering import Covering
from domain import Domain

class Dessin():
  # for deserialization, you can pass precomputed group details and a serialized
  # tree in the `data` dictionary [IMPLEMENT]
  def __init__(self, domain, prec):
    self.domain = domain
    
    # if the dessin is hyperbolic, find its covering map and build a fundamental
    # domain
    if domain.geometry < 0:
      self.covering = Covering(*domain.orders, prec)
      self.build_tree()
  
  def build_tree(self):
    # extract tiling data, for convenience
    degree = self.domain.degree
    orders = self.domain.orders
    permutations = self.domain.group.gens()
    
    # extract covering data and address map, for convenience
    flip = self.covering.flip
    rot_ccw = (self.covering.rot(0, 1), self.covering.rot(1, 1))
    midpoint = self.covering.midpoint
    address = self.covering.address
    
    # find a fundamental domain
    self.route = [degree*[None], degree*[None]]
    self.rep = [degree*[None], degree*[None]]
    self.edge_gluings = []
    self.vertex_gluings = []
    frontier = [[1], [1]]
    side = 1
    color = 1
    while True:
      if not frontier[side]:
        side = 1-side
      if not frontier[side]:
        break
      edge = frontier[side].pop(0)
      ##[DOM ALG] print('side ' + str(side) + ', edge ' + str(edge))
      ##[DOM ALG] print('|' + str(self.route[0]))
      ##[DOM ALG] print('|' + str(self.route[1]))
      # explore the opposite half-edge
      if self.route[1-side][edge-1] == None:
        # route to the opposite half-edge from this one
        if self.route[side][edge-1] == None:
          self.route[1-side][edge-1] = []
          self.rep[1-side][edge-1] = identity(3)
        else:
          self.route[1-side][edge-1] = self.route[side][edge-1]
          self.rep[1-side][edge-1] = matmul(self.rep[side][edge-1], flip)
        ##[DOM ALG] print('flip')
        ##[DOM ALG] print('|' + str(self.route[0]))
        ##[DOM ALG] print('|' + str(self.route[1]))
        
        # petal the opposite half-edge
        for cnt in range(orders[1-side]-1):
          next_edge = permutations[1-side](edge)
          ##[DOM ALG] print('  petaling; next edge: ' + str(next_edge))
          if self.route[1-side][next_edge-1] == None:
            self.route[1-side][next_edge-1] = self.route[1-side][edge-1] + [1-side]
            self.rep[1-side][next_edge-1] = matmul(self.rep[1-side][edge-1], rot_ccw[1-side])
            frontier[1-side].append(next_edge)
          else:
            self.vertex_gluings.append((1-side, edge, next_edge, color))
            ##[DOM ALG] print('  vertex gluing ' + str(self.vertex_gluings[-1]))
            color += 1
            break
          ##[DOM ALG] print('  |' + str(self.route[0]))
          ##[DOM ALG] print('  |' + str(self.route[1]))
          edge = next_edge
      elif side == 0:
        # glue to the opposite half-edge
        self.edge_gluings.append((edge, color))
        ##[DOM ALG] print('  edge gluing ' + str(self.edge_gluings[-1]))
        color += 1
      side = 1-side
    
    # find the addresses of the edge representatives in our fundamental domain
    c_nudge = cos(0.1*pi/orders[0])
    s_nudge = sin(0.1*pi/orders[0])
    nudge_ccw = np.array([
      [c_nudge, -s_nudge, 0],
      [s_nudge,  c_nudge, 0],
      [      0,        0, 1]
    ])
    nudge_cw = np.array([
      [ c_nudge, s_nudge, 0],
      [-s_nudge, c_nudge, 0],
      [       0,       0, 1]
    ])
    midpoint_upper = matmul(nudge_ccw, midpoint)
    midpoint_lower = matmul(nudge_cw, midpoint)
    addresses_upper = [
      [address(matmul(g, midpoint_upper))[0] for g in self.rep[0]],
      [address(matmul(matmul(g, flip), midpoint_upper))[0] for g in self.rep[1]]
    ]
    addresses_lower = [
      [address(matmul(g, midpoint_lower))[0] for g in self.rep[0]],
      [address(matmul(matmul(g, flip), midpoint_lower))[0] for g in self.rep[1]]
    ]
    
    for side in range(2):
      ##[DOM ALG] print('side ' + str(side) + ' representatives:')
      for g in self.rep[side]:
        mid = matmul(g, midpoint)
        ##[DOM ALG] print(mid[0:2] / (1 + mid[2]))
    
    # build the triangle tree for our fundamental domain
    tree = self.domain.tree
    for edge in range(1, degree+1):
      for side in range(2):
        tree.store(addresses_upper[side][edge-1], side, True, None, None)
        tree.store(addresses_lower[side][edge-1], side, True, None, None)
    for (edge, color) in self.edge_gluings:
      for side in range(2):
        for addresses in [addresses_upper, addresses_lower]:
          tree.store(addresses[side][edge-1], side, True, None, color)
          tree.store(addresses[side][edge-1], side, True, None, color)
    for (side, edge, next_edge, color) in self.vertex_gluings:
      if side == 0:
        address_ccw = addresses_upper[side][edge-1]
        address_cw = addresses_lower[side][next_edge-1]
      else:
        address_ccw = addresses_lower[side][edge-1]
        address_cw = addresses_upper[side][next_edge-1]
      tree.store(address_ccw, side, True, color, None)
      tree.store(address_cw, side, True, color, None)

##[TEST]
if __name__ == '__main__':
  ##dessin = Dessin(Domain([(1,2,3,4),(1,3,4,2),(1,3,4)], 'a'), 20) # 4T5-4_4_3.1
  dessin = Dessin(Domain([(1,2,3,4,5),(1,2,5,4),(1,2,4,3)], 'a'), 20) # 5T3-5_4.1_4.1
  print('final routes')
  print(dessin.route[0])
  print(dessin.route[1])
  for side in range(2):
    print('side ' + str(side) + ' representatives:')
    for g in dessin.rep[side]:
      mid = matmul(g, dessin.covering.midpoint)
      print(mid[0:2] / (1 + mid[2]))
  print('final edge gluings')
  print(dessin.edge_gluings)
  print('final vertex gluings')
  print(dessin.vertex_gluings)
