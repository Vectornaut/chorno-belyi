# coding=utf-8

from sage.all import PermutationGroup
from sage.categories.permutation_groups import PermutationGroups
import numpy as np
from numpy import identity, matmul, pi, cos, sin

from covering import Covering
from triangle_tree import TriangleTree

class Dessin(Covering):
  # `group` will be passed to PermutationGroup, unless it's already in the
  # PermutationGroups category. for deserialization, you can pass precomputed
  # group details and a serialized tree in the `data` dictionary
  def __init__(self, group, orbit, prec, tag=None, data=None):
    # store independent metadata
    if group in PermutationGroups:
      self.group = group
    else:
      self.group = PermutationGroup(group, canonicalize=False)
    self.orbit = orbit
    self.tag = tag
    
    # store dependent metadata and fundamental domain
    if data:
      self.degree = data['degree']
      self.t_number = data['t_number']
      self.orders = data['orders']
      self.passport = data['passport']
      self.tree = TriangleTree.from_dict(data['tree'], legacy)
    else:
      self.degree = int(self.group.degree())
      self.t_number = int(self.group.gap().TransitiveIdentification())
      self.orders = tuple(s.order() for s in self.group.gens())
      
      # initialize covering
      super().__init__(*self.orders, prec)
      
      # store passport
      label = 'T'.join(map(str, [self.degree, self.t_number]))
      partition_str = '_'.join([
        '.'.join(map(str, s.cycle_type()))
        for s in self.group.gens()
      ])
      self.passport = '-'.join([label, partition_str])
      
      # find a fundamental domain
      s = self.group.gens()
      self.route = [self.degree*[None], self.degree*[None]]
      self.rep = [self.degree*[None], self.degree*[None]]
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
        print('side ' + str(side) + ', edge ' + str(edge))
        print('|' + str(self.route[0]))
        print('|' + str(self.route[1]))
        # explore the opposite half-edge
        if self.route[1-side][edge-1] == None:
          # route to the opposite half-edge from this one
          if self.route[side][edge-1] == None:
            self.route[1-side][edge-1] = []
            self.rep[1-side][edge-1] = identity(3)
          else:
            self.route[1-side][edge-1] = self.route[side][edge-1]
            self.rep[1-side][edge-1] = matmul(self.rep[side][edge-1], self.flip)
          print('flip')
          print('|' + str(self.route[0]))
          print('|' + str(self.route[1]))
          
          # petal the opposite half-edge
          for cnt in range(self.orders[1-side]-1):
            next_edge = s[1-side](edge)
            print('  petaling; next edge: ' + str(next_edge))
            if self.route[1-side][next_edge-1] == None:
              self.route[1-side][next_edge-1] = self.route[1-side][edge-1] + [1-side]
              self.rep[1-side][next_edge-1] = matmul(self.rep[1-side][edge-1], self.rot_ccw[1-side])
              frontier[1-side].append(next_edge)
            else:
              self.vertex_gluings.append((1-side, edge, next_edge, color))
              print('  vertex gluing ' + str(self.vertex_gluings[-1]))
              color += 1
              break
            print('  |' + str(self.route[0]))
            print('  |' + str(self.route[1]))
            edge = next_edge
        elif side == 0:
          # glue to the opposite half-edge
          self.edge_gluings.append((edge, color))
          print('  edge gluing ' + str(self.edge_gluings[-1]))
          color += 1
        side = 1-side
      
      # find the addresses of the edge representatives in our fundamental domain
      c_nudge = cos(0.1*pi/self.orders[0])
      s_nudge = sin(0.1*pi/self.orders[0])
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
      midpoint_upper = matmul(nudge_ccw, self.midpoint)
      midpoint_lower = matmul(nudge_cw, self.midpoint)
      addresses_upper = [
        [self.address(matmul(g, midpoint_upper))[0] for g in self.rep[0]],
        [self.address(matmul(matmul(g, self.flip), midpoint_upper))[0] for g in self.rep[1]]
      ]
      addresses_lower = [
        [self.address(matmul(g, midpoint_lower))[0] for g in self.rep[0]],
        [self.address(matmul(matmul(g, self.flip), midpoint_lower))[0] for g in self.rep[1]]
      ]
      
      for side in range(2):
        print('side ' + str(side) + ' representatives:')
        for g in self.rep[side]:
          mid = matmul(g, self.midpoint)
          print(mid[0:2] / (1 + mid[2]))
      
      # build the triangle tree for our fundamental domain
      self.tree = TriangleTree()
      for edge in range(1, self.degree+1):
        for side in range(2):
          self.tree.store(addresses_upper[side][edge-1], side, True, None, None)
          self.tree.store(addresses_lower[side][edge-1], side, True, None, None)
      for (edge, color) in self.edge_gluings:
        for side in range(2):
          for addresses in [addresses_upper, addresses_lower]:
            self.tree.store(addresses[side][edge-1], side, True, None, color)
            self.tree.store(addresses[side][edge-1], side, True, None, color)
      for (side, edge, next_edge, color) in self.vertex_gluings:
        if side == 0:
          address_ccw = addresses_upper[side][edge-1]
          address_cw = addresses_lower[side][next_edge-1]
        else:
          address_ccw = addresses_lower[side][edge-1]
          address_cw = addresses_upper[side][next_edge-1]
        self.tree.store(address_ccw, side, True, color, None)
        self.tree.store(address_cw, side, True, color, None)
  
  def name(self):
    permutation_str = ','.join([s.cycle_string() for s in self.group.gens()])
    all_but_tag = '-'.join([self.passport, self.orbit, permutation_str])
    if self.tag == None:
      return all_but_tag
    else:
      return '-'.join([all_but_tag, self.tag])

##[TEST]
if __name__ == '__main__':
  ##dessin = Dessin([(1,2,3,4),(1,3,4,2),(1,3,4)], 'a', 20) # 4T5-4_4_3.1
  dessin = Dessin([(1,2,3,4,5),(1,2,5,4),(1,2,4,3)], 'a', 20) # 5T3-5_4.1_4.1
  print('final routes')
  print(dessin.route[0])
  print(dessin.route[1])
  for side in range(2):
    print('side ' + str(side) + ' representatives:')
    for g in dessin.rep[side]:
      mid = matmul(g, dessin.midpoint)
      print(mid[0:2] / (1 + mid[2]))
  print('final edge gluings')
  print(dessin.edge_gluings)
  print('final vertex gluings')
  print(dessin.vertex_gluings)
