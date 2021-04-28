# coding=utf-8

from sage.all import PermutationGroup
from sage.categories.permutation_groups import PermutationGroups
from numpy import identity, matmul

from covering import Covering

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
      ## read fundamental domain
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
      if not (g is None):
        print(matmul(g, dessin.midpoint))
      else:
        print(g)
  print('final edge gluings')
  print(dessin.edge_gluings)
  print('final vertex gluings')
  print(dessin.vertex_gluings)
