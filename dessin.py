# coding=utf-8

from sage.all import PermutationGroup
from sage.categories.permutation_groups import PermutationGroups

from covering import Covering

class Dessin(Covering):
  # `group` will be passed to PermutationGroup, unless it's already in the
  # PermutationGroups category. for deserialization, you can pass precomputed
  # group details and a serialized tree in the `data` dictionary
  def __init__(self, group, orbit, tag=None, data=None):
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
      
      # store passport
      label = 'T'.join(map(str, [self.degree, self.t_number]))
      partition_str = '_'.join([
        '.'.join(map(str, s.cycle_type()))
        for s in self.group.gens()
      ])
      self.passport = '-'.join([label, partition_str])
      
      # find a fundamental domain
      s = self.group.gens()
      self.route = [[[]] + (self.degree-1)*[None], self.degree*[None]]
      self.edge_gluings = []
      self.vertex_gluings = []
      frontier = [[1], []]
      side = 0
      color = 1
      skips = 0
      while skips < 2:
        print('frontier ' + str(side) + ': ' + str(frontier[side]))
        if frontier[side]:
          skips = 0
          print('side: ' + str(side))
          edge = frontier[side].pop(0)
          for cnt in reversed(range(self.orders[side])):
            print('edge: ' + str(edge))
            print(self.route[0])
            print(self.route[1])
            # explore the half-edge on the other side
            if self.route[1-side][edge-1] == None:
              self.route[1-side][edge-1] = self.route[side][edge-1]
              frontier[1-side].append(edge)
            else:
              self.edge_gluings.append((edge, color))
              color += 1
            print('explore opposite half-edge')
            print(self.route[0])
            print(self.route[1])
            
            # rotate to the next half-edge
            if cnt > 0:
              next_edge = s[side](edge)
              print('next edge: ' + str(next_edge))
              if self.route[side][next_edge-1] == None:
                self.route[side][next_edge-1] = self.route[side][edge-1] + [side]
              else:
                self.vertex_gluings.append((side, edge, next_edge, color))
                print('vertex gluing ' + str((side, edge, next_edge, color)))
                color += 1
                break
              edge = next_edge
        else:
          skips += 1
        side = 1-side

##[TEST]
if __name__ == '__main__':
  ##dessin = Dessin([(1,2,3,4),(1,3,4,2),(1,3,4)], 'a')
  dessin = Dessin([(1,2,3,4,5),(1,2,5,4),(1,2,4,3)], 'a')
  print(dessin.route[0])
  print(dessin.route[1])
  print(dessin.vertex_gluings)
