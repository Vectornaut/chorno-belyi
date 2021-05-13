from sage.all import PermutationGroup
from sage.categories.permutation_groups import PermutationGroups
import json

from triangle_tree import TriangleTree

class Domain:
  # `group` will be passed to PermutationGroup, unless it's already in the
  # PermutationGroups category. for deserialization, you can pass precomputed
  # group details and a serialized tree in the `data` dictionary
  def __init__(self, group, orbit, tag=None, data=None, legacy=False):
    # store independent metadata
    if group in PermutationGroups:
      self.group = group
    else:
      self.group = PermutationGroup(group, canonicalize=False)
    self.orbit = orbit
    self.tag = tag
    
    # store dependent metadata and tree
    if data:
      self.degree = data['degree']
      self.t_number = data['t_number']
      self.orders = data['orders']
      self.passport = data['passport']
      self.tree = TriangleTree.from_dict(data['tree'], legacy)
    else:
      self.degree = int(self.group.degree())
      self.t_number = int(self.group.gap().TransitiveIdentification())
      self.orders = tuple(int(s.order()) for s in self.group.gens())
      
      # store passport
      label = 'T'.join(map(str, [self.degree, self.t_number]))
      partition_str = '_'.join([
        '.'.join(map(str, s.cycle_type()))
        for s in self.group.gens()
      ])
      self.passport = '-'.join([label, partition_str])
      
      # start a triangle tree
      self.tree = TriangleTree()
  
  def name(self):
    permutation_str = ','.join([s.cycle_string() for s in self.group.gens()])
    all_but_tag = '-'.join([self.passport, self.orbit, permutation_str])
    if self.tag == None:
      return all_but_tag
    else:
      return '-'.join([all_but_tag, self.tag])
  
  class Encoder(json.JSONEncoder):
    def default(self, obj):
      if obj in PermutationGroups:
        return [s.cycle_string() for s in obj.gens()]
      else:
        return obj.__dict__
  
  def dump(self, fp, **kwargs):
    json.dump(self, fp, cls=Domain.Encoder, **kwargs)
  
  def dumps(self, **kwargs):
    return json.dumps(self, cls=Domain.Encoder, **kwargs)
  
  @staticmethod
  def from_dict(data, legacy=False):
    return Domain(data['group'], data['orbit'], data['tag'], data, legacy)
  
  @staticmethod
  def load(fp, legacy=False, **kwargs):
    return Domain.from_dict(json.load(fp, **kwargs), legacy)
  
  @staticmethod
  def loads(s, legacy=False, **kwargs):
    return Domain.from_dict(json.loads(s, **kwargs), legacy)

def json_test():
  dom = Domain([[(1,2)], [(2,3)], [(3,1)]], 'a')
  dom.tree.store([0, 0, 0], 0, None, None, 9000)
  dom.tree.store([0, 1], 0, None, None, 901)
  dom.tree.store([0, 1, 1], 0, None, None, 9011)
  dom.tree.store([2, 0, 1], 0, None, None, 9201)
  dom.tree.store([2], 0, None, 82, None)
  print(dom.tree)
  print()
  
  serialized = dom.dumps()
  reconstituted = Domain.loads(serialized)
  print(reconstituted.tree)

def domain_test():
  permutations = [
    [(1, 4, 5, 2, 6, 7)],
    [(2, 7, 6), (3, 5, 4)],
    [(1, 2, 3, 4)]
  ]
  dom = Domain(permutations, 'a')
  return dom.name() == '7T7-6.1_3.3.1_4.1.1.1-a-(1,4,5,2,6,7),(2,7,6)(3,5,4),(1,2,3,4)'
