from sage.all import PermutationGroup
from sage.categories.permutation_groups import PermutationGroups
import json
from sage.all import ZZ ##[TEMP] needed for serialization of Sage integers

# highlight styles
NONE = 0
L_HALF = 1
R_HALF = 2
L_WHOLE = 3
R_WHOLE = 4
WHOLE = 5

class TriangleTree:
  def __init__(self):
    self.children = [None, None, None]
    self.highlight = NONE
    self.color = 0
  
  def __str__(self):
    return self.show('*', 0)
  
  def show(self, k, level):
    text = level*' ' + '{} ({}, {})'.format(k, self.highlight, self.color)
    if hasattr(self, 'index') and self.index != None:
      text += ' #' + str(self.index)
    for k in range(3):
      if self.children[k] != None:
        text += '\n' + self.children[k].show(k, level+1)
    return text
  
  # store the given highlight and color data at the given address, creating any
  # nodes on the way to the address that don't exist already
  def store(self, address, highlight=None, color=None):
    if address == []:
      if highlight != None: self.highlight = highlight
      if color != None: self.color = color
    else:
      k = address[0]
      if self.children[k] == None: self.children[k] = TriangleTree()
      self.children[k].store(address[1:], highlight, color)
  
  def drop(self, address):
    if address == []:
      if all(child == None for child in self.children):
        # this node can be severed from the tree
        return True
      else:
        # this node can't be severed, so just clear its highlighting and tell
        # the nodes above to do nothing
        self.highlight = NONE
        self.color = 0
        return False
    else:
      k = address[0]
      if self.children[k] == None:
        # there's nothing stored at the address provided, so tell the nodes
        # above to do nothing
        return False
      elif self.children[k].drop(address[1:]):
        # child `k` can be severed from the tree
        self.children[k] = None
        if (
          self.children[(k+1)%3] == None
          and self.children[(k+2)%3] == None
          and self.highlight == NONE
        ):
          # this node can be severed too
          return True
  
  # list the tree's nodes, depth first, and give each node an `index` attribute
  # that hold its list index plus the given offset. if `list` is provided,
  # append the list of nodes to it. otherwise, store the list of nodes in the
  # `list` attribute of the root node
  def flatten(self, offset=0, list=None):
    # start a node list, if none is provided
    if list == None:
      list = []
      self.list = list ## i don't know how much of a speed boost this actually gives, and it creates a circular reference that makes serialization a pain
    
    # list this node
    self.index = len(list) + offset
    list.append(self)
    
    for k in range(3):
      if self.children[k] != None:
        self.children[k].flatten(offset, list)
  
  def dump(self, fp, **kwargs):
    json.dump(self, fp, default=lambda obj: obj.__dict__, **kwargs)
  
  def dumps(self, **kwargs):
    return json.dumps(self, default=lambda obj: obj.__dict__, **kwargs)
  
  # for JSON deserialization, build a TriangleTree recursively from nested dicts
  #
  #   StackOverflow user martineau
  #   https://stackoverflow.com/a/23597335/1644283
  #
  @staticmethod
  def from_dict(data):
    tree = TriangleTree()
    for k in range(3):
      if child := data['children'][k]:
        tree.children[k] = TriangleTree.from_dict(child)
    tree.highlight = data['highlight']
    tree.color = data['color']
    return tree
  
  @staticmethod
  def load(fp, **kwargs):
    return TriangleTree.from_dict(json.load(fp, **kwargs))
  
  @staticmethod
  def loads(s, **kwargs):
    return TriangleTree.from_dict(json.loads(s, **kwargs))

class DessinDomain:
  # `group` will be passed to PermutationGroup, unless it's already in the
  # PermutationGroups category. for deserialization, you can pass precomputed
  # group details and a serialized tree in the `data` dictionary
  def __init__(self, group, orbit, tag = None, data = None):
    # store independent metadata
    if group in PermutationGroups:
      self.group = group
    else:
      self.group = PermutationGroup(group, canonicalize = False)
    self.orbit = orbit
    self.tag = tag
    
    # store dependent metadata and tree
    if data:
      self.degree = data['degree']
      self.t_number = data['t_number']
      self.orders = data['orders']
      self.passport = data['passport']
      self.tree = TriangleTree.from_dict(data['tree'])
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
      
      # start a triangle tree
      self.tree = TriangleTree()
  
  def name(self):
    permutation_str = ','.join([s.cycle_string() for s in self.group.gens()])
    all_but_tag = '-'.join([self.passport, self.orbit, permutation_str])
    if self.tag == None:
      return all_but_tag
    else:
      return '-'.join([all_but_tag, tag])
  
  class Encoder(json.JSONEncoder):
    def default(self, obj):
      if obj in PermutationGroups:
        return [s.cycle_string() for s in obj.gens()]
      elif obj in ZZ: ##[TEMP] for handling old instances
        return int(obj)
      else:
        return obj.__dict__
  
  def dump(self, fp, **kwargs):
    json.dump(self, fp, cls=DessinDomain.Encoder, **kwargs)
  
  def dumps(self, **kwargs):
    return json.dumps(self, cls=DessinDomain.Encoder, **kwargs)
  
  @staticmethod
  def from_dict(data):
    return DessinDomain(data['group'], data['orbit'], data['tag'], data)
  
  @staticmethod
  def load(fp, **kwargs):
    return DessinDomain.from_dict(json.load(fp, **kwargs))
  
  @staticmethod
  def loads(s, **kwargs):
    return DessinDomain.from_dict(json.loads(s, **kwargs))

def tree_test():
  tree = TriangleTree()
  tree.store([0, 0, 0], highlight=9000)
  tree.store([0, 1], highlight=901)
  tree.store([0, 1, 1], highlight=9011)
  tree.store([2, 0, 1], highlight=9201)
  tree.store([2], color=92)
  
  tree.flatten()
  print(tree)
  print()
  for node in tree.list:
    print('({}, {})'.format(node.highlight, node.color))
  print()
  
  for address in [[0, 1], [0, 0, 0], [0, 1, 1], [2, 0], [2, 0, 1]]:
    tree.drop(address)
    print('drop ' + ''.join(map(str, address)))
    print(tree)
    print()
  
  print('note that node 2 is severable because its highlight mode is 0, even though its color is nonzero')
  print()

def json_test():
  dom = DessinDomain([[(1,2)], [(2,3)], [(3,1)]], 'a')
  dom.tree.store([0, 0, 0], highlight=9000)
  dom.tree.store([0, 1], highlight=901)
  dom.tree.store([0, 1, 1], highlight=9011)
  dom.tree.store([2, 0, 1], highlight=9201)
  dom.tree.store([2], color=92)
  print(dom.tree)
  print()
  
  serialized = dom.dumps()
  reconstituted = DessinDomain.loads(serialized)
  print(reconstituted.tree)

def domain_test():
  permutations = [
    [(1, 4, 5, 2, 6, 7)],
    [(2, 7, 6), (3, 5, 4)],
    [(1, 2, 3, 4)]
  ]
  dom = DessinDomain(permutations, 'a')
  return dom.name() == '7T7-6.1_3.3.1_4.1.1.1-a-(1,4,5,2,6,7),(2,7,6)(3,5,4),(1,2,3,4)'
