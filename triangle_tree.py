from sage.all import PermutationGroup
from sage.groups.perm_gps.permgroup import PermutationGroup_generic
import os, re, pickle ## for adding metadata to existing domains

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
      self.list = list
    
    # list this node
    self.index = len(list) + offset
    list.append(self)
    
    for k in range(3):
      if self.children[k] != None:
        self.children[k].flatten(offset, list)

class DessinDomain:
  # if `group` isn't a PermutationGroup, it'll be passed to the PermutationGroup
  # constructor
  def __init__(self, group, orbit, tag = None):
    # store metadata
    if isinstance(group, PermutationGroup_generic):
      self.group = group
    else:
      self.group = PermutationGroup(group, canonicalize = False)
    self.degree = self.group.degree()
    self.t_number = self.group.gap().TransitiveIdentification()
    self.orders = tuple(s.order() for s in self.group.gens())
    self.orbit = orbit
    self.tag = tag
    
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

def add_metadata(dry_run = True):
    old_name_format = 'domain\\-.*-(.)-(\\(.*\\)),(\\(.*\\)),(\\(.*\\))\\.pickle'
    print('')
    for filename in os.listdir('domains/old'):
      if info := re.search(old_name_format, filename):
        # get metadata
        domain = DessinDomain(info.groups()[1:], info.groups()[0])
        new_name = domain.name() + '.pickle'
        print(filename + '\n-> ' + new_name + '\n')
        
        # add fundamental domain tree
        try:
          with open('domains/old/' + filename, 'rb') as file:
            domain.tree = pickle.load(file)
        except (pickle.UnpicklingError, AttributeError,  EOFError, ImportError, IndexError) as ex:
          print(ex)
        
        if not dry_run:
          try:
            with open('domains/' + new_name, 'wb') as file:
              pickle.dump(domain, file)
          except pickle.PicklingError as ex:
            print(ex)

def metadata_test():
  old_name_format = 'domain\\-.*-(.)-(\\(.*\\)),(\\(.*\\)),(\\(.*\\))\\.pickle'
  for filename in [
    'domain-5_4_6-5.1.1_4.2.1_3.2.2-a-(1,5,7,6,3),(1,2,3,4)(5,6),(1,2)(3,5,4)(6,7).pickle',
    'domain-6.1_3.3.1_4.1.1.1-a-(1,4,5,2,6,7),(2,7,6)(3,5,4),(1,2,3,4).pickle'
  ]:
    print(re.search(old_name_format, filename).groups())
  

def tree_test():
  tree = TriangleTree()
  tree.store([0, 0, 0], highlight=9000)
  tree.store([0, 1], highlight=901)
  tree.store([0, 1, 1], highlight=9011)
  ##tree.store([0, 1, 2], highlight=9012)
  tree.store([2, 0, 1], highlight=9201)
  tree.store([2], color=92)
  
  tree.flatten()
  print(tree)
  print('')
  for node in tree.list:
    print('({}, {})'.format(node.highlight, node.color))
  print('')
  
  for address in [[0, 1], [0, 0, 0], [0, 1, 1], [2, 0], [2, 0, 1]]:
    tree.drop(address)
    print('drop ' + ''.join(map(str, address)))
    print(tree)
    print('')
  
  print('note that node 2 is severable because its highlight mode is 0, even though its color is nonzero')
  print('')

def domain_test():
  permutations = [
    [(1, 4, 5, 2, 6, 7)],
    [(2, 7, 6), (3, 5, 4)],
    [(1, 2, 3, 4)]
  ]
  dom = DessinDomain(permutations, 'a')
  return dom.name() == '7T7-6.1_3.3.1_4.1.1.1-a-(1,4,5,2,6,7),(2,7,6)(3,5,4),(1,2,3,4)'
