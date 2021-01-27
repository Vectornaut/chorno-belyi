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

if __name__ == '__main__':
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
