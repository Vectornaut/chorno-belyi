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
    list += [self]
    
    for k in range(3):
      if self.children[k] != None:
        self.children[k].flatten(offset, list)

if __name__ == '__main__':
  tree = TriangleTree()
  tree.store([0, 0, 0], highlight=9000)
  tree.store([0, 1], highlight=901)
  tree.store([0, 1, 1], highlight=9011)
  tree.store([2, 0, 1], highlight=9201)
  tree.store([2], color=92)
  
  tree.flatten()
  print(tree)
  print('')
  for node in tree.list:
    print('({}, {})'.format(node.highlight, node.color))
