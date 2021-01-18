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
    self.index = None
  
  def __str__(self):
    return self.show('*', 0)
  
  def show(self, k, level):
    text = level*' ' + '{} ({}, {})'.format(k, self.highlight, self.color)
    if self.index != None:
      text += ' #' + str(self.index)
    for k in range(3):
      if self.children[k] != None:
        text += '\n' + self.children[k].show(k, level+1)
    return text
  
  # store the given highlight and color data at the given address, creating any
  # nodes on the way to the address that don't exist already
  def store(self, address, highlight, color=0):
    if address == []:
      self.highlight = highlight
      self.color = color
    else:
      k = address[0]
      if self.children[k] == None: self.children[k] = TriangleTree()
      self.children[k].store(address[1:], highlight, color)
  
  # index each node with an integer in the order of a depth-first traversal,
  # starting at the given index. return the index after the last one used
  def index_down(self, index=0):
    self.index = index
    index += 1
    for k in range(3):
      if self.children[k] != None:
        index = self.children[k].index_down(index)
    return index

if __name__ == '__main__':
  tree = TriangleTree()
  tree.store([0, 0, 0], 9000)
  tree.store([0, 1], 901)
  tree.store([0, 1, 1], 9011)
  tree.store([2, 0, 1], 9201)
  tree.store([2], 92)
