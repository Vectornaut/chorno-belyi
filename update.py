import sys, os, re, json, pickle

import triangle_tree
from triangle_tree import DessinDomain

# switch to the new `lit`, `trim` format that allows half-triangles to be
# colored independently
def go_halfsies():
  in_dir = 'domains/onesies/'
  for filename in os.listdir(in_dir):
    if re.match(r'.*\.json$', filename):
      try:
        with open(in_dir + filename, 'r') as in_file:
          domain = DessinDomain.load(in_file, legacy=True)
        with open('domains/' + filename, 'w') as out_file:
          domain.dump(out_file)
      except (json.JSONDecodeError, TypeError, ValueError, OSError) as ex:
        print(ex)

def update_pickled_domains():
  for filename in os.listdir('domains/old'):
    if re.match(r'.*\.pickle$', filename):
      try:
        # open pickled domain
        with open('domains/old/' + filename, 'rb') as in_file:
          domain = pickle.load(in_file)
        
        # the tree's root node may include an element list. this list can't be
        # serialized to JSON, because it contains a circular reference
        if hasattr(domain.tree, 'list'): del domain.tree.list
        
        # serialize to JSON
        with open('domains/' + domain.name() + '.json', 'w') as out_file:
          domain.dump(out_file)
      except (pickle.UnpicklingError, AttributeError,  EOFError, ImportError, IndexError, OSError) as ex:
        print(ex)

if __name__ == '__main__' and sys.flags.interactive == 0:
  go_halfsies()
