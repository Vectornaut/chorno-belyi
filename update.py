import sys, os, re, json

import triangle_tree
from triangle_tree import DessinDomain

# store outer and inner trim in separate fields, instead of using sign to
# specify whether each trim should be applied to the outer or inner edge
def separate_trim():
  in_dir = 'domains/signed/'
  for filename in os.listdir(in_dir):
    if re.match(r'.*\.json$', filename):
      try:
        with open(in_dir + filename, 'r') as in_file:
          domain = DessinDomain.load(in_file, legacy=True)
        with open('domains/' + filename, 'w') as out_file:
          domain.dump(out_file)
      except (json.JSONDecodeError, TypeError, ValueError, OSError) as ex:
        print(ex)

if __name__ == '__main__' and sys.flags.interactive == 0:
  separate_trim()
