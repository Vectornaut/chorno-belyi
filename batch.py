import sys, os, json
from argparse import ArgumentParser
from itertools import chain
from vispy import app
import vispy.io as io

from domain import Domain
from dessin import Dessin
from canvas import DomainCanvas

app.use_app(backend_name='PyQt5', call_reuse=True)

def is_nonempty_list(a):
  return a and isinstance(a, list)

class Orbit:
  def __init__(self, orbit_spec):
    # parse specification
    name, lone_triple_str, paired_triple_str = orbit_spec.split('|')
    label_sep = name.rindex('-')
    self.passport = name[:label_sep]
    self.passport_path = self.passport.replace('-', '/').replace('_', '/')
    self.label = name[label_sep + 1:]
    self.index = None
    lone_triples = json.loads(lone_triple_str)
    paired_triples = json.loads(paired_triple_str)
    
    # compute common data by creating the first domain
    lone_done = 0
    paired_done = 0
    if is_nonempty_list(lone_triples):
      first_domain = Domain(lone_triples[0], self.label)
      self.domains = [first_domain]
      lone_done = 1
    elif is_nonempty_list(paired_triples):
      first_domain = Domain(paired_triples[0], self.label)
      self.domains = [first_domain, first_domain.conj()]
      paired_done = 1
    else:
      raise Exception('Orbit ' + name + ' must come with a non-empty list of permutation triples')
    self.geometry = first_domain.geometry
    self.common_data = {
      'degree': first_domain.degree,
      't_number': first_domain.t_number,
      'orders': first_domain.orders,
      'geometry': self.geometry,
      'passport': self.passport
    }
    
    # create the rest of the domains
    self.domains.extend(Domain(triple, self.label, data=self.common_data) for triple in lone_triples[lone_done:])
    for triple in paired_triples[paired_done:]:
      domain = Domain(triple, self.label)
      self.domains.extend([domain, domain.conj()])
    
    # initialize lazy attributes
    self._dessins = None
  
  def dessins(self):
    if not self._dessins:
      self._dessins = [Dessin(domain, 20) for domain in self.domains]
    return self._dessins
  
  def to_dict(self):
    return {
      'passport_path': self.passport_path,
      'label': self.label,
      'index': self.index,
      'dessins': self.domains
    }

if __name__ == '__main__' and sys.flags.interactive == 0:
  # read dessins
  try:
    with open('LMFDB_triples_conjugates.txt', 'r') as file:
      orbits = map(Orbit, file.readlines()[1:])
      hyp_orbits = filter(lambda orbit : orbit.geometry < 0, orbits)
  except (json.JSONDecodeError, OSError) as ex:
    print(ex)
    sys.exit(1)
  
  # set up canvas
  canvas = DomainCanvas(4, 4, 3, size=(400, 400))
  
  # handle command line options
  parser = ArgumentParser()
  parser.add_argument('-n', dest='n_max', type=int, action='store', default=30)
  parser.add_argument('--all', dest='partial', action='store_false')
  parser.add_argument('--dry-run', dest='dry_run', action='store_true')
  args = parser.parse_args()
  
  # render dessins
  n = 0
  n_colors = 0
  for orbit in hyp_orbits:
    for dessin in orbit.dessins():
      canvas.set_domain(dessin.domain)
      image = canvas.render()
      name = dessin.domain.name()
      if args.dry_run:
        print(name + '.png')
      else:
        io.write_png(os.path.join('batch-export', name + '.png'), image)
      n_colors = max(n_colors, dessin.n_colors)
      
      n += 1
      if args.partial and n >= args.n_max:
        print('{} dessins, {} edge colors'.format(n, n_colors))
        sys.exit(0)
  print('{} dessins, {} edge colors'.format(n, n_colors))
