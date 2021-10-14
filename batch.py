import sys, os, json
from argparse import ArgumentParser
from itertools import chain
from vispy import app
import vispy.io as io

from domain import Domain
from dessin import Dessin
from canvas import DomainCanvas

app.use_app(backend_name='PyQt5', call_reuse=True)

class Orbit:
  def __init__(self, orbit_spec):
    name, triple_str = orbit_spec.split('|')
    label_sep = name.rindex('-')
    self.passport = name[:label_sep]
    self.passport_path = self.passport.replace('-', '/').replace('_', '/')
    self.label = name[label_sep + 1:]
    self.index = None
    self.triples = json.loads(triple_str)
    assert self.triples and isinstance(self.triples, list), 'Orbit ' + name + ' must come with a non-empty list of permutation triples'
    
    # initialize lazy attributes
    self._geometry = None
    self._domains = None
    self._dessins = None
  
  def geometry(self):
    if not self._domains:
      # just compute the first domain
      self._domains = [Domain(self.triples[0], self.label)]
    return self._domains[0].geometry
  
  def domains(self):
    if not self._domains:
      self._domains = [Domain(triple, self.label) for triple in self.triples]
    elif len(self._domains) < len(self.triples):
      # the `geometry` method has computed the first domain already
      self._domains.extend(Domain(triple, self.label) for triple in self.triples[1:])
    return self._domains
  
  def dessins(self):
    if not self._dessins:
      self._dessins = [Dessin(domain, 20) for domain in self.domains()]
    return self._dessins
  
  def to_dict(self):
    return {
      'passport_path': self.passport_path,
      'label': self.label,
      'index': self.index,
      'dessins': self.domains()
    }

if __name__ == '__main__' and sys.flags.interactive == 0:
  # read dessins
  try:
    with open('LMFDB_triples.txt', 'r') as file:
      orbits = map(Orbit, file.readlines()[1:])
      hyp_orbits = filter(lambda orbit : orbit.geometry() < 0, orbits)
  except (json.JSONDecodeError, OSError) as ex:
    print(ex)
    sys.exit(1)
  
  # set up canvas
  canvas = DomainCanvas(4, 4, 3, size=(400, 400))
  
  # handle command line options
  parser = ArgumentParser()
  parser.add_argument('-n', dest='n_max', type=int, action='store', default=20)
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
