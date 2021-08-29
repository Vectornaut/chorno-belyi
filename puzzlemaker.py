import sys, os, json
from django.conf import settings
from django.template import Engine, Context
from vispy import app
import vispy.io as io

from domain import Domain
from dessin import Dessin
from canvas import DomainCanvas

app.use_app(backend_name='PyQt5', call_reuse=True)

class Passport:
  def __init__(self, first_orbit):
    self.name = first_orbit.passport
    self.orbits = [first_orbit]
    first_orbit.index = 0
  
  # if `orbit` belongs to this passport, add it and return True. otherwise,
  # return False
  def append(self, orbit):
    if orbit.passport == self.name:
      orbit.index = self.orbits[-1].index + 1
      self.orbits.append(orbit)
      return True
    else:
      return False
  
  def to_dict(self):
    return {
      'passport': self.name,
      'orbits': [orbit.to_dict() for orbit in self.orbits]
    }

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
      'dessin_names': [dessin.domain.name() for dessin in self.dessins()]
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
  
  # consolidate passports
  hyp_passports = []
  for orbit in hyp_orbits:
    # try to append `orbit` to the current passport. if it doesn't belong to
    # that passport, start a new one
    if (not hyp_passports) or (not hyp_passports[-1].append(orbit)):
      hyp_passports.append(Passport(orbit))
  
  # set up canvas
  canvas = DomainCanvas(4, 4, 3, size=(400, 400))
  
  # render dessins
  dry_run = '--dry-run' in sys.argv[1:]
  n_max = 20
  settings.configure() # use Django's default settings
  puzzle_template = Engine(dirs='.').get_template('puzzle.html')
  n_done = 0
  for passport in hyp_passports:
    orbits = passport.orbits
    if len(orbits) > 1 and not all(len(orbit.dessins()) == 1 for orbit in orbits):
      passport_dir = os.path.join('docs', passport.name)
      if dry_run:
        print(os.path.split(passport_dir)[1])
      else:
        if not os.path.isdir(passport_dir):
          os.mkdir(passport_dir, mode=0o755)
        passport_context = Context(passport.to_dict())
        puzzle_page = puzzle_template.render(passport_context)
        with open(os.path.join(passport_dir, 'index.html'), 'w') as file:
          file.write(puzzle_page)
      for orbit in orbits:
        for dessin in orbit.dessins():
          canvas.set_domain(dessin.domain)
          image = canvas.render()
          name = dessin.domain.name()
          if dry_run:
            print(2*' ' + name + '.png')
          else:
            io.write_png(os.path.join(passport_dir, name + '.png'), image)
      n_done += 1
      if n_done >= n_max:
        break
