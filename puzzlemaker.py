import sys, os, json
from vispy import app
import vispy.io as io

from dessin import Dessin
from canvas import DomainCanvas

app.use_app(backend_name='PyQt5', call_reuse=True)

class Passport:
  def __init__(self, first_orbit):
    self.name = first_orbit.passport
    self.orbits = [first_orbit]
  
  # if `orbit` belongs to this passport, add it and return True. otherwise,
  # return False
  def append(self, orbit):
    if orbit.passport == self.name:
      self.orbits.append(orbit)
      return True
    else:
      return False

class Orbit:
  def __init__(self, orbit_spec):
    name, triple_str = orbit_spec.split('|')
    self.label = name[name.rindex('-') + 1:]
    triples = json.loads(triple_str)
    self.dessins = [Dessin(triple, self.label, 20) for triple in triples]
    assert self.dessins, 'Orbit ' + name + ' must include at least one dessin'
    self.passport = self.dessins[0].domain.passport
    self.geometry = self.dessins[0].geometry

if __name__ == '__main__' and sys.flags.interactive == 0:
  # read dessins
  try:
    with open('LMFDB_triples.txt', 'r') as file:
      orbits = map(Orbit, file.readlines()[1:51])
      hyp_orbits = filter(lambda orbit : orbit.geometry < 0, orbits)
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
  for passport in hyp_passports:
    orbits = passport.orbits
    if len(orbits) > 1 or len(orbits[0].dessins) > 1:
      passport_dir = os.path.join('batch-export', passport.name)
      if dry_run:
        print(os.path.split(passport_dir)[1])
      elif not os.path.isdir(passport_dir):
        os.mkdir(passport_dir, mode=0o755)
      for orbit in orbits:
        for dessin in orbit.dessins:
          canvas.set_domain(dessin.domain)
          image = canvas.render()
          name = dessin.domain.name()
          if dry_run:
            print(2*' ' + name + '.png')
          else:
            io.write_png(os.path.join(passport_dir, name + '.png'), image)
