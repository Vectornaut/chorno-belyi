import sys, os, json
from vispy import app
import vispy.io as io

from dessin import Dessin
from canvas import DomainCanvas

app.use_app(backend_name='PyQt5', call_reuse=True)

class Orbit:
  def __init__(self, orbit_spec):
    name, triple_str = orbit_spec.split('|')
    self.given_name = name[name.rindex('-') + 1:]
    triples = json.loads(triple_str)
    self.dessins = [Dessin(triple, self.given_name, 20) for triple in triples]
    if self.dessins:
      self.passport = self.dessins[0].domain.passport
      self.geometry = self.dessins[0].geometry
    else:
      throw('Orbit ' + name + ' is empty')

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
  hyp_orbits_by_passport = {}
  for orbit in hyp_orbits:
    if orbit.passport in hyp_orbits_by_passport:
      hyp_orbits_by_passport[orbit.passport].append(orbit)
    else:
      hyp_orbits_by_passport[orbit.passport] = [orbit]
  
  # set up canvas
  canvas = DomainCanvas(4, 4, 3, size=(400, 400))
  
  # render dessins
  dry_run = '--dry-run' in sys.argv[1:]
  for passport in hyp_orbits_by_passport:
    orbits = hyp_orbits_by_passport[passport]
    if len(orbits) > 1 or len(orbits[0].dessins) > 1:
      passport_dir = os.path.join('batch-export', passport)
      if dry_run:
        print(os.path.split(passport_dir)[1])
      elif not os.path.isdir(passport_dir):
        os.mkdir(passport_dir, mode=0o755)
      for orbit in orbits:
        orbit_dir = os.path.join(passport_dir, orbit.given_name)
        if dry_run:
          print(2*' ' + os.path.split(orbit_dir)[1])
        elif not os.path.isdir(orbit_dir):
          os.mkdir(orbit_dir, mode=0o755)
        for dessin in orbit.dessins:
          canvas.set_domain(dessin.domain)
          image = canvas.render()
          name = dessin.domain.permutation_str()
          if dry_run:
            print(4*' ' + name + '.png')
          else:
            io.write_png(os.path.join(orbit_dir, name + '.png'), image)
