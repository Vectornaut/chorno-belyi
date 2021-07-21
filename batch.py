import sys, json
from PyQt5.QtGui import QSurfaceFormat
from itertools import chain
from vispy import app
import vispy.io as io

from dessin import Dessin
from canvas import DomainCanvas

app.use_app(backend_name='PyQt5', call_reuse=True)

def parse_orbit_spec(orbit_spec):
  name, triple_str = orbit_spec.split('|')
  orbit = name[name.rindex('-') + 1:]
  triples = json.loads(triple_str)
  return [Dessin(triple, orbit, 20) for triple in triples]

if __name__ == '__main__' and sys.flags.interactive == 0:
  # read dessins
  try:
    with open('LMFDB_triples.txt', 'r') as file:
      dessin_orbits = map(parse_orbit_spec, file.readlines()[1:51])
      dessins = chain.from_iterable(dessin_orbits)
      hyp_dessins = filter(lambda dessin : dessin.geometry < 0, dessins)
  except (json.JSONDecodeError, OSError) as ex:
    print(ex)
    sys.exit(1)
  
  # set OpenGL version and profile
  format = QSurfaceFormat()
  format.setVersion(4, 1)
  format.setProfile(QSurfaceFormat.CoreProfile)
  QSurfaceFormat.setDefaultFormat(format)
  
  # set up canvas
  canvas = DomainCanvas(4, 4, 3, size=(400, 400))
  
  # render dessins
  for dessin in hyp_dessins:
    canvas.set_domain(dessin.domain)
    image = canvas.render()
    name = dessin.domain.name()
    io.write_png('batch-export/' + name + '.png', image)
