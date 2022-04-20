import sys, os, json
from sage.all import PermutationGroup

# the medial graph of a dessin d'enfant. a dessin is a ribbon graph with black
# and white vertices. the vertices of the medial graph are the edges of the
# dessin, represented by integers 1, ..., n. the arrows of the medial graph
# are the counterclockwise edge adjacencies of the dessin; the ordered pair
# (a, b) represents the arrow a --> b, which says that going counterclockwise
# from edge `a` brings you to edge `b`. each arrow of the medial graph inherits
# the color of the associated dessin vertex. the black and white arrows are
# stored in the lists `black_arrows` and `white_arrows`, respectively
#
# within each passport, the orbits are arbitrarily labeled 'a', 'b', 'c', ....
# the dessin's label is stored in the string `label`. we want a classifier that
# takes two medial graphs, which are guaranteed to come from the same passport,
# and decides whether they come from the same orbit
#
# within each passport, all the dessins can be embedded in the same topological
# surface. the integer `geometry` tells you whether that surface is:
#
#    1   spherical
#    0   euclidean
#   -1   hyperbolic
#
class DessinMedialGraph:
  def __init__(self, black_arrows, white_arrows, label, geometry):
    self.black_arrows = black_arrows
    self.white_arrows = white_arrows
    self.label = label
    self.geometry = geometry
  
  @staticmethod
  def from_triple(triple, label, geometry):
    return DessinMedialGraph(
      list(enumerate(triples[0], 1))
      list(enumerate(triples[1], 1))
      label,
      geometry
    )
  
  def to_dict(self):
    return {
      'black_arrows': self.black_arrows
      'white_arrows': self.white_arrows
    }
  
  @staticmethod
  def from_dict(data, label, geometry):
    return DessinMedialGraph(
      data['black_arrows'],
      data['white_arrows'],
      label,
      geometry
    )

class TrainingOrbit:
  def __init__(self, passport, label, geometry, dessins):
    self.passport = passport
    self.label = label
    sefl.geometry = geometry
    self.dessins = dessins
  
  @staticmethod
  def from_spec(orbit_spec):
    # extract orbit data
    name, triple_str = orbit_spec.split('|')
    label_sep = name.rindex('-')
    passport = name[:label_sep]
    label = name[label_sep + 1:]
    triples = json.loads(triple_str)
    assert triples and isinstance(self.triples, list), 'Orbit ' + name + ' must come with a non-empty list of permutation triples'
    
    # find the geometry type:
    #    1   spherical
    #    0   euclidean
    #   -1   hyperbolic
    example_group = PermutationGroup(triples[0], canonicalize=False)
    orders = tuple(int(s.order()) for s in example_group.gens())
    assert len(orders) == 3
    p, q, r = orders
    geometry = int(np.sign(q*r + r*p + p*q - p*q*r))
    
    # build dessins
    self.dessins = [DessinMedialGraph.from_triple(triple, label, geometry) for triple in triples]
    
    # build orbit
    return TrainingOrbit(passport, label, geometry, dessins)
  
  def to_dict(self):
    return {
      'passport': self.passport,
      'label': self.label,
      'geometry': self.geometry,
      'dessins': [dessin.to_dict() for dessin in self.dessins]
    }
  
  @staticmethod
  def from_dict(data):
    passport = data['passport']
    label = data['label']
    geometry = data['geometry']
    dessins = [DessinMedialGraph.from_dict(dessin, label, geometry) for dessin in data['dessins']]
    return TrainingOrbit(passport, label, geometry, dessins)

if __name__ == '__main__' and sys.flags.interactive == 0:
  # read orbit specifications
  try:
    with open('LMFDB_triples.txt', 'r') as in_file:
      orbits = map(Orbit, file.readlines()[1:])
  except (json.JSONDecodeError, OSError) as ex:
    print(ex)
    sys.exit(1)
  
  # sort orbits into passports
  passports = {}
  for orbit in orbits:
    if passports.has_key(orbit.passport):
      passports[orbit.passport].push(orbit.to_dict)
    else:
      passports[orbit.passport] = [orbit.to_dict]
  
  # write orbits with dessin medial graphs
  try:
    with open('dessin_training.json', 'w') as out_file:
      json.dump([orbit.to_dict() for orbit in orbits])
  except (TypeError, ValueError, OSError) as ex:
    self.error_dialog.setText('Error saving file.')
    self.error_dialog.setDetailedText(str(PicklingError))
    self.error_dialog.exec()
