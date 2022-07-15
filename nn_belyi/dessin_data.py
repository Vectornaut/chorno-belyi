import sys, os, json
import numpy as np
from sage.all import PermutationGroup
import torch, torch_geometric

DATA_FILE = "../data-nn/dessin_training.json"

################################################################################
#
# Classes
#
################################################################################

##############################
# Passport class

class Passport:
  """
  A collection of Dessins with a Galois action. The Passport stores
  the orbits of the Dessins as TrainingOrbit objects
  """

  def __init__(self, label, orbits):
    self._label = label
    self._orbits = orbits
    
  def orbits(self):
    return self._orbits

  def label(self):
    return self._label

  def dessins(self):
    return sum([x.dessins for x in self.orbits()], [])
  
  def labelled_training_set(self):

    # Extract the dessins in the training orbits
    
    n = len(self.orbits())
    #orbit_label = {self.orbits().dessins[i] : i for i in range(n)}

    dessins = self.dessins()

    output = []
    
    # Don't vectorize. Just label.
    for x in dessins:
      for y in dessins:
        if x.label == y.label:
          output.append([(x, y), 1])
        else:
          output.append([(x, y), 0])

    a, b = zip(*output)
    return list(a), list(b)
  
##############################
# DessinMedialGraph class

class DessinMedialGraph:
  """
  The medial graph of a dessin d'enfant. a dessin is a ribbon graph with black
  and white vertices. the vertices of the medial graph are the edges of the
  dessin, represented by integers 1, ..., n. the arrows of the medial graph
  are the counterclockwise edge adjacencies of the dessin; the ordered pair
  (a, b) represents the arrow a --> b, which says that going counterclockwise
  from edge `a` brings you to edge `b`. each arrow of the medial graph inherits
  the color of the associated dessin vertex. the black and white arrows are
  stored in the lists `black_arrows` and `white_arrows`, respectively

  within each passport, the orbits are arbitrarily labeled 'a', 'b', 'c', ....
  the dessin's label is stored in the string `label`. we want a classifier that
  takes two medial graphs, which are guaranteed to come from the same passport,
  and decides whether they come from the same orbit

  within each passport, all the dessins can be embedded in the same topological
  surface. the integer `geometry` tells you whether that surface is:

    1   spherical
    0   euclidean
   -1   hyperbolic

  """
  
  class DessinMedialGraph:
    def __init__(self, black_arrows, white_arrows, label, orders, geometry):
      self.black_arrows = black_arrows
      self.white_arrows = white_arrows
      self.label = label
      self.orders = orders
      self.geometry = geometry
    
    @staticmethod
    def from_triple(triple, label, orders, geometry):
      return DessinMedialGraph(
        list(enumerate(triple[0], 1)),
        list(enumerate(triple[1], 1)),
        label,
        orders,
        geometry
      )
    
    def to_dict(self):
      return {
        'black_arrows': self.black_arrows,
        'white_arrows': self.white_arrows
      }
    
    @staticmethod
    def from_dict(data, label, orders, geometry):
      return DessinMedialGraph(
        data['black_arrows'],
        data['white_arrows'],
        label,
        orders,
        geometry
      )
  
  def vectorize(self):
    """
    Test function to see if we can feed tensor-flow reasonably well
    """
    
    edges = sum(self.white_arrows) + sum(self.black_arrows)

    # Normalize so that the length of this vector is 40.
    if len(edges) < 30:
      edges = edges + [0 for i in range(30-len(edges))]
    else:
      edges = edges[0:30]

    return edges
  
  def to_geom_data(self):

    # Set the label to be one of three values.
    if self.geometry > 0:
      y = 2
    elif self.geometry == 0:
      y = 1
    else:
      y = 0
    
    # build edge tensor
    # Pytorch *really* assumes everything in sight is 0-indexed.
    edges = [[a-1, b-1] for [a,b] in self.black_arrows + self.white_arrows]
    edge_index = torch.tensor(edges)
    
    # buld vertex tensor
    vertsb = {x for x in sum(self.black_arrows, [])}
    vertsw = {x for x in sum(self.white_arrows, [])}
    #verts = [[len([arrow for arrow in self.black_arrows if x in arrow])]
    #         for x in vertsb.union(vertsw)]
    
    verts = [[1] for x in vertsb.union(vertsw)]
    
    return torch_geometric.data.Data(
      x=torch.tensor(verts, dtype=torch.float),
      edge_index = edge_index.t().contiguous(),
      y=torch.tensor([y], dtype=torch.long))


  def to_genus_data(self):
    pass


  def to_generator_data(self):
    # Compute the degree [In the dessin] of the vertices of the appropriate colour
    # 
    # Find the LCM of the cycles 
    pass

  
##############################
# Training Orbit class

class TrainingOrbit:
  def __init__(self, passport, label, orders, geometry, dessins):
    self.passport = passport
    self.label = label
    self.orders = orders
    self.geometry = geometry
    self.dessins = dessins
  
  @staticmethod
  def from_spec(orbit_spec):
    # extract orbit data
    name, triple_str = orbit_spec.split('|')
    label_sep = name.rindex('-')
    passport = name[:label_sep]
    label = name[label_sep + 1:]
    triples = json.loads(triple_str)
    assert triples and isinstance(triples, list), 'Orbit ' + name + ' must come with a non-empty list of permutation triples'
    
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
    dessins = [DessinMedialGraph.from_triple(triple, label, orders, geometry) for triple in triples]
    
    # build orbit
    return TrainingOrbit(passport, label, orders, geometry, dessins)
  
  def to_dict(self):
    return {
      'passport': self.passport,
      'label': self.label,
      'orders': self.orders,
      'geometry': self.geometry,
      'dessins': [dessin.to_dict() for dessin in self.dessins]
    }
  
  @staticmethod
  def from_dict(data):
    passport = data['passport']
    label = data['label']
    orders = data['orders']
    geometry = data['geometry']
    dessins = [DessinMedialGraph.from_dict(dessin, label, orders, geometry) for dessin in data['dessins']]
    return TrainingOrbit(passport, label, orders, geometry, dessins)


################################################################################
#
# Read, write, and create raw training data.
#
################################################################################


# read a training data file into a dictionary that maps each passport string
# to the list of orbits in that passport, encoded as TrainingOrbit objects
def load_json_data(path):
  with open(path, 'r') as file:
    passports = json.load(file)

  return {
    label: Passport(label, [TrainingOrbit.from_dict(data) for data in orbits])
    for label, orbits in passports.items()
  }


def read_lmfdb_data():
  with open('../LMFDB_triples.txt', 'r') as in_file:
    return map(TrainingOrbit.from_spec, in_file.readlines()[1:])


def sort_orbits_into_passports(orbits):
  # sort orbits into passports
  passports = {}
  for orbit in orbits:
    if orbit.passport in passports:
      passports[orbit.passport].append(orbit.to_dict())
    else:
      passports[orbit.passport] = [orbit.to_dict()]
      
  return passports


def write_json_data(passports):
  # write orbits with dessin medial graphs
  with open('../data-nn/dessin_training.json', 'w') as out_file:
    json.dump(passports, out_file)


def make_data_from_lmfdb():
  orbits = read_lmfdb_data()
  passports = sort_orbits_into_passports(orbits)
  write_training_data(passports)

  
################################################################################
#
# Create a pytorch style Dataset
#
################################################################################

import shutil, os
import torch
from torch_geometric.data import InMemoryDataset

class DessinGeometryDataset(InMemoryDataset):
  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
    super().__init__(root, transform, pre_transform, pre_filter)
    self.data, self.slices = torch.load(self.processed_paths[0])
  
  @property
  def raw_file_names(self):
    return ['dessin_training.json']
  
  @property
  def processed_file_names(self):
    return ['I_hope_this_works.data']
  
  def download(self):
    shutil.copyfile("../data-nn/dessin_training.json", self.raw_paths[0])
  
  def process(self):
    ## whatever your script does to read dessin_training.json and spit out a
    ## list of x.to_geom_data() outputs

    raw_file = self.raw_paths[0]
    data = list(load_json_data(raw_file).values())

    
    data_list = geometry_training_set(data) ## the list of x.to_geom_data() outputs
    
    data, slices = self.collate(data_list)
    torch.save((data, slices), self.processed_paths[0])


################################################################################
#
# Create training data
#
################################################################################

def labelled_training_data(data):
  xvalues = []
  yvalues = []

  for datum in data:
    x, y = datum.labelled_training_set()
    xvalues += x
    yvalues += y

  return xvalues, yvalues

def geometry_training_set(data):
  xvalues = []
  yvalues = []

  dessins = sum(list(sum([x.dessins for x in datum.orbits()], []) for datum in data), [])

  return [x.to_geom_data() for x in dessins]


    


################################################################################
#
# Turn dessin into torch_geometric.data.Data
#
################################################################################
