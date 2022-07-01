# Script for running from shell.

from importlib import reload
import dessin_data as dd
from dessin_data import DessinGeometryDataset

if True:
  passports = list(dd.load_json_data("../data-nn/dessin_training.json").values())
  dessins = sum([pp.dessins() for pp in passports], [])

dataset = DessinGeometryDataset("Geometry")

