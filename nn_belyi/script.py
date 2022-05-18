# Script for running from shell.

from importlib import reload
import dessin_data as dd

data = list(dd.load_json_data(dd.DATA_FILE).values())[10:30]
