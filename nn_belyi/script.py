# Script for running from shell.

from importlib import reload
import dessin_data as dd
import AI_functions as AI

data = list(dd.load_json_data(dd.DATA_FILE).values())[10:30]
