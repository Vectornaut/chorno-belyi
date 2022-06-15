# Script for running from shell.

from importlib import reload
import dessin_data as dd
# import AI_functions as AI

data = list(dd.load_json_data(dd.DATA_FILE).values())[10:30]

class DessinGeometryDataset(InMemoryDataset):
  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
    super().__init__(root, transform, pre_transform, pre_filter)
    self.data, self.slices = torch.load(self.processed_paths[0])
  
  @property
  def raw_file_names(self):
    return ['dessin_training.json']
  
  @property
  def processed_file_names(self):
    return []
  
  def download(self):
    print('Nothing to download')
  
  def process(self):
    ## whatever your script does to read dessin_training.json and spit out a
    ## list of x.to_geom_data() outputs
    data_list = ## the list of x.to_geom_data() outputs
    
    data, slices = self.collate(data_list)
    torch.save((data, slices), self.processed_paths[0])

