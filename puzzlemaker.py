import sys, os, json
from argparse import ArgumentParser
from django.conf import settings
from django.template import Engine, Context
from vispy import app
import vispy.io as io

from canvas import DomainCanvas
from batch import Orbit

app.use_app(backend_name='PyQt5', call_reuse=True)

class Passport:
  def __init__(self, first_orbit):
    # basic info
    self.name = first_orbit.passport
    self.orbits = [first_orbit]
    first_orbit.index = 0
    
    # puzzle text
    self.features = ''
    self.flavor = ''
    self.trivia = ''
    self.mathjax = False
  
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
      'orbits': [orbit.to_dict() for orbit in self.orbits],
      'features': self.features,
      'flavor': self.flavor,
      'trivia': self.trivia,
      'mathjax': self.mathjax
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
  
  # set up canvas and Django engine
  canvas = DomainCanvas(4, 4, 3, size=(400, 400))
  settings.configure() # use Django's default settings
  engine = Engine(dirs='.')
  
  # handle command line options
  parser = ArgumentParser()
  parser.add_argument('-n', dest='n_max', type=int, action='store', default=30)
  parser.add_argument('--dry-run', dest='dry_run', action='store_true')
  parser.add_argument('--no-puzzles', dest='puzzles', action='store_false')
  parser.add_argument('--no-pics', dest='pics', action='store_false')
  ##[SELECT PAGES] parser.add_argument('pages', metavar='pages', nargs='*', help='pages to render')
  args = parser.parse_args()
  
  # read puzzle text
  try:
    with open('puzzle_text.json', 'r') as file:
      puzzle_text = json.loads(file.read())
  except (json.JSONDecodeError, OSError) as ex:
    print(ex)
    sys.exit(1)
  
  # render dessins and puzzle pages
  puzzle_template = engine.get_template('puzzle.html')
  puzzles = []
  n_colors = 0
  for passport in hyp_passports:
    orbits = passport.orbits
    if len(orbits) > 1 and not all(len(orbit.triples) == 1 for orbit in orbits):
      if passport.name in puzzle_text:
        text = puzzle_text[passport.name]
        if 'features' in text: passport.features = text['features']
        if 'flavor' in text: passport.flavor = text['flavor']
        if 'trivia' in text: passport.trivia = text['trivia']
      passport_dir = os.path.join('docs', passport.name)
      if args.dry_run:
        print(os.path.split(passport_dir)[1])
      elif (args.puzzles or args.pics) and not os.path.isdir(passport_dir):
        os.mkdir(passport_dir, mode=0o755)
      if args.puzzles:
        passport_context = Context(passport.to_dict())
        puzzle_page = puzzle_template.render(passport_context)
        if args.dry_run:
          print('  index.html')
        else:
          with open(os.path.join(passport_dir, 'index.html'), 'w') as file:
            file.write(puzzle_page)
      if args.pics:
        for orbit in orbits:
          for dessin in orbit.dessins():
            canvas.set_domain(dessin.domain)
            image = canvas.render()
            name = dessin.domain.name()
            if args.dry_run:
              print(2*' ' + name + '.png')
            else:
              io.write_png(os.path.join(passport_dir, name + '.png'), image)
            n_colors = max(n_colors, dessin.n_colors)
      
      puzzles.append(passport)
      if len(puzzles) >= args.n_max:
        break
  print('{} puzzles, {} edge colors'.format(len(puzzles), n_colors))
  
  list_template = engine.get_template('puzzles.html')
  list_context = Context({'passports': [passport.to_dict() for passport in puzzles]})
  list_page = list_template.render(list_context)
  with open(os.path.join('docs', 'puzzles.html'), 'w') as file:
    file.write(list_page)
