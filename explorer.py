import sys, os, re, json
import PyQt5.QtWidgets as qt
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QSurfaceFormat, QValidator
from vispy import app
import vispy.io as io
from math import floor

from domain import Domain
from dessin import Dessin
from canvas import DomainCanvas

app.use_app(backend_name='PyQt5', call_reuse=True)

class DessinControlPanel(qt.QWidget):
  def __init__(self, canvas, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    # store a pointer to the DomainCanvas this panel controls
    self.canvas = canvas
  
  def showing(self):
    return (
      hasattr(self.parentWidget(), 'currentWidget')
      and self == self.parentWidget().currentWidget()
    )
  
  def change_controls(self):
    if self.showing():
      self.take_the_canvas()
  
  def take_the_canvas(self):
    pass

class TilingPanel(DessinControlPanel):
  def __init__(self, canvas, *args, **kwargs):
    super().__init__(canvas, *args, **kwargs)
    self.setLayout(qt.QHBoxLayout())
    
    # add order spinners
    self.order_spinners = []
    for n in canvas.covering.orders:
      spinner = qt.QSpinBox()
      spinner.setValue(n)
      spinner.valueChanged.connect(self.change_controls)
      self.layout().addWidget(spinner)
      self.order_spinners.append(spinner)
    self.set_minimums()
  
  # set the spinner minimums so that any allowed change to a single spinner will
  # keep the tiling hyperbolic. the hyperbolicity condition is
  #
  #   p*q*r - q*r - r*p - p*q > 0
  #
  # for vertex orders p, q, r
  def set_minimums(self):
    # to keep the tiling hyperbolic, each vertex order has to stay above
    #
    #   m*n / (m*n - m - n),
    #
    # where `m` and `n` are the orders of the other two vertices. we set its
    # minimum to the smallest integer greater than this value. the `floor` in
    # our implementation is safe for integer ratios because IEEE floating-point
    # division is exact for all integers from 0 to 2^(# significand bits)
    #
    #   Daniel Lemire, "Fast exact integer divisions using floating-point operations"
    #   https://lemire.me/blog/2017/11/16/fast-exact-integer-divisions-using-floating-point-operations/
    #
    orders = [spinner.value() for spinner in self.order_spinners]
    for k in range(3):
      m = orders[(k+1)%3]
      n = orders[(k+2)%3]
      self.order_spinners[k].setMinimum(floor(1 + m*n / (m*n - m - n)))
  
  def change_controls(self):
    super().change_controls()
    self.set_minimums()
  
  def take_the_canvas(self):
    self.canvas.load_empty_tree(True)
    self.canvas.set_tiling(*[
      spinner.value()
      for spinner in self.order_spinners
    ])
    self.canvas.set_working(False)
    self.canvas.set_selection(None)
    self.canvas.update()

class PermutationValidator(QValidator):
  pmt_format = QRegExp(r'(\((\d+,)*\d+\))+')
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, *kwargs)
  
  def validate(self, input, pos):
    if self.pmt_format.exactMatch(input):
      return (self.Acceptable, input, pos)
    else:
      return (self.Intermediate, input, pos)

class WorkingPanel(DessinControlPanel):
  def __init__(self, canvas, *args, **kwargs):
    super().__init__(canvas, *args, **kwargs)
    self.setLayout(qt.QVBoxLayout())
    
    # start list of working domains
    self.domains = []
    
    # add domain entry bar
    entry_bar = qt.QWidget()
    entry_bar.setLayout(qt.QHBoxLayout())
    self.pmt_fields = []
    pmt_validator = PermutationValidator()
    for n in range(3):
      field = qt.QLineEdit()
      field.setValidator(pmt_validator)
      field.textChanged.connect(self.check_entry_format)
      field.returnPressed.connect(self.new_domain)
      self.pmt_fields.append(field)
      entry_bar.layout().addWidget(field)
    self.orbit_field = qt.QLineEdit()
    self.orbit_field.setMaximumWidth(30)
    self.orbit_field.textChanged.connect(self.check_entry_format)
    self.orbit_field.returnPressed.connect(self.new_domain)
    self.tag_field = qt.QLineEdit()
    self.tag_field.returnPressed.connect(self.new_domain)
    self.new_button = qt.QPushButton('New')
    self.new_button.setEnabled(False)
    self.new_button.clicked.connect(self.new_domain)
    entry_bar.layout().addWidget(self.orbit_field)
    entry_bar.layout().addWidget(self.tag_field)
    entry_bar.layout().addWidget(self.new_button)
    self.layout().addWidget(entry_bar)
    
    # add domain chooser bar
    chooser_bar = qt.QWidget()
    chooser_bar.setLayout(qt.QHBoxLayout())
    self.save_button = qt.QPushButton('Save')
    self.save_button.setEnabled(False)
    self.domain_box = qt.QComboBox()
    self.save_button.clicked.connect(self.save_domain)
    self.domain_box.currentTextChanged.connect(self.change_controls)
    self.domain_box.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)
    chooser_bar.layout().addWidget(self.save_button)
    chooser_bar.layout().addWidget(self.domain_box)
    self.layout().addWidget(chooser_bar)
    
    # create error dialog
    self.error_dialog = qt.QMessageBox()
  
  def check_entry_format(self):
    self.new_button.setEnabled(
      all([field.hasAcceptableInput() for field in self.pmt_fields])
      and self.orbit_field.text() != ''
    )
  
  def new_domain(self):
    try:
      cycle_strs = [field.text() for field in self.pmt_fields]
      orbit = self.orbit_field.text()
      tag = self.tag_field.text()
      dessin = Dessin(cycle_strs, orbit, 20, tag if tag else None)
    except Exception as ex:
      self.error_dialog.setText("Error computing dessin metadata.")
      self.error_dialog.setDetailedText(str(ex))
      self.error_dialog.exec()
    else:
      p, q, r = dessin.domain.orders
      if p*q*r - q*r - r*p - p*q > 0:
        # add new domain
        self.domains.append(dessin.domain)
        box = self.domain_box
        box.addItem(dessin.domain.name())
        
        # set new covering. since `covering` has the same orders as `domain`,
        # we'll automatically skip the covering computation step in the
        # subsequent call to
        #
        #   change_controls -> take_the_canvas -> set_domain
        #
        self.canvas.set_covering(dessin.covering)
        
        # choose new domain
        box.setCurrentIndex(box.count()-1)
        self.change_controls()
      else:
        self.error_dialog.setText('Order triple {}, {}, {} not of hyperbolic type.'.format(p, q, r))
        self.error_dialog.setDetailedText(None)
        self.error_dialog.exec()
  
  def change_controls(self):
    super().change_controls()
    self.save_button.setEnabled(self.domain_box.currentText() != '')
  
  def take_the_canvas(self):
    index = self.domain_box.currentIndex()
    if index < 0:
      self.canvas.set_domain(None, working=False)
    else:
      self.canvas.set_domain(self.domains[index], working=True)
  
  def save_domain(self):
    index = self.domain_box.currentIndex()
    domain = self.domains[index]
    try:
      with open('domains/' + domain.name() + '.json', 'w') as file:
        domain.dump(file)
    except (TypeError, ValueError, OSError) as ex:
      self.error_dialog.setText('Error saving file.')
      self.error_dialog.setDetailedText(str(PicklingError))
      self.error_dialog.exec()

class SavedPanel(DessinControlPanel):
  def __init__(self, canvas, *args, **kwargs):
    super().__init__(canvas, *args, **kwargs)
    self.setLayout(qt.QHBoxLayout())
    
    # add domain chooser bar
    self.passport_box = qt.QComboBox()
    self.orbit_box = qt.QComboBox()
    self.domain_box = qt.QComboBox()
    self.passport_box.currentTextChanged.connect(self.list_orbits)
    self.orbit_box.currentTextChanged.connect(self.list_domains)
    self.domain_box.currentTextChanged.connect(self.change_controls)
    self.passport_box.setMinimumContentsLength(18)
    self.passport_box.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLength)
    self.orbit_box.setMinimumContentsLength(1)
    self.orbit_box.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLength)
    self.domain_box.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)
    for box in [self.passport_box, self.orbit_box, self.domain_box]:
      self.layout().addWidget(box)
    
    # open saved domains
    self.domains = {}
    for filename in os.listdir('domains'):
      if re.match(r'.*\.json$', filename):
        try:
          with open('domains/' + filename, 'r') as file:
            dom = Domain.load(file)
        except (json.JSONDecodeError, OSError) as ex:
          print(ex)
        else:
          if not dom.passport in self.domains:
            self.domains[dom.passport] = {}
          if not dom.orbit in self.domains[dom.passport]:
            self.domains[dom.passport][dom.orbit] = []
          self.domains[dom.passport][dom.orbit].append(dom)
    
    # list passports. when we add the first one, the resulting
    # `currentTextChanged` signal will call `list_orbits`
    for passport in self.domains:
      self.passport_box.addItem(passport)
  
  def list_orbits(self, passport):
    self.orbit_box.blockSignals(True)
    self.orbit_box.clear()
    self.orbit_box.blockSignals(False)
    
    # when we add the first orbit to `orbit_box`, the resulting
    # `currentTextChanged` signal will call `list_domains`
    if passport:
      for orbit in self.domains[passport]:
        self.orbit_box.addItem(orbit)
  
  def list_domains(self, orbit):
    self.domain_box.blockSignals(True)
    self.domain_box.clear()
    self.domain_box.blockSignals(False)
    
    if orbit:
      passport = self.passport_box.currentText()
      for domain in self.domains[passport][orbit]:
        permutation_str = ','.join([s.cycle_string() for s in domain.group.gens()])
        if domain.tag == None:
          self.domain_box.addItem(permutation_str)
        else:
          self.domain_box.addItem('-'.join([permutation_str, domain.tag]))
  
  def take_the_canvas(self):
    passport = self.passport_box.currentText()
    if passport:
      orbit = self.orbit_box.currentText()
      index = self.domain_box.currentIndex()
      self.canvas.set_domain(self.domains[passport][orbit][index], working=False)
    else:
      self.canvas.set_domain(None)

class DomainExplorer(qt.QMainWindow):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.setWindowTitle('Chorno-Belyi')
    self.resize(700, 900)
    
    # set up central panel
    central = qt.QWidget()
    central.setLayout(qt.QVBoxLayout())
    self.setCentralWidget(central)
    
    # add domain canvas
    self.canvas = DomainCanvas(4, 4, 3, size=(1200, 1200))
    self.canvas.native.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
    central.layout().addWidget(self.canvas.native)
    
    # add work info bar
    work_info_bar = qt.QWidget()
    work_info_bar.setLayout(qt.QHBoxLayout())
    self.canvas.selection_display = qt.QLabel()
    self.canvas.paint_display = qt.QLabel()
    self.canvas.paint_display.setMaximumWidth(40)
    self.canvas.paint_display.setAlignment(Qt.AlignCenter)
    export_button = qt.QPushButton("Export")
    export_button.clicked.connect(self.export_image)
    work_info_bar.layout().addWidget(self.canvas.selection_display)
    work_info_bar.layout().addWidget(self.canvas.paint_display)
    work_info_bar.layout().addWidget(export_button)
    central.layout().addWidget(work_info_bar)
    
    # set up control panels for tilings, working domains, and saved domains
    tiling_panel = TilingPanel(self.canvas)
    working_panel = WorkingPanel(self.canvas)
    saved_panel = SavedPanel(self.canvas)
    
    # add mode tabs
    self.control_panels = qt.QTabWidget()
    self.control_panels.addTab(tiling_panel, "Tiling")
    self.control_panels.addTab(working_panel, "Working domain")
    self.control_panels.addTab(saved_panel, "Saved domain")
    self.control_panels.currentChanged.connect(self.change_mode)
    central.layout().addWidget(self.control_panels)
  
  def change_mode(self, index):
    self.control_panels.currentWidget().take_the_canvas()
  
  def export_image(self):
    image = self.canvas.render()
    if self.canvas.domain:
      name = self.canvas.domain.name()
    else:
      name = "tiling-(" + ','.join(map(str, self.canvas.orders)) + ")"
    io.write_png("export/" + name + ".png", image)

if __name__ == '__main__' and sys.flags.interactive == 0:
  # set OpenGL version and profile
  format = QSurfaceFormat()
  format.setVersion(4, 1)
  format.setProfile(QSurfaceFormat.CoreProfile)
  QSurfaceFormat.setDefaultFormat(format)
  
  main_app = qt.QApplication(sys.argv)
  window = DomainExplorer()
  window.show()
  main_app.exec_()
