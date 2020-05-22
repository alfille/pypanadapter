""" Paul H Alfille version
Based on PyPanadapter forked from https://github.com/mcogoni/pypanadapter
Marco Cogoni's excellent work
This version does:
1. Menu system
2. Slidable panels
3. Change radios
4. command line arguments
5. Also upgraded to Qt5 aned python3
"""


from rtlsdr import *
from time import sleep
import math
import random
import numpy as np
from scipy.signal import welch, decimate
import pyqtgraph as pg
#import pyaudio
#from PyQt4 import QtCore, QtGui
from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import argparse # for parsing the command line


#FS = 2.4e6 # Sampling Frequency of the RTL-SDR card (in Hz) # DON'T GO TOO LOW, QUALITY ISSUES ARISE
N_AVG = 128 # averaging over how many spectra

class Device():
    @classmethod
    def List(cls): # List of types
        return [c for c in cls.__subclasses__()]
            
    @classmethod
    def Match(cls,name): # List of types
        cl = cls.List()
        for c in cl:
            if name == c.__name__:
                return c
        return cl[0]

class Radio(Device):
    name = "None"
    
class TS180S(Radio):
    name = "Kenwood TS-180S"
    # Intermediate Frquency
    IF = 8.8315E6
    
class X5105(Radio):
    name = "Xiegu X5105"
    # Intermediate Frquency
    IF = 70.455E6

class TS480(Radio):
    name = "Kenwood TS-480SAT"
    # Intermediate Frquency
    IF = 73.095E6

class Custom(Radio):
    name = "Custom Radio"
    # Intermediate Frquency
    def __init(self,IF):
        self.IF = IF

class SDR(Device):
    name = "None"
    def __init(self):
        self.driver = None
        
    def Close(self):
        self.driver = None

class RTLSDR(SDR):
    SampleRate = 2.56E6  # Sampling Frequency of the RTL-SDR card (in Hz) # DON'T GO TOO LOW, QUALITY ISSUES ARISE
    name = "RTL-SDR"
    def __init__(self):
        self.driver = RtlSdr()
        self.driver.sample_rate = self.SampleRate
        
    def SetFrequency(self, IF ):
        if IF < 30.E6 :
            #direct sampling needed (ADC Q vs 1=ADC I)
            self.driver.set_direct_sampling(2)
        else:
            self.driver.set_direct_sampling(0)
            
        self.driver.center_freq = IF
        
    def Read(self,size):
        # IQ inversion to correct low-high frequencies
        return np.flip(self.driver.read_samples(size))
        
    def Close(self):
        if self.driver:
            self.driver.close()
        self.driver = None

class RandSDR(SDR):
    SampleRate = 2.56E6  # Sampling Frequency of the RTL-SDR card (in Hz) # DON'T GO TOO LOW, QUALITY ISSUES ARISE
    name = "Random"
    def __init__(self):
        pass
        
    def SetFrequency(self, IF ):
        pass
        
    def Read(self,size):
        return 2*(np.random.random(int(size))+np.random.random(int(size))*1j)-(1.+1.j)
        
    def Close(self):
        self.driver = None
 
class appState:
    def __init__( self, sdr_class=None, radio_class=None ):
        self._sdr_class = sdr_class
        self._radio_class = radio_class
        self._resetNeeded = False
        self._Loop = True # for initial entry into loop

    @property
    def Loop(self):
        l = self._Loop
        self._Loop = False # Stay false after read
        return l
                
    @Loop.setter
    def Loop(self,value):
        self._Loop = value

    @property
    def resetNeeded(self):
        rn = self._resetNeeded
        self._resetNeeded = False
        return rn
    
    @property
    def radio_class( self ):
        return self._radio_class
        
    @radio_class.setter
    def radio_class( self, radio_class ):
        if radio_class != self._radio_class:
            self._resetNeeded = True
        self._radio_class = radio_class

    @property    
    def sdr_class( self ):
        return self._sdr_class
        
    @sdr_class.setter
    def sdr_class( self, sdr_class ):
        if sdr_class != self._sdr_class:
            self._resetNeeded = True
        self._sdr_class = sdr_class

#global
AppState = appState()

class PanAdapter():
    def __init__(self ):
        self.mode = 0 # LSB or USB
        global AppState
        
        self.radio_class = AppState.radio_class # The radio
        
        # configure device
        self.sdr_class = AppState.sdr_class # the SDR panadapter
        print(f'Starting PAN radio {self.radio_class.name} sdr {self.sdr_class.name}\n')
        
        # open sdr (or at least try
        try:
            self.sdr = self.sdr_class()
        except:
            print("Could not open sdr {self.sdr_class} -- switch to random\n")
            AppState.sdr_class = RandSDR
            self.sdr_class = AppState.sdr_class
            print(f'Starting PAN radio {self.radio_class.name} sdr {self.sdr_class.name}\n',)
            self.sdr = self.sdr_class()
            
        self.changef( self.radio_class.IF )
        self.widget = SpectrogramWidget(self.sdr)
                
    def read(self):
        self.widget.read_collected.emit(self.sdr.Read(self.widget.N_AVG*self.widget.N_FFT))

    def changef(self, F_SDR):
        self.sdr.SetFrequency( F_SDR )


    def update_mode(self):
        if self.widget.mode!=self.mode:
            sign = (self.widget.mode-self.mode)
            sign /= math.fabs(sign)
            if sign<0:
                sign = 0
            self.changef(self.radio.IF-sign*3000)
            self.mode = self.widget.mode
    
    def close(self):
        self.sdr.Close()
        
    def Loop(self, y_n ):
        global AppState
        AppState.Loop=y_n
        self.qApp.quit()

class SpectrogramWidget(QtWidgets.QMainWindow):

    #define a custom signal
    read_collected = QtCore.pyqtSignal(np.ndarray)

    def __init__(self,sdr):
        super(SpectrogramWidget, self).__init__()
        
        global AppState
        self.sdr = sdr

        self.init_ui()
        self.qt_connections()

        self.N_FFT = 2048 # FFT bins
        self.N_WIN = 1024  # How many pixels to show from the FFT (around the center)
        self.N_AVG = N_AVG
        self.fft_ratio = 2.

        self.makeWaterfall()
        self.makeSpectrum()

        pg.setConfigOptions(antialias=False)

        self.mode = 0 # USB=0, LSB=1: defaults to USB
        self.scroll = -1
        
        self.init_image()
        self.makeMenu()

        self.read_collected.connect(self.update)

        self.show()

    def makeWaterfall( self ):
        self.waterfall = pg.ImageItem()
        self.plotwidget1.addItem(self.waterfall)
        c=QtGui.QCursor()

        self.plotwidget1.hideAxis("left")
        #self.plotwidget1.hideAxis("bottom")

        # RED-GREEN Colormap
        pos = np.array([0., 0.5, 1.])
        color = np.array([[0,0,0,255], [0,255,0,255], [255,0,0,255]], dtype=np.ubyte)

        # MATRIX Colormap
        pos = np.array([0., 1.])
        color = np.array([[0,0,0,255], [0,255,0,255]], dtype=np.ubyte)

        # BLUE-YELLOW-RED Colormap
        pos = np.array([0.,                 0.4,              1.])
        color = np.array([[0,0,90,255], [200,2020,0,255], [255,0,0,255]], dtype=np.ubyte)

        cmap = pg.ColorMap(pos, color)
        pg.colormap
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        self.minlev = 220
        self.maxlev = 140

        # set colormap
        self.waterfall.setLookupTable(lut)
        self.waterfall.setLevels([self.minlev, self.maxlev])

        # setup the correct scaling for x-axis
        bw_hz = self.sdr.SampleRate/float(self.N_FFT) * float(self.N_WIN)/1.e6/self.fft_ratio
        self.waterfall.scale(bw_hz,1)
#        self.setLabel('bottom', 'Frequency', units='kHz')
        
#        self.text_leftlim = pg.TextItem("-%.1f kHz"%(bw_hz*self.N_WIN/2.))
#        self.text_leftlim.setParentItem(self.waterfall)
#        self.plotwidget1.addItem(self.text_leftlim)
#        self.text_leftlim.setPos(0, 0)

#        self.text_rightlim = pg.TextItem("+%.1f kHz"%(bw_hz*self.N_WIN/2.))
#        self.text_rightlim.setParentItem(self.waterfall)
#        self.plotwidget1.addItem(self.text_rightlim)
#        self.text_rightlim.setPos(bw_hz*(self.N_WIN-64), 0)

    def changeRadio( self, radio_class ):
        global AppState
        AppState.radio_class = radio_class
        if AppState.resetNeeded:
            self.Loop(True)

    def makeMenu( self ):
        global AppState
        menu = self.menuBar()

        filemenu = menu.addMenu('&File')

        m = QtWidgets.QAction('&Restart',self)
        m.triggered.connect(lambda state, y=True: self.Loop(y))
        filemenu.addAction(m)

        m = QtWidgets.QAction('&Quit',self)
        m.triggered.connect(lambda state, y=False: self.Loop(y))
        filemenu.addAction(m)

        radiomenu = menu.addMenu('&Radio')

        for r in Radio.List():
            #print(f'IF={AppState.radio.IF} list={type(r)} Actual={type(AppState.radio)}')
            y = AppState.radio_class == r
            m = QtWidgets.QAction(r.name,self,checkable=y,checked=y)
#            m.setObjectName(r.name)
            m.triggered.connect(lambda state,nr=r: self.changeRadio(nr))
            radiomenu.addAction(m)
        viewmenu = menu.addMenu('&View')

        v1 = QtWidgets.QAction('Waterfall',self,checkable=True,checked=True)
        v2 = QtWidgets.QAction('Spectrogram',self,checkable=True,checked=True)
        v1.triggered.connect(lambda state: self.TogglePanel(self.plotwidget1,v1,v2))
        v2.triggered.connect(lambda state: self.TogglePanel(self.plotwidget2,v1,v2))
        viewmenu.addAction(v2)
        viewmenu.addAction(v1)
  
    def TogglePanel( self, panel, menu1, menu2 ):
        if panel.isHidden():
            panel.setVisible( True )
            menu1.setDisabled( False )
            menu2.setDisabled( False )
        else:
            panel.setVisible( False )
            if panel != self.plotwidget1:
                menu1.setDisabled( True )
            else:
                menu2.setDisabled( True )
        
    def makeSpectrum( self ):
        self.spectrum_plot = self.plotwidget2.plot()
        self.plotwidget2.setYRange(-250, -100, padding=0.)
        #self.plotwidget2.showGrid(x=True, y=True)

        self.plotwidget2.hideAxis("left")
        self.plotwidget2.hideAxis("bottom")

    def init_image(self):
        self.img_array = 250*np.ones((self.N_WIN//4, self.N_WIN))
        # Plot the grid
        for x in [0, self.N_WIN//2, self.N_WIN-1]:
            if x==0 or x==self.N_WIN-1:
                #pass
                self.img_array[:,x] = 0
            else:
                #pass
                self.img_array[:,x] = 0


    def init_ui(self):
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle('PEPYSCOPE - IS0KYB')
        
        self.split = QtWidgets.QSplitter()
        self.split.setOrientation(QtCore.Qt.Vertical)
        
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.split)

        self.plotwidget2 = pg.PlotWidget()
        self.split.addWidget(self.plotwidget2)

        self.plotwidget1 = pg.PlotWidget()
        self.split.addWidget(self.plotwidget1)

        hbox = QtWidgets.QHBoxLayout()

        self.zoominbutton = QtWidgets.QPushButton("ZOOM IN")
        self.zoomoutbutton = QtWidgets.QPushButton("ZOOM OUT")
        self.avg_increase_button = QtWidgets.QPushButton("AVG +")
        self.avg_decrease_button = QtWidgets.QPushButton("AVG -")
        self.modechange = QtWidgets.QPushButton("USB")
        self.invertscroll = QtWidgets.QPushButton("Scroll")
        self.autolevel = QtWidgets.QPushButton("Auto Levels")

        hbox.addWidget(self.zoominbutton)
        hbox.addWidget(self.zoomoutbutton)
        hbox.addWidget(self.modechange)
        hbox.addWidget(self.invertscroll)
        hbox.addStretch()

        hbox.addWidget(self.autolevel)
        hbox.addWidget(self.avg_increase_button)
        hbox.addWidget(self.avg_decrease_button)

        vbox.addLayout(hbox)
        self.win.setLayout(vbox)

        self.setGeometry(10, 10, 1024, 512)
        self.setCentralWidget(self.win)
        
        self.StatBar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.StatBar)

    def qt_connections(self):
        self.zoominbutton.clicked.connect(self.on_zoominbutton_clicked)
        self.zoomoutbutton.clicked.connect(self.on_zoomoutbutton_clicked)
        self.modechange.clicked.connect(self.on_modechange_clicked)
        self.invertscroll.clicked.connect(self.on_invertscroll_clicked)
        self.avg_increase_button.clicked.connect(self.on_avg_increase_clicked)
        self.avg_decrease_button.clicked.connect(self.on_avg_decrease_clicked)
        self.autolevel.clicked.connect(self.on_autolevel_clicked)

    def on_avg_increase_clicked(self):
        if self.N_AVG<512:
            self.N_AVG *= 2
        print( self.N_AVG )

    def on_avg_decrease_clicked(self):
        if self.N_AVG>1:
            self.N_AVG /= 2
        print( self.N_AVG )


    def on_modechange_clicked(self):
        if self.mode == 0:
            self.modechange.setText("LSB")
        elif self.mode == 1:
            self.modechange.setText("USB")
        self.mode += 1
        if self.mode>1:
            self.mode = 0

    def on_autolevel_clicked(self):
        tmp_array = np.copy(self.img_array[self.img_array>0])
        tmp_array = tmp_array[tmp_array<250]
        tmp_array = tmp_array[:]
        print( tmp_array.shape )

        self.minminlev = np.percentile(tmp_array, 99)
        self.minlev = np.percentile(tmp_array, 80)
        self.maxlev = np.percentile(tmp_array, 0.3)
        print( self.minlev, self.maxlev )
        self.waterfall.setLevels([self.minlev, self.maxlev])

        self.plotwidget2.setYRange(-self.minminlev, -self.maxlev, padding=0.3)

    def on_invertscroll_clicked(self):
        self.scroll *= -1
        self.init_image()

    def on_zoominbutton_clicked(self):
        if self.fft_ratio<512:
            self.fft_ratio *= 2
        #self.waterfall.scale(0.5,1)
    
    def on_zoomoutbutton_clicked(self):
        if self.fft_ratio>1:
            self.fft_ratio /= 2
        #self.waterfall.scale(2.0,1)
 
    def zoomfft(self, x, ratio = 1):
        f_demod = 1.
        t_total = (1/self.sdr.SampleRate) * self.N_FFT * self.N_AVG
        t = np.arange(0, t_total, 1 / self.sdr.SampleRate)
        lo = 2**.5 * np.exp(-2j*np.pi*f_demod * t) # local oscillator
        x_mix = x*lo
        
        power2 = int(np.log2(ratio))
        for mult in range(power2):
            x_mix = decimate(x_mix, 2) # mix and decimate

        return x_mix 


    def update(self, chunk):
        bw_hz = self.sdr.SampleRate/float(self.N_FFT) * float(self.N_WIN)
        self.win.setWindowTitle('PEPYSCOPE - IS0KYB - N_FFT: %d, BW: %.1f kHz' % (self.N_FFT, bw_hz/1000./self.fft_ratio))

        if self.fft_ratio>1:
            chunk = self.zoomfft(chunk, self.fft_ratio)

        sample_freq, spec = welch(chunk, self.sdr.SampleRate, window="hamming", nperseg=self.N_FFT,  nfft=self.N_FFT)
        spec = np.roll(spec, self.N_FFT//2, 0)[self.N_FFT//2-self.N_WIN//2:self.N_FFT//2+self.N_WIN//2]
        
        # get magnitude 
        psd = abs(spec)
        # convert to dB scale
        psd = -20 * np.log10(psd)

        # Plot the grid
        for x in [0, self.N_WIN//2, self.N_WIN-1]:
            #pass
            psd[x] = 0            

        # roll down one and replace leading edge with new data
        self.img_array[-1:] = psd
        self.img_array = np.roll(self.img_array, -1*self.scroll, 0)

        for i, x in enumerate(range(0, self.N_WIN-1, ((self.N_WIN)//10))):
            if i!=5 and i!=10:
                if self.scroll>0:
                    for y in range(5,15):
                        #pass
                        self.img_array[y,x] = 0
                elif self.scroll<0:
                    for y in range(-10,-2):
                        #pass
                        self.img_array[y,x] = 0

        #self.spectrum_plot.plot()

        self.waterfall.setImage(self.img_array.T, autoLevels=False, opacity = 1.0, autoDownsample=True)

#        self.text_leftlim.setPos(0, 0)
#        self.text_leftlim.setText(text="-%.1f kHz"%(bw_hz/2000./self.fft_ratio))
#        #self.text_rightlim.setPos(bw_hz*1000, 0)
#        self.text_rightlim.setText(text="+%.1f kHz"%(bw_hz/2000./self.fft_ratio))

        self.spectrum_plot.setData(np.arange(0,psd.shape[0]), -psd, pen="g")

        #self.plotwidget2.plot(x=[0,0], y=[-240,0], pen=pg.mkPen('r', width=1))
        #self.plotwidget2.plot(x=[self.N_WIN/2, self.N_WIN//2], y=[-240,0], pen=pg.mkPen('r', width=1))
        #self.plotwidget2.plot(x=[self.N_WIN-1, self.N_WIN-1], y=[-240,0], pen=pg.mkPen('r', width=1))

    def Loop(self, y_n ):
        global AppState
        AppState.Loop=y_n
        QtWidgets.qApp.quit()

def CommandLine():
    """Setup argparser object to process the command line"""
    cl = argparse.ArgumentParser(description="PyPanadapter - radio panadapter using an SDR dongle on the IF (intermediate frequency of a radio by Paul H Alfille based on code of Marco Cogoni")
    cl.add_argument("-s","--sdr",help="SDR model",choices=[c.__name__ for c in SDR.List()],nargs='?',default=SDR.List()[0].__name__)
    cl.add_argument("-r","--radio",help="Radio model",choices=[r.__name__ for r in Radio.List()],nargs='?',default=Radio.List()[0].__name__)
    cl.add_argument("-i","--if",help="Intermediate frequency -- overwrites radio default",type=float)
    return cl.parse_args()

def main(args):
    args = CommandLine() # Get args from command line

    AppState.sdr_class = SDR.Match( args.sdr )
    AppState.radio_class = Radio.Match( args.radio )
    
    t = QtCore.QTimer()

    while AppState.Loop:
        app = QtWidgets.QApplication([])
        pan = PanAdapter()
        t.timeout.connect(pan.update_mode)
        t.timeout.connect(pan.read)
        t.start(50) # max theoretical refresh rate 100 fps
        app.exec_()
        t.stop()
        app = None

if __name__ == '__main__':
#    for c in Radio.List():
#        print(c.__name__)
    sys.exit(main(sys.argv))
