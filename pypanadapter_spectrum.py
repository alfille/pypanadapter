#/usr/bin/env python3

""" Paul H Alfille version
Based on PyPanadapter forked from https://github.com/mcogoni/pypanadapter
Marco Cogoni's excellent work
This version does:
1. Menu system
2. Slidable panels
3. Change radios
4. command line arguments
5. Also upgraded to Qt5 and python3
"""


#from rtlsdr import *
from time import sleep
import math
import random
import numpy as np
from scipy.signal import welch, decimate
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui, QtDBus
import sys
import argparse # for parsing the command line

try:
    import pyaudio
    Flag_audio = True
except:
    print("Could not find pyaudio module -- try <pip3 install pyaudio>")
    Flag_audio = False
    
try:
    import random
    Flag_random = True
except:
    print("Could not find random module -- try <pip3 install random>")
    Flag_random = False
    
try:
    import rtlsdr
    Flag_rtlsdr = True
except:
    print("Could not find RTLSDR module -- try <pip3 install rtlsdr>")
    Flag_rtlsdr = False
    
try:
    import SoapySDR
    Flag_soapy = True
except:
    print("Could not find SoapySDR module -- see https://github.com/pothosware/SoapySDR/wiki/PythonSupport")
    Flag_soapy = False
    
#FS = 2.4e6 # Sampling Frequency of the RTL-PanClass card (in Hz) # DON'T GO TOO LOW, QUALITY ISSUES ARISE
Initial_N_AVG = 128 # averaging over how many spectra

class SubclassManager():
    # base class that gives subclass list and matching
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

class Radio(SubclassManager):
    # device being pan-adapterd. Have make / model / intermediate frequency
    make = "None"
    model = "None"
    
class TS180S(Radio):
    make = "Kenwood"
    model = "TS-180S"
    # Intermediate Frquency
    IF = 8.8315E6
    
class X5105(Radio):
    make = "Xiegu"
    model = "X5105"
    # Intermediate Frquency
    IF = 70.455E6

class TS480(Radio):
    make = "Kenwood"
    model = "TS-480SAT"
    # Intermediate Frquency
    IF = 73.095E6

class Custom(Radio):
    make = "Custom"
    model = "Specified"
    # Intermediate Frquency
    def __init(self,IF):
        self.IF = IF

class PanClass(SubclassManager):
    # Pan adapter device including RTLSDR
    name = "None"
    def __init(self):
        self.driver = None
        
    def Close(self):
        self.driver = None

class RTLSDR(PanClass):
    SampleRate = 2.56E6  # Sampling Frequency of the RTL-PanClass card (in Hz) # DON'T GO TOO LOW, QUALITY ISSUES ARISE
    name = "RTLSDR"
    def __init__(self,serial=None,index=None,host=None,port=12345):
        if not Flag_rtlsdr:
            raise ValueError
        self.serial = serial
        self.index = index
        self.host = host
        self.port = port

        try:
            if self.serial:
                self.driver = RtlSdr(serial_number = self.serial)
            elif self.index:
                self.driver = RtlSdr(self.index)
            elif self.host:
                self.driver = RtlSdrTcpClient( hostname=self.host, port=self.port ) 
            else:
                self.driver = RtlSdr()
        except:
            print("RTLSDR not found")
            raise
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

class AudioPan(PanClass):
    SampleRate = 44100.0
    name = "Audio"
    def __init__(self,index):
        global AppState
        if AppState.audio:
            try:
                info = AppState.audio.get_device_info_by_index( index)
                self.name = info['name']
                type(self).name = self.name
                self.SampleRate = info['defaultSampleRate']
                type(self).SampleRate = self.SampleRate
                self.driver = AppState.audio.open(format=pyaudio.paFloat32, channels=1, rate=int(self.SampleRate), input=True, output=False, input_device_index = index )
            except:
                print("Could not open audio device")
                self.driver = None
        self.index = index
        
    def SetFrequency(self, IF ):
        self.center_freq = IF
        
    def Read(self,size):
        if self.driver:
            # I only
            try:
                return np.fromstring( self.driver.read(size), 'float32' )
            except:
                pass
        return np.zeros( (size,), 'float32' )+.000000000001
        
        
    def Close(self):
        if self.driver:
            self.driver.close()
        self.driver = None

class RandomPan(PanClass):
    SampleRate = 2.56E6  # Sampling Frequency of the RTL-PanClass card (in Hz) # DON'T GO TOO LOW, QUALITY ISSUES ARISE
    name = "Random"
    def __init__(self):
        if not Flag_random:
            raise ValueError
        
    def SetFrequency(self, IF ):
        pass
        
    def Read(self,size):
        return 2*(np.random.random(int(size))+np.random.random(int(size))*1j)-(1.+1.j)
        
    def Close(self):
        self.driver = None

class ConstgantPan(PanClass):
    SampleRate = 2.56E6  # Sampling Frequency of the RTL-PanClass card (in Hz) # DON'T GO TOO LOW, QUALITY ISSUES ARISE
    name = "Constant"
    def __init__(self):
        pass
        
    def SetFrequency(self, IF ):
        pass
        
    def Read(self,size):
        return 2*(np.zeroes(int(size)))+.5
        
    def Close(self):
        self.driver = None

class TransmissionMode(SubclassManager):
    # Signal types
    _changed = False
    _mode = None
    _list = None
    _index = 0
    _len = 0

    @classmethod
    def next(cls):
        if cls._list == None:
            cls._list = cls.List()
            cls._len = len(cls._list)
        cls._index += 1
        if cls._index >= cls._len:
            cls._index = 0
        cls._changed = True

    @classmethod
    def changed(cls):
        c = cls._changed
        cls._changed = False
        return c
        
    @classmethod
    def mode(cls):
        return cls._list[cls._index]

class AM(TransmissionMode):
    @classmethod
    def freq(cls,radio_class):
        return radio_class.IF
        
class USB(TransmissionMode):
    @classmethod
    def freq(cls,radio_class):
        return radio_class.IF-3000
        
class LSB(TransmissionMode):
    @classmethod
    def freq(cls,radio_class):
        return radio_class.IF+3000
        
 
class appState:
    # holds program "state"
    refresh = 50 # default refresh timer in msec
    def __init__( self, panadapter=None, radio_class=None ):
        self._panadapter = panadapter
        self._radio_class = radio_class
        self._resetNeeded = False
        self._Loop = True # for initial entry into loop
        self.timer = QtCore.QTimer() #default
        self.refresh = type(self).refresh
        self._soapylist = {}
        self.discover = None
        if Flag_audio:
            self.audio = pyaudio.PyAudio()
        else:
            self.audio = None

    def Start(self,proc):
        self.timer.timeout.connect(proc)
        self.timer.start(self.refresh)
        
    def Stop(self):
        self.timer.stop()

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
    def panadapter( self ):
        return self._panadapter
        
    @panadapter.setter
    def panadapter( self, panadapter ):
        if panadapter != self._panadapter:
            self._resetNeeded = True
        self._panadapter = panadapter
        
    def SoapyAdd( self, address, port, name ):
        #print("add",address,port,name)
        self._soapylist[(address,port)] = name
        
    def SoapyDel( self, address, port ):
        #print("del",address,port,name)
        del self._soapylist[(address,port)]
        
    @property
    def soapylist( self ):
        return self._soapylist
        
    @property
    def audiolist(self):
        alist = {}
        if self.audio:
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                try:
                    if info['maxInputChannels'] > 0:
                        n = info['name']
                        r = info['defaultSampleRate']
                        alist[i] = (n,r)
                except:
                    pass
        return alist
                    

#global
AppState = appState()

class ApplicationDisplay():
    # container class
    def __init__(self ):
        # Comes in with Panadapter set and readio_class set.
        global AppState
        
        self.radio_class = AppState.radio_class # The radio
        
        # configure device
        self.panadapter = AppState.panadapter # the PanClass panadapter
        print(f'Starting PAN radio {self.radio_class.make} / {self.radio_class.model} sdr {self.panadapter.name}\n')
        
        self.changef( self.radio_class.IF )
        self.widget = SpectrogramWidget()
                
    def read(self):
        global AppState
        if TransmissionMode.changed():
            self.changef(TransmissionMode.mode().freq(self.radio_class))
        self.widget.read_collected.emit(AppState.panadapter.Read(self.widget.N_AVG*self.widget.N_FFT))

    def changef(self, F_SDR):
        global AppState
        AppState.panadapter.SetFrequency( F_SDR )
    
    def close(self):
        global AppState
        AppState.panadapter.Close()
        
    def Loop(self, y_n ):
        global AppState
        AppState.Loop=y_n
        self.qApp.quit()

class Panels():
    # manage the displaypanels -- waterfall and spectrogram so far
    # some complicated logic for menu system to never allow no panels, and disable the menu entry that might allow it, to be clearer.
    # actually can handle an arbitrary number of panels
    List = []
    def __init__(self, name, func, split):
        self.name = name
        self.plot = pg.PlotWidget()
        func(self.plot)
        split.addWidget(self.plot)
        self.visible = True
        type(self).List.append( self )

    @classmethod
    def clear( cls ):
        cls.List.clear()
    
    @classmethod
    def addMenus( cls, menu, master ):
        for p in cls.List:
            v = QtWidgets.QAction(p.name,master,checkable=True,checked=p.visible)
            v.triggered.connect(p.toggle)
            p.menu = v
            menu.addAction(v)

    def toggle( self ):
        if self.visible:
            self.plot.setVisible( False )
            self.visible = False
            v = [ p for p in type(self).List if p.visible ]
            if len(v) == 1:
                v[0].menu.setDisabled( True )        
        else:
            self.plot.setVisible( True )
            self.visible = True
            for p in type(self).List:
                p.menu.setDisabled( False )
        
class SpectrogramWidget(QtWidgets.QMainWindow):
    # Display class
    #define a custom signal
    read_collected = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(SpectrogramWidget, self).__init__()
        
        self.N_FFT = 2048 # FFT bins
        self.N_WIN = 1024  # How many pixels to show from the FFT (around the center)
        self.N_AVG = Initial_N_AVG
        self.fft_ratio = 2.

        self.init_ui()
        self.qt_connections()

        pg.setConfigOptions(antialias=False)

        self.scroll = -1
        
        self.init_image()
        self.makeMenu()

        self.read_collected.connect(self.update)

        self.show()

    def makeWaterfall( self, panel ):
        global AppState
        
        self.waterfall = pg.ImageItem()
        panel.addItem(self.waterfall)
        c=QtGui.QCursor()

        panel.hideAxis("left")
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
        bw_hz = AppState.panadapter.SampleRate/float(self.N_FFT) * float(self.N_WIN)/1.e6/self.fft_ratio
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

    def setRandom( self ):
        global AppState
        AppState.panadapter = RandomPan()
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

        panmenu = menu.addMenu('&Panadapter')
        self.makePanMenu(panmenu)
        
        radiomenu = menu.addMenu('&Radio')

        make_dict = {}
        for r in Radio.List():
            if r.make not in make_dict:
                make_dict[r.make] = radiomenu.addMenu(r.make)
            y = AppState.radio_class == r
            m = QtWidgets.QAction(r.model,self,checkable=y,checked=y)
            m.triggered.connect(lambda state,nr=r: self.changeRadio(nr))
            make_dict[r.make].addAction(m)

        viewmenu = menu.addMenu('&View')
        Panels.addMenus( viewmenu, self )

    def makePanMenu( self, menu ):
        global AppState

        menu.clear()
        
        m = QtWidgets.QAction('&Rescan sources',self)
        m.triggered.connect(lambda state,m=menu: self.makePanMenu(m))
        menu.addAction(m)
        
        menu.addSeparator()
        
        if AppState.discover:
            menu.addAction(QtWidgets.QAction('Remote via SoapySDR',self))
            for ((address,port),name) in AppState.soapylist.items():
                #print("Pan",name)
                m = QtWidgets.QAction('{}\t\t{} : {}'.format(name,address,port),self)
                m.triggered.connect(lambda state,a=address,p=port,n=name: self.setsoapy(a,p,n))
                menu.addAction(m)
            menu.addSeparator()
            
        if Flag_audio:
            menu.addAction(QtWidgets.QAction('Local audio sources',self))
            for (index,(name,rate)) in AppState.audiolist.items():
                #print("Pan",name)
                m = QtWidgets.QAction('{}. {}\t{}'.format(index,name,rate),self)
                m.triggered.connect(lambda state,i=index: self.setaudio(i))
                menu.addAction(m)
            menu.addSeparator()

        if Flag_random:
            m = QtWidgets.QAction('Random',self)
            m.triggered.connect(self.setrandom)
            menu.addAction(m)
            menu.addSeparator()

        if True:
            m = QtWidgets.QAction('True',self)
            m.triggered.connect(self.setrandom)
            menu.addAction(m)
            menu.addSeparator()
            
        m = QtWidgets.QAction('&Manual',self)
        m.triggered.connect(lambda state,m=menu: self.makePanMenu(m))
        menu.addAction(m)
        
    
    def setrandom( self ):
        global AppState
        AppState.panadapter = RandomPan()
        if AppState.resetNeeded:
            self.Loop(True)

    def setsoapy( self, address, port, name):
        print("setsoapy",address,port,name)

    def setaudio( self, index ):
        global AppState
        AppState.panadapter = AudioPan(index)
        if AppState.resetNeeded:
            self.Loop(True)

    def makeSpectrum( self, panel ):
        self.spectrum_plot = panel.plot()
        panel.setYRange(-250, -100, padding=0.)
        #panel.showGrid(x=True, y=True)

        panel.hideAxis("left")
        panel.hideAxis("bottom")

    def init_image(self):
        self.img_array = 250*np.ones((self.N_WIN//4, self.N_WIN))
        # Plot the grid
        for x in [0, self.N_WIN//2, self.N_WIN-1]:
            if x==0 or x==self.N_WIN-1:
                self.img_array[:,x] = 0
            else:
                self.img_array[:,x] = 0

    def init_ui(self):
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle('PEPYSCOPE - IS0KYB')
        
        self.split = QtWidgets.QSplitter()
        self.split.setOrientation(QtCore.Qt.Vertical)
        
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.split)

        Panels.clear()
        self.w_pan = Panels( "Waterfall", self.makeWaterfall, self.split )
        self.s_pan = Panels( "Spectrogram", self.makeSpectrum, self.split )

        hbox = QtWidgets.QHBoxLayout()

        self.zoominbutton = QtWidgets.QPushButton("ZOOM IN")
        self.zoomoutbutton = QtWidgets.QPushButton("ZOOM OUT")
        self.avg_increase_button = QtWidgets.QPushButton("AVG +")
        self.avg_decrease_button = QtWidgets.QPushButton("AVG -")
        self.modechange = QtWidgets.QPushButton(TransmissionMode.mode().__name__)
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
        TransmissionMode.next()
        self.modechange.setText(TransmissionMode.mode().__name__)

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

        self.panels[1].setYRange(-self.minminlev, -self.maxlev, padding=0.3)

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
        global AppState
        f_demod = 1.
        t_total = (1/AppState.panadapter.SampleRate) * self.N_FFT * self.N_AVG
        t = np.arange(0, t_total, 1 / AppState.panadapter.SampleRate)
        lo = 2**.5 * np.exp(-2j*np.pi*f_demod * t) # local oscillator
        x_mix = x*lo
        
        power2 = int(np.log2(ratio))
        for mult in range(power2):
            x_mix = decimate(x_mix, 2) # mix and decimate

        return x_mix 

    def update(self, chunk):
        global AppState
        bw_hz = AppState.panadapter.SampleRate/float(self.N_FFT) * float(self.N_WIN)
        self.win.setWindowTitle('PEPYSCOPE - IS0KYB - N_FFT: %d, BW: %.1f kHz' % (self.N_FFT, bw_hz/1000./self.fft_ratio))

        if self.fft_ratio>1:
            chunk = self.zoomfft(chunk, self.fft_ratio)

        sample_freq, spec = welch(chunk, AppState.panadapter.SampleRate, window="hamming", nperseg=self.N_FFT,  nfft=self.N_FFT)
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

# from https://gist.github.com/fladi/8bebdba4c47051afa7cad46e3ead6763 Michael Fladischer
# from https://gist.github.com/fladi/8bebdba4c47051afa7cad46e3ead6763 Michael Fladischer
class Service(QtCore.QObject):

    def __init__(
        self,
        interface,
        protocol,
        name,
        stype,
        domain,
        host=None,
        aprotocol=None,
        address=None,
        port=None,
        txt=None
    ):
        super(Service, self).__init__()
        self.interface = interface
        self.protocol = protocol
        self.name = name
        self.stype = stype
        self.domain = domain
        self.host = host
        self.aprotocol = aprotocol
        self.address = address
        self.port = port
        self.txt = txt


    def __str__(self):
        return '{s.name} ({s.stype}.{s.domain})'.format(s=self)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class Discoverer(QtCore.QObject):

    def __init__(self, parent, service, interface=-1, protocol=-1, domain='local'):
        super(Discoverer, self).__init__(parent)
        self.protocol = protocol
        self.bus = QtDBus.QDBusConnection.systemBus()
        self.bus.registerObject('/', self)
        self.server = QtDBus.QDBusInterface(
            'org.freedesktop.Avahi',
            '/',
            'org.freedesktop.Avahi.Server',
            self.bus
        )
        flags = QtCore.QVariant(0)
        flags.convert(QtCore.QVariant.UInt)
        browser_path = self.server.call(
            'ServiceBrowserNew',
            interface,
            self.protocol,
            service,
            domain,
            flags
        )
        print('New ServiceBrowser: {}'.format(browser_path.arguments()))
        self.bus.connect(
            'org.freedesktop.Avahi',
            browser_path.arguments()[0],
            'org.freedesktop.Avahi.ServiceBrowser',
            'ItemNew',
            self.onItemNew
        )
        self.bus.connect(
            'org.freedesktop.Avahi',
            browser_path.arguments()[0],
            'org.freedesktop.Avahi.ServiceBrowser',
            'ItemRemove',
            self.onItemRemove
        )
        self.bus.connect(
            'org.freedesktop.Avahi',
            browser_path.arguments()[0],
            'org.freedesktop.Avahi.ServiceBrowser',
            'AllForNow',
            self.onAllForNow
        )
        
    def __delete__(self):
        #print("Delete Discoverer")
        self.server = None
        self.bus = None

    @QtCore.pyqtSlot(QtDBus.QDBusMessage)
    def onItemNew(self, msg):
        global AppState
        #print('Avahi service discovered: {}'.format(msg.arguments()))
        flags = QtCore.QVariant(0)
        flags.convert(QtCore.QVariant.UInt)
        resolved = self.server.callWithArgumentList(
            QtDBus.QDBus.AutoDetect,
            'ResolveService',
            [
                *msg.arguments()[:5],
                self.protocol,
                flags
            ]
        ).arguments()
        try:
            #print('\tAvahi service resolved: {}'.format(resolved))
            AppState.SoapyAdd( resolved[7],resolved[8],resolved[2] )
        except:
            #print("Incomplete entry -- ignored")
            pass

    @QtCore.pyqtSlot(QtDBus.QDBusMessage)
    def onItemRemove(self, msg):
        global AppState
        arguments = msg.arguments()
        #print('Avahi service removed: {}'.format(arguments))
        AppState.SoapyDel( resolved[7],resolved[8] )

    @QtCore.pyqtSlot(QtDBus.QDBusMessage)
    def onAllForNow(self, msg):
        #print('Avahi emitted all signals for discovered peers')
        pass

def CommandLine():
    """Setup argparser object to process the command line"""
    cl = argparse.ArgumentParser(description="PyPanadapter - radio panadapter using an PanClass dongle on the IF (intermediate frequency of a radio by Paul H Alfille based on code of Marco Cogoni")
    cl.add_argument("-s","--sdr",help="Panadapter model",choices=[c.__name__ for c in PanClass.List()],nargs='?',default=PanClass.List()[0].__name__)
    cl.add_argument("-r","--radio",help="Radio model",choices=[r.__name__ for r in Radio.List()],nargs='?',default=Radio.List()[0].__name__)
    cl.add_argument("-i","--if",help="Intermediate frequency -- overwrites radio default",type=float)
    return cl.parse_args()

def main(args):
    global AppState
    args = CommandLine() # Get args from command line

    TransmissionMode.next() # prime mode list

    sdr_class = PanClass.Match( args.sdr )
    # open sdr (or at least try
    try:
        AppState.panadapter = sdr_class()
    except:
        print(f'Could not open panadapter {sdr_class.name} -- switch to random\n')
        if Flag_random:
            AppState.panadapter = RandomPan()
        else:
            AppState.panadapter = ConstantPan()

    AppState.radio_class = Radio.Match( args.radio )
    
    while AppState.Loop:
        app = QtWidgets.QApplication([])
        if not AppState.discover:
            try:
                AppState.discover = Discoverer(app, '_soapy._tcp')
            except:
                AppState.discover = None
#        print("AppState.discover",AppState.discover)
        display = ApplicationDisplay()
        AppState.Start(display.read)
        app.exec_()
        AppState.Stop()
        app = None

if __name__ == '__main__':
    sys.exit(main(sys.argv))
