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

# QT5 forced
from PyQt5 import QtCore, QtWidgets, QtGui, QtDBus

# standard libraries
import time
import math
import sys
import signal
import argparse # for parsing the command line

import scipy.signal
import scipy.special
import numpy as np
import pyqtgraph as pg

# local module (also on github independently as https://github.com/alfille/NewtRap
import newtrap

try:
    import pyaudio
    Flag_audio = True
except:
    print("Could not find pyaudio module\n\trun <pip3 install pyaudio>")
    Flag_audio = False
    
try:
    import random
    Flag_random = True
except:
    print("Could not find random module\n\trun <pip3 install random>")
    Flag_random = False
    
try:
    import rtlsdr
    Flag_rtlsdr = True
except:
    print("Could not find RTLSDR module\n\trun <pip3 install rtlsdr>")
    Flag_rtlsdr = False
    
try:
    import SoapySDR
    Flag_soapy = True
except:
    print("Could not find SoapySDR module\n\tsee https://github.com/pothosware/SoapySDR/wiki/PythonSupport")
    Flag_soapy = False
    
try:
    import usb.core
    Flag_USB = True
except:
    print("Could not find pyusb module\n\trun <pip3 pyusb>\n\tsee https://github.com/pyusb/pyusb")
    Flag_USB = False
    
FFT_SIZE = 2048 # initial size
FRAME_RATE = 10 # data refesh rate in Hz
FRAME_TIME = 1./FRAME_RATE

class SubclassManager():
    # base class that gives subclass list and matching
    @classmethod
    def List(cls,level=1): # List of types
        if level==-1:
            # only end-subclasses (might be awkward if empty leaves)
            s = cls.List(1)
            if s==[]:
                # no subclasses level
                return cls.List(0)
            # recurse to lower level
            return sum([c.List(-1) for c in s],[])
        elif level==0:
            # This level
            return [cls]
        elif level==1:
            # All direct subclasses -- default option
            return [ c for c in cls.__subclasses__() ]
        else:
            # Recursive level of subclases
            return sum([s.List(level-1) for s in cls.List(1)],[]) 
            
    @classmethod
    def Match(cls,name,level=1): # List of types
        cl = cls.List(level)
        for c in cl:
            if name == c.__name__:
                return c
        return None

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

class CustomRadio(Radio):
    make = "CustomRadio"
    model = "Specified"
    # Intermediate Frquency
    def __init(self,IF):
        self.IF = IF

class ModelessMenu():
    # Controls a modeless menu -- create, destroy, positioning
    def __init__( self, MenuControl ):
        self.MenuControl = MenuControl # Class of dialog control
        self.modeless = None
        # subclasses must define MenuControl
        
    def __del__(self):
        self.close_modeless()

    def close_modeless( self ):
        if self.modeless:
            if self.modeless.isVisible():
                self.modeless.close()
            self.modeless = None

    def Menu_open( self, parent ):
        if self.modeless:
            self.close_modeless()
        else:
            self.modeless = self.MenuControl(self,parent)
            self.modeless.setModal(False)
            try:
                x_loc = type(self.modeless).default_x_loc
            except NameError:
                x_loc = 0
            self.modeless.move(x_loc,parent.geometry().bottom())
            self.modeless.show()
            self.modeless.raise_()
            self.modeless.activateWindow()

# Subclass pyqtgraph context menus
# Referehce: https://groups.google.com/forum/#!topic/pyqtgraph/3jWiatJPilc
# by Morgan Cherioux

class CustomRadioViewBox(pg.ViewBox):
    signalShowT0 = QtCore.Signal()
    signalShowS0 = QtCore.Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setRectMode() # Set mouse mode to rect for convenient zooming
        self.menu = None # Override pyqtgraph ViewBoxMenu
        self.menu = self.getMenu() # Create the menu

    def raiseContextMenu(self, ev):
        print("Context",ev)
        if not self.menuEnabled():
            return
        menu = self.getMenu()
        pos  = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))

    def getMenu(self):
        if self.menu is None:
            self.menu = QtGui.QMenu()
#            self.viewAll = QtGui.QAction("Vue d\'ensemble", self.menu)
#            self.viewAll.triggered.connect(self.autoRange)
#            self.menu.addAction(self.viewAll)
            self.leftMenu = QtGui.QMenu("Left click")
            group = QtGui.QActionGroup(self)
            pan = QtGui.QAction(u'Move', self.leftMenu)
            zoom = QtGui.QAction(u'Zoom', self.leftMenu)
            self.leftMenu.addAction(pan)
            self.leftMenu.addAction(zoom)
            pan.triggered.connect(self.setPanMode)
            zoom.triggered.connect(self.setRectMode)
            pan.setCheckable(True)
            zoom.setCheckable(True)
            pan.setActionGroup(group)
            zoom.setActionGroup(group)
            self.menu.addMenu(self.leftMenu)
            self.menu.addSeparator()
            self.showT0 = QtGui.QAction(u'Amplitude markers', self.menu)
            self.showT0.triggered.connect(self.emitShowT0)
            self.showT0.setCheckable(True)
            self.showT0.setEnabled(False)
            self.menu.addAction(self.showT0)
            self.showS0 = QtGui.QAction(u'Integration zone', self.menu)
            self.showS0.setCheckable(True)
            self.showS0.triggered.connect(self.emitShowS0)
            self.showS0.setEnabled(False)
            self.menu.addAction(self.showS0)
        return self.menu

    def emitShowT0(self):
        self.signalShowT0.emit()

    def emitShowS0(self):
        self.signalShowS0.emit()

    def setRectMode(self):
        self.setMouseMode(self.RectMode)

    def setPanMode(self):
        self.setMouseMode(self.PanMode)

class PltWidget(pg.PlotWidget):
    """
    Subclass of PlotWidget
    """
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, viewBox=CustomRadioViewBox(), **kwargs )

class PanClass(SubclassManager):
    # Pan adapter device including RTLSDR
    _name = "Panadapter"
    def __init(self):
        print("Pan class")
        self.driver = None
        
    @property
    def name( self ):
        return getattr(self,'_name', type(self)._name )

    @name.setter
    def name( self, name ):
        self._name = name
        
    def Close(self):
        self.driver = None
        
    def __del__(self):
        #print("Closing ",self._name)
        self.Close()
        
    @property
    def Mode( self ):
        # Stream or Block
        return type(self).Mode
        
    def Menu( self, menu, parent, action=False ):
        if action:
            # action true -- just make an active choice (for modeless dialog usually)
            act = QtWidgets.QAction("Pan &Configure",parent)
            menu.addAction(act)
            return act
        else:
            # Make a submenu stub
            return menu.addMenu( "Pan &Configure" )

class PanStreamClass(PanClass):
    # All PanAdapters that stream Data
    _name = "None"
    Mode = 'Stream'
    
    # Stream is a continuous flow of data, with a notification technique (callback)
    # self.driver identifies the device
    # self.stream is the actual stream
    # Needs to be updated when chuck size changed ( usually stopped and restarted )

    def __init__(self):
        PanClass.__init__(self)
#        super(PanStreamClass,self).__init__()
        self.stream = None
        self.chunk_size = 0
        
    def stream_open( self ):
        print("Implementation Error: stream_open routine not defined for Panadapter ",self.name)

    def stream_close( self ):
        print("Implementation Error: stream_close routine not defined for Panadapter ",self.name)

    def Stream( self, callback , chunk_size ) :
        self.callback = callback
        self.chunk_size = chunk_size

        if self.stream:
            # close an existing stream
            self.stream_close()
        self.stream = self.stream_open()
        
class PanBlockClass(PanClass):
    # All PanAdapters that stream Data
    _name = "None"
    Mode = 'Block'

    def __init(self):
        super().__init__()
        
class RTLSDRControl(QtWidgets.QDialog):
    default_x_loc = 0 # startup screen location

    def __init__(self,caller,parent):
        super().__init__(parent)
        self.resize(400,400)
        self.parent = parent
        self.caller = caller
        
        global AppState
        
        form = QtWidgets.QFormLayout()
        self.form = form
        
        m = QtWidgets.QDoubleSpinBox(self.parent,minimum=15e3,maximum=3e9,singleStep=50e3,decimals=0,suffix=' Hz',)
        form.addRow( '&Center Frequency',m)
        m.setValue( AppState.panadapter.driver.fc )
        m.valueChanged.connect(lambda d: AppState.panadapter.driver.set_center_freq(d))
        
        m = QtWidgets.QDoubleSpinBox(self.parent,minimum=1.2e6,maximum=3.2e6,singleStep=50e3,decimals=0,suffix=' Hz',)
        form.addRow( '&Sample Rate',m)
        m.setValue( AppState.panadapter.driver.sample_rate )
        m.valueChanged.connect(lambda d: AppState.panadapter.driver.set_sample_rate(d))
        
        m = QtWidgets.QDoubleSpinBox(self.parent,minimum=0,maximum=3.2e6,singleStep=50e3,decimals=0,suffix=' Hz',)
        form.addRow( '&Bandwidth',m)
        m.setValue( AppState.panadapter.driver.bandwidth )
        m.valueChanged.connect(lambda d: AppState.panadapter.driver.set_bandwidth(d))
        
        m = QtWidgets.QCheckBox('&Automatic Gain Control (agc)',parent)
        AppState.panadapter.driver.set_agc_mode(False) # default off
        m.setCheckState(QtCore.Qt.Unchecked)
        m.stateChanged.connect(self.AGCtoggle)
        form.addRow(m)
        
        m = QtWidgets.QComboBox(self.parent)
        self.wgain = m
        form.addRow( '&Gain (dB)',m)
        gg = AppState.panadapter.driver.gain
        ig = None
        for i,g in enumerate(AppState.panadapter.driver.get_gains()):
            if gg == g:
                ig = i
            m.addItem(str(g/10.))
        if ig:
            self.wgain.setCurrentIndex(ig)
        else:
            self.wgain.setCurrentIndex(0)
        m.setEditable(False)
        m.currentIndexChanged.connect(lambda i: AppState.panadapter.driver.set_gain(int(float(self.wgain.itemText(i))*10)))
        
        m = QtWidgets.QComboBox(self.parent)
        form.addRow( '&Direct Sampling',m)
        ds = {
            'No' : False,
            'Yes - i channel':'i',
            'Yes - q channel':'q'
            }
        for d in ds:
            m.addItem(d)
        m.setEditable(False)            
        m.currentIndexChanged.connect(lambda i, m=m, ds=ds: AppState.panadapter.driver.set_direct_sampling(ds[m.itemText(i)]))
        if AppState.panadapter.driver.fc < 30e6:
            m.setCurrentIndex(m.findText('Yes - q channel'))
        else:
            m.setCurrentIndex(m.findText('No'))
        
        m = QtWidgets.QDoubleSpinBox(self.parent,minimum=0,maximum=1000,singleStep=1,decimals=0,suffix=' Hz',)
        form.addRow( '&Frequency correction',m)
        m.setValue( AppState.panadapter.driver.freq_correction )
        m.valueChanged.connect(lambda d: AppState.panadapter.driver.set_freq_correction(d))
        
        tl = { 0:'Unknown',1:'E4000',2:'FC0012',3:'FC0013',4:'FC2580',5:'R820T',6:'R828D' }
        n = AppState.panadapter.driver.get_tuner_type()
        if n in tl:
            m = QtWidgets.QLabel(tl[n],parent)
        else:
            m = QtWidgets.QLabel('Tuner type '+str(n),parent)
        form.addRow( 'Tuner chip',m)
        
        self.setLayout(form)
        
    def AGCtoggle( self, s ):
        global AppState
        if s == QtCore.Qt.Checked:
            AppState.panadapter.driver.set_manual_gain_enabled(False)
            self.wgain.hide()
            self.form.labelForField(self.wgain).hide()
        else:
            AppState.panadapter.driver.set_manual_gain_enabled(True)
            self.wgain.show()
            self.form.labelForField(self.wgain).show()
            
    def closeEvent( self, event ):
        # Window closed directly
        # Make sure WaveformPan knows about our closing
        self.caller.modeless = None
        event.accept()
        
class RTLSDR(PanBlockClass,ModelessMenu):
    SampleRate = 2.56E6  # Sampling Frequency of the RTLSDR card (in Hz) # DON'T GO TOO LOW, QUALITY ISSUES ARISE
    _name = "RTLSDR"
    def __init__(self,serial=None,index=None,host=None,port=12345):
        self.driver = None
        if not Flag_rtlsdr:
            return
        self.serial = serial
        self.index = index
        self.host = host
        self.port = port
        self.SampleRate = type(self).SampleRate

        try:
            if self.serial:
                self.driver = rtlsdr.RtlSdr(serial_number = self.serial)
                self._name = f'RTLSDR serial {serial}'
            elif self.index:
                self.driver = rtlsdr.RtlSdr(self.index)
                self._name = f'RTLSDR index {index}'
            elif self.host:
                self.driver = rtlsdr.RtlSdrTcpClient( hostname=self.host, port=self.port ) 
                self._name = f'RTLSDR @{host}:{port}'
            else:
                self.driver = rtlsdr.RtlSdr()
                self._name = 'RTLSDR'
        except:
            print("RTLSDR not found")
            self.driver = None
            return
        self.driver.sample_rate = self.SampleRate

        # Circumvent python heirachy madness
        ModelessMenu.__init__( self, RTLSDRControl )
        
    def __del__(self):
        if self.driver:
            self.driver.close()
        
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

    def Menu( self, menu, parent ):
        this_menu = super().Menu(menu, parent, True )
        
        this_menu.triggered.connect(lambda state, p=parent: self.Menu_open(p) )
        
#        this_menu.addAction(waveform)
        return this_menu
        
class SoapyPan(PanBlockClass):

    SampleRate = 2.56E6  # Sampling Frequency of the RTLSDR card (in Hz) # DON'T GO TOO LOW, QUALITY ISSUES ARISE
    _name = "SoapySDR"

    def __init__(self, dictionary ):
        #print(type(self),dictionary)
        self.driver = None
        if not Flag_soapy:
            print("No SoapySDR support")
            return

        required_keys = ['driver','name']
        # Check for essential arguments
        for k in required_keys:
            if k not in dictionary:
                print('SoapySDR missing {} entry'.format(k))
                return

        self.SampleRate = type(self).SampleRate
        self._size = 0 # size of buffer -- will expand as needed but allows a persistent buffer

        #print("About to try ",dictionary)
        try:
            self.driver = SoapySDR.Device(dictionary)
            if not self.driver:
                print("SoapySDR cannot load ",dictionary['driver'],' device ',dictionary['name'])
        except:
            print("SoapySDR driver not found for ",dictionary['driver'],' device ',dictionary['name'])
            self.driver = None

        # Check if ok till here
        if not self.driver:
            return
        
        #query device info
        print("SoapySDR driver loaded",dictionary['driver'],' device ',dictionary['name'])
        print(self.driver.listAntennas('ant ',SoapySDR.SOAPY_SDR_RX, 0))
        print(self.driver.listGains('gain ',SoapySDR.SOAPY_SDR_RX, 0))
        freqs = self.driver.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0)
        for freqRange in freqs:
            print('freq ',freqRange)
        
        #setup a stream (complex floats)
        self.rxStream = self.driver.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self.driver.activateStream(self.rxStream) #start streaming

        
    def __del__(self):
        if self.driver:
            self.driver.close()
        
    def SetFrequency(self, IF ):
        self.driver.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, IF)
        
    def Read(self,size):
        if size != self_size:
            self._size = size
            self.buf = numpy.array([0]*size, numpy.complex64) 
        sr = self.driver.readStream(self.rxStream, [self.buf], size)
        return self.buf
        
    def Close(self):
        if self.driver:
            self.driver.close()
        self.driver = None

class SoapyRemotePan(SoapyPan):
    
    SampleRate = 2.56E6  # Sampling Frequency of the RTLSDR card (in Hz) # DON'T GO TOO LOW, QUALITY ISSUES ARISE
    _name = "SoapyRemote"
    
    def __init__(self, dictionary ):
        self.driver = None
        #print(type(self),dictionary)

        if not Flag_soapy:
            print("No SoapySDR")
            return

        # Check for essential arguments
        required_keys = ['address','port','name']
        for k in required_keys:
            if k not in dictionary:
                print('SoapySDR missing {} entry'.format(k))
                return

        args = {}
        args['driver'] = 'remote'
        if ':' in dictionary['address']:
            # IPV6
            args['remote'] = "tcp://" + "[" + dictionary['address'] + "]:" + str(dictionary['port'])
        else:
            # IPV4
            args['remote'] = "tcp://" + dictionary['address'] + ":" + str(dictionary['port'])

        for k in dictionary:
            if k not in required_keys:
                args[k] = dictionary[k]
        
        # Add Name back in
        args['name'] = dictionary['name']

        super().__init__(args)

class AudioPan(PanStreamClass):
    SampleRate = 44100.0
    _name = "Audio"
    def __init__(self,index):
        global AppState
        super().__init__()

        if not Flag_audio:
            return
        if AppState.audio:
            try:
                info = AppState.audio.get_device_info_by_index( index)
                self._name = info['name']
                self.SampleRate = info['defaultSampleRate']
                self.driver = index
            except:
                print("Could not open audio device")
    
    def stream_open( self ):
        global AppState
        if not self.driver:
            return None
        try:
            stream = AppState.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                frames_per_buffer=self.chunk_size,
                rate=int(self.SampleRate),
                input=True,
                output=False,
                stream_callback = self.audio_callback,
                input_device_index = self.driver
                )
            return stream
        except:
            return None

    def stream_close( self ):
        if self.stream:
            self.stream.close()
            self.stream = None

    def __del__(self):
        if self.driver:
            self.driver.close()
        
    def SetFrequency(self, IF ):
        self.center_freq = IF
        
    def audio_callback( self, in_data, frame_count, time_info, status_flags ):
        self.callback( np.frombuffer( in_data, 'float32' ) ) 
        return (None, pyaudio.paContinue)

    def Close(self):
        if self.driver:
            #shutdown the stream
            self.driver.deactivateStream(self.rxStream) #stop streaming
            self.driver.closeStream(self.rxStream)
        self.driver = None

class Waveform(SubclassManager):
    def __init__(self):
        self._size = 0
        self._phase = 0.
        self.new_phase = self._phase
        self._cycles = 1.
        self.new_cycles = self._cycles
        self._skew = .5
        self.new_skew = self._skew
        
    @property
    def name(self):
        return type(self).__name__
        
    @property
    def phase(self):
        return self._phase
        
    @phase.setter
    def phase( self, p ):
        self.new_phase = p

    @property
    def cycles(self):
        return self._cycles
        
    @cycles.setter
    def cycles( self, c ):
        if c == 0.:
            c = 1.
        self.new_cycles = c
        
    @property
    def skew(self):
        return self._skew
        
    @skew.setter
    def skew( self, s ):
        self.new_skew = s
        
    def makeArray(self):
         self.data = (np.zeros(self._size)+.00000000001) + (np.zeros(self._size)+.0000000001)*1j
       
    def create( self ):
        # default
        pass
            
    def Read( self, size ):
        #print("Size",self._size)
        if (self._size != size) or (self._cycles != self.new_cycles) or (self._phase != self.new_phase)or (self._skew != self.new_skew):
            
            self._size = size
            self._phase = self.new_phase
            self._cycles = self.new_cycles
            self._skew = self.new_skew
            
            # tests
            if self._cycles > self._size:
                self._cycles = self._size
            if self._cycles <= 0.:
                self._cycles = 1.
            self._phase %= 360.
            if self._phase < 0.:
                self._phase += 360.
            if self._skew < 0.:
                self._skew = 0.
            if self._skew > 1.:
                self._skew = 1.
                
            self.new_phase = self._phase
            self.new_cycles = self._cycles
            self.new_skew = self._skew
            #print("Create cy ph sk",size,self._cycles,self._phase,self.skew)
                        
            self.makeArray() 
            self.create()

            #phase correction
            cycle_length = int(self._size / self._cycles)
            self.data = np.roll(self.data,int(cycle_length*self._phase/360.))

        return self.data
        
class Impulse(Waveform):
    _name = "Impulse"
    def __init__(self):
        super().__init__()
        
    def create(self):
        cycle_length = int(self._size / self._cycles)
        hcl = cycle_length // 2
        for i in range( hcl,self._size,cycle_length ):
            self.data[i] += 1
            if i > 0:
                self.data[i-1] += self._skew
            if i < self._size-1:
                self.data[i+1] += self._skew
                            
class Square(Waveform):
    _name = "Square wave"
    def __init__(self):
        super().__init__()
        
    def create(self):
        a = np.linspace(0,self._cycles,self._size)
        self.data[np.mod(a,1)<self._skew] = 1
                            
class J0Bessel(Waveform):
    _name = "J0 Bessel"
    def __init__(self):
        super().__init__()
        
    def create(self):
        s = self._skew
        if s < .01:
            s = .01
        L = 10./s
        a = np.mod(np.linspace(0,self._cycles*L,self._size),L)
        self.data = scipy.special.jv(0,a)
                            
class J1Bessel(Waveform):
    _name = "J1 Bessel"
    def __init__(self):
        super().__init__()
        
    def create(self):
        s = self._skew
        if s < .01:
            s = .01
        L = 10./s
        a = np.mod(np.linspace(0,self._cycles*L,self._size),L)
        self.data = scipy.special.jv(1,a)
                            
class Logit(Waveform):
    _name = "Logit (x^p*(1-x^q)"
    def __init__(self):
        super().__init__()
        
    def create(self):
        p = self._skew
        if p < .01:
            p = .01
        if p > .99:
            p = .99
        q = 1 - p
        mx = p*math.pow(q,q/p)
        a = np.mod(np.linspace(0,self._cycles,self._size),1)
        self.data = a**q * ( 1 - a**p ) / mx
                            
class Sawtooth(Waveform):
    _name = "Sawtooth"
    def __init__(self):
        super().__init__()
        
    def create(self):
        s = self._skew
        if s < .001:
            s = .001
        elif s > .999:
            s = .999
        p = math.log(.5) / math.log(s)

        a = np.linspace(0,self._cycles,self._size)
        self.data = np.power(np.mod(a,1),p)

class Triangle(Waveform):
    _name = "Triangles"
    def __init__(self):
        super().__init__()
        
    def create(self):
        cycle_length = int(self._size / self._cycles)

        if self._skew == 0.:
            for i in range(self._size):
                self.data[i] += 1 - ( i % cycle_length ) / cycle_length
        elif self._skew == 1.:
            for i in range(self._size):
                self.data[i] += ( i % cycle_length ) / cycle_length
        else:
            skew_length = cycle_length * self._skew
            for i in range(self._size):
                x = i % cycle_length
                if x < skew_length:
                    self.data[i] += x / skew_length
                else:
                    self.data[i] += (cycle_length - x) / (cycle_length - skew_length)

class Bits(Waveform):
    _name = "Bits"
    def __init__(self):
        super().__init__()
        
    def create(self):
        mx = 0
        cycle_length = int(self._size / self._cycles)

        for i in range(0,self._size,cycle_length):
            for j in range(i,i+cycle_length):
                if j >= self._size:
                    break
                d = 0
                e = j
                while e > 0 :
                    d += e & 1
                    e >>= 1
                mx = max(mx,d)
                self.data[j] = d

        # normalize
        self.data /= mx

class Totient(Waveform):
    _name = "Euler's Totient"
    def __init__(self):
        super().__init__()
        
    def create(self):

        cycle_length = int( self._size / self._cycles )

        # Python 3 program to calculate Euler's Totient Function using Euler's product formula 
        # This code is based on an example by Nikita Tiwari. 

        for i in range(self._size):
            n = ( i % cycle_length ) + 1 # Offset to avoid 0
            result = n # Initialize result as n 
            
            # Consider all prime factors of n and for every prime factor p, multiply result with (1 - 1 / p) 
            p = 2
            while p * p <= n : 
                # Check if p is a prime factor. 
                if n % p == 0 : 
                    # If yes, then update n and result 
                    while n % p == 0 : 
                        n //= p 
                    result *= (1.0 - (1.0 / p)) 
                p += 1
                
                
            # If n has a prime factor greater than sqrt(n) 
            # (There can be at-most one such prime factor) 
            if n > 1 : 
                result *= (1.0 - (1.0 / n)) 

            self.data[i] = result / cycle_length
    
class Distance(Waveform):
    _name = "Distance to Power"
    def __init__(self):
        super().__init__()
        
    def create(self):
        cycle_length = int( self._size / self._cycles )
        
        a = np.linspace(1,self._size+1,self._size+1)
        p = a ** (1 + self._skew)
        a = np.mod(a[:self._size],cycle_length)

        self.data = p[np.searchsorted(p,a)]-a

        mx = np.max(self.data)
        self.data /= mx
    
class Sine(Waveform):
    _name = "Sine wave"
    def __init__(self):
        super().__init__()
        
    def create(self):
        a = np.linspace(0,2*np.pi*self._cycles,self._size)
        self.data = np.sin(a) + np.cos(a)*1j

class Random(Waveform):
    _name = "Random"
    def __init__(self):
        super().__init__()
        
    def Read(self,size):
        return 2*(np.random.random(size)+np.random.random(size)*1j)-(1.+1.j)

class WaveformParameter():
    def __init__( self, name, getter, setter, lower, upper, to_spinbox, to_slider ):
        self._name = name
        self._getter = getter
        self._setter = setter
        self._lower = lower
        self._upper = upper
        self._value = 0
        self.to_spinbox = to_spinbox
        self.to_slider = to_slider
        self.left = None
        self.right = None
        
    @property
    def name( self ):
        return self._name
        
    def getValue( self ):
        return self._getter()
        
    def setValue( self, value ):
        self._setter( value )

    @property
    def value( self ):
        return self._value

    @value.setter
    def value( self, v ):
        self._value = v

    def leftChange( self ):
        i = self.left.value()
        f = self.to_spinbox(i)
        self.right.setValue(f)
        self._setter(f)
        self.signal.emit()
        
    def rightChange( self, f ):
        f = self.right.value()
        i = self.to_slider(f)
        self.left.setValue(i)
        self._setter(f)
        self.signal.emit()

    def set_pair( self, parent, signal ):
        self.signal = signal

        self.left = QtWidgets.QSlider(QtCore.Qt.Horizontal,parent)
        self.left.setRange( self.to_slider(self._lower), self.to_slider(self._upper) )
        self.left.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.left.setTickInterval(10)
        self.left.valueChanged.connect(self.leftChange)

        self.right = QtWidgets.QDoubleSpinBox(parent)
        #self.right.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.right.setRange( self._lower, self._upper )
        self.right.setDecimals(2)
        self.right.valueChanged.connect(self.rightChange)

        self.update()
        
        return (self.left,self.right)
        
    def update( self ):
        value = self._getter()
        self.left.setValue(self.to_slider(value))
        self.right.setValue(value)

class WaveformControl(QtWidgets.QDialog):
    default_x_loc = 0 # startup screen location
    math_change_signal = QtCore.pyqtSignal()
    def __init__(self,caller,parent):
        super().__init__(parent)
        self.resize(400,400)
        self.parent = parent
        self.caller = caller
        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(1,1) # slider
        grid.setRowStretch(0,1) # graph
        grid.setRowStretch(1,1) # choose waveform
        self.setLayout(grid)
        
        self.phase = WaveformParameter( 'phase', 
            caller.get_phase, caller.set_phase,
            0, 359,
            (lambda i: float(i) ),
            (lambda f: int(f) ) 
            )
        self.cycles = WaveformParameter( 'cycles', 
            caller.get_cycles, caller.set_cycles,
            .01, 1000, 
            (lambda i: math.pow( 10,i/100. ) ),
            (lambda f: int(math.log10(f)*100) ) 
            )
        self.skew = WaveformParameter( 'skew', 
            caller.get_skew, caller.set_skew,
            0., 1.,
            (lambda i: float(i/100.) ),
            (lambda f: int(f*100) ) 
            )
        
        self.math_change_signal.connect(self.ShowCurve)

        row = 0
        plot = pg.PlotWidget()
        plot.setBackground('w') 
        plot.setXRange(0,360,padding = 0) 
        plot.setYRange(-1,1,padding = .05)
        plot.hideAxis('bottom')
        data =  caller.waveform.Read(360)
        self.real = plot.plot(np.real(data),pen=pg.mkPen(color='k',width=2))
        self.imag = plot.plot(np.imag(data),pen=pg.mkPen(color='r',width=2))
        grid.addWidget(plot,row,0,1,3)
        
        row += 1
        grid.addWidget(QtWidgets.QLabel('Waveform',parent),row,0)
        combo = QtWidgets.QComboBox(parent)
        for w in WaveformList():
            m = combo.addItem(w)
        combo.setCurrentIndex(combo.findText(caller.waveform.name))
        combo.setEditable(False)
        combo.currentIndexChanged.connect(lambda w:self.changeWaveform(combo.itemText(w)))
        grid.addWidget(combo,row,1,1,2)
        
            
        for wp in [ self.phase, self.skew, self.cycles ]:
            row += 1
            name = wp.name
            grid.addWidget(QtWidgets.QLabel(name[:1].upper() + name[1:].lower(),parent),row,0)
            (left,right) = wp.set_pair(parent,self.math_change_signal)
            grid.addWidget(left,row,1)
            grid.addWidget(right,row,2)
                
    def changeWaveform( self, w ):
        self.caller.waveform_select(w)
        for wp in [ self.phase, self.skew, self.cycles ]:
            wp.update()
        self.ShowCurve()
    
    def closeEvent( self, event ):
        # Window closed directly
        # Make sure WaveformPan knows about our closing
        self.caller.modeless = None
        event.accept()
        
    def ShowCurve(self):
        data =  self.caller.waveform.Read(360)
        self.real.setData(np.real(data))
        self.imag.setData(np.imag(data))

def WaveformList():
    # generates a list of waveform types
    modules = [l.__name__ for l in Waveform.List()]
    if not Flag_random:
        modules.remove('Random')
    return modules

class WaveformPan(PanBlockClass,ModelessMenu):
    SampleRate = 2.56E6 
    _name = "Waveform"

    def __init__(self):
        self.driver = True
        self.squarewave = False
        self.waveform_select( sorted(WaveformList())[0] )

        # Circumvent python heirachy madness
        ModelessMenu.__init__( self, WaveformControl )

    def SetFrequency(self, IF ):
        pass
        
    def Read(self,size):
        return self.waveform.Read( int(size) )
        
    def Close(self):
        self.driver = None
    
    def waveform_select( self, w ):
        self.module_chosen = w
        self.waveform = Waveform.Match( w )()
        self.name = self.waveform.name
        self.waveform.phase = self.phase 
        self.waveform.cycles = self.cycles 
        
    def get_skew( self ):
        return self.waveform.skew
    def set_skew( self, x ):
        self.waveform.skew = x
    skew = property( get_skew, set_skew )
        
    def get_cycles( self ):
        return self.waveform.cycles
    def set_cycles( self, x ):
        self.waveform.cycles = x
    cycles = property( get_cycles, set_cycles )
        
    def get_phase( self ):
        return self.waveform.phase
    def set_phase( self, x ):
        self.waveform.phase = x
    phase = property( get_phase, set_phase )
        
    def Menu( self, menu, parent ):
        this_menu = super().Menu(menu, parent, True )
        
        this_menu.triggered.connect(lambda state, p=parent: self.Menu_open(p) )
        
#        this_menu.addAction(waveform)
        return this_menu

class   FFTTaperingMenu  (ModelessMenu):
    def __init__(self):
        super().__init__(FFTTaperingControl)

class   FFTTaperingControl (QtWidgets.QDialog):
    default_x_loc = 800 # startup screeen location
    taper_list = {
            'barthann'  :   [],
            'bartlett'  :   [],
            'blackmanharris':[],
            'blackman'  :   [],
            'bohman'    :   [],
            'boxcar'    :   [],
            'flattop'   :   [],
            'hamming'   :   [],
            'hann'      :   [],
            'parzen'    :   [],
            'nuttall'   :   [],
            'triang'    :   [],
            'kaiser'    :   [('beta',14)],
            'gaussian'  :   [('stndrd dev',7)],
            'general gaussian':[('power',1.5),('stndrd dev',7)],
            'slepian'   :   [('bandwidth',.3)],
            'dpss'      :   [('nrml 1/2 bandwdth',3)],
            'chebwin'   :   [('atten db',100)],
            'exponential':  [('decay scale',3)],
            'tukey'     :   [('taper frac',.3)]
            }
    taper_size = 51 # points in the window function
    fft_size = 2048

    def __init__(self,caller,parent):
        global AppState
        super().__init__(parent)
        self.resize(400,500)
        self.parent = parent
        self.caller = caller
        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(1,1) # slider
        grid.setRowStretch(0,1) # graph
        grid.setRowStretch(1,2) # choose waveform
        self.setLayout(grid)
        
        row = 0
        # Taper graph
        self.plot0 = PltWidget() # for custom context window
        self.plot0.setBackground('w') 
        self.plot0.setXRange(0,type(self).taper_size-1,padding = 0) 
        self.plot0.setYRange(0,1,padding = .05)
        self.plot0.hideAxis('bottom')
        self.taperplot = None
        grid.addWidget(self.plot0,row,0,1,2)
        
        row += 1
        # FFT graph
        self.plot1 = PltWidget() # for custom context window
        self.plot1.setBackground('w') 
        self.plot1.setXRange(0,1027,padding = 0) 
        self.plot1.setYRange(-140,0,padding = .05)
        self.plot1.hideAxis('bottom')
        self.plot1.showGrid(x=False,y=True,alpha=.5)
        self.fftplot = None
        grid.addWidget(self.plot1,row,0,1,2)
        
        row += 1
        # Taper choose
        grid.addWidget(QtWidgets.QLabel('Taper function',parent),row,0)
        self.combo = QtWidgets.QComboBox(parent)
        for w in sorted(type(self).taper_list.keys()):
            m = self.combo.addItem(w)
        self.combo.currentIndexChanged.connect(self.NewTaper)
        self.combo.setEditable(False)
        grid.addWidget(self.combo,row,1)
        
        row += 1
        # Optional parameter 0
        self.P0text = QtWidgets.QLabel('param1',parent)
        self.P0val = QtWidgets.QDoubleSpinBox(parent)
#        self.P0val.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        grid.addWidget(self.P0text,row,0)
        grid.addWidget(self.P0val,row,1)
        self.P0val.valueChanged.connect(self.ShowCurve)
        
        row += 1
        # Optional parameter 1
        self.P1text = QtWidgets.QLabel('param2',parent)
        self.P1val = QtWidgets.QDoubleSpinBox(parent)
#        self.P1val.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        grid.addWidget(self.P1text,row,0)
        grid.addWidget(self.P1val,row,1)
        self.P1val.valueChanged.connect(self.ShowCurve)
        
        # set current taper
        current_taper = AppState.fft_tapering
        if isinstance( current_taper, tuple ):
            current_taper = current_taper[0]
        indx = self.combo.findText(current_taper)
        if indx < 0:
            indx = 0
        self.combo.setCurrentIndex(indx)
    
    def closeEvent( self, event ):
        # Window closed directly
        self.caller.modeless = None
        event.accept()
        
    def NewTaper( self, i ):
        self.taper = self.combo.itemText(i)
        p = type(self).taper_list[self.taper]
        if len(p) == 2:
            # 2 parameters
            self.P1text.setText(p[1][0])
            self.P1val.setValue(p[1][1])
            self.P1text.setVisible(True)
            self.P1val.setVisible(True)
            self.P0text.setText(p[0][0])
            self.P0val.setValue(p[0][1])
            self.P0text.setVisible(True)
            self.P0val.setVisible(True)
        elif len(p)==1:
            self.P1text.setVisible(False)
            self.P1val.setVisible(False)
            self.P0text.setText(p[0][0])
            self.P0val.setValue(p[0][1])
            self.P0text.setVisible(True)
            self.P0val.setVisible(True)
        else:
            self.P1text.setVisible(False)
            self.P1val.setVisible(False)
            self.P0text.setVisible(False)
            self.P0val.setVisible(False)
        self.ShowCurve()

    def ShowCurve(self):
        global AppState
        
        # find taper and possible parameters
        # and set globally in AppState 
        p = type(self).taper_list[self.taper]
        if len(p) == 2:
            p1 = self.P1val.value()
            p0 = self.P0val.value()
            AppState.fft_tapering = (self.taper,p0,p1)
        elif len(p) == 1:
            p0 = self.P0val.value()
            AppState.fft_tapering = (self.taper,p0)
        else:
            AppState.fft_tapering = self.taper
            
        # get taper shape and plot
        taperdata = scipy.signal.get_window(AppState.fft_tapering,51)
        if self.taperplot:
            self.taperplot.setData(taperdata)
        else:
            self.taperplot = self.plot0.plot(taperdata,pen=pg.mkPen(color='k',width=2))
            
        # get fft and plot
        fft = np.fft.fft(taperdata, 2048) / (len(taperdata)/2.0)
#        taperfft = 20 * np.log10(np.abs(np.fft.fftshift(fft / np.max(np.abs(fft)))))
        taperfft = 20 * np.log10(np.abs(fft / np.max(np.abs(fft))))
        if self.fftplot:
            self.fftplot.setData(taperfft)
        else:
            self.fftplot = self.plot1.plot(taperfft,pen=pg.mkPen(color='k',width=2))
                
class   FFTSizeMenu  (ModelessMenu):
    def __init__(self):
        super().__init__(FFTSizeControl)

class   FFTSizeControl (QtWidgets.QDialog):
    default_x_loc = 400 # startup screeen location

    def __init__(self,caller,parent):
        global AppState
        super().__init__(parent)
        self.resize(400,200)
        self.parent = parent
        self.caller = caller

        form = QtWidgets.QFormLayout()
        
        self.fft_size = self.combo( [2**i for i in range(5,16)], AppState.fft_size )
        self.fft_size.currentIndexChanged.connect(self.set_fft_size)
        form.addRow('FFT &Size',self.fft_size)
        
        self.fft_avg = self.combo( [2**i for i in range(0,9)], AppState.fft_avg )
        self.fft_avg.currentIndexChanged.connect(self.set_fft_avg)
        form.addRow('FFT &Averaged',self.fft_avg)
        
        self.setLayout(form)
    
    def set_value( self, box, value ):
        # Find a value in the sorted value list
        lst = [int(box.itemText(i)) for i in range(box.count()) ]
        i = np.searchsorted( lst, value )
        if i == box.count():
            i -= 1
        box.setCurrentIndex(i)
        
    def set_fft_size(self, i ):
        global AppState
        old_size = AppState.fft_size
        AppState.fft_size = int(self.fft_size.currentText())
        # Make avg compensate
        self.set_value( self.fft_avg, AppState.fft_avg * old_size / AppState.fft_size )
        self.parent.fft_change_signal.emit()
    
    def set_fft_avg(self, i ):
        global AppState
        AppState.fft_avg = int(self.fft_avg.currentText())
        self.parent.fft_change_signal.emit()
    
    def combo(self, valuelist, default ):
        c = QtWidgets.QComboBox(self.parent)
        c.setEditable(False)
        c.addItems([str(x) for x in valuelist])
        self.set_value( c , default )
        return c

    def closeEvent( self, event ):
        # Window closed directly
        self.caller.modeless = None
        event.accept()
        
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

class Data():
    # holds the data from the device in buffer that rolls off the end
    def __init__(self,chunk_size=8196*2):
        self.lock = QtCore.QMutex() # For adding or pulling data from panadapter
        self.lock.unlock()
        self.chunk_size = chunk_size
        self.max_size = self.chunk_size * 16
        self.target_size = self.max_size * .9

        # controller for delay time
        self.delay_time = 0.
        self.NR = newtrap.NewtRap( self.target_size, .01*FRAME_TIME, 0, FRAME_TIME, self.delay_time )
        
    def new_real( self ):
        self.lock.lock()
        self.data = np.zeros(self.max_size)
        self.real = True
        self.new_common()
    
    def new_complex( self ):
        self.lock.lock()
        self.data = np.zeros(self.max_size)*(1+1j)
        self.real = False
        self.new_common()
    
    def new_common( self ):
        self.size = 0 # posisition of next entry in buffer
        self.real_size = 0 # max buffer size (buffer overwritten from start when some_data)
        self.total_size = 0 # Total bytes read -- inefficient if real_size < total_size
        self.delay_time = .01
        self.lock.unlock()
        return self
    
    def add( self, chunk ):
        length = len(chunk)
        self.lock.lock()
        
        # add length to buffer (fold back on overflow)
        new_size = self.size + length
        if new_size > self.max_size:
            # overwrite from start of buffer
            self.size = 0
            new_size = length
        
        # Check target_size
        self.target_size = np.clip( self.target_size, 8192, self.max_size )
        
        self.data[self.size:new_size] = chunk

        self.size = new_size
        self.real_size = max( self.real_size,self.size )
        self.total_size += length

        self.lock.unlock()

        # pause to not overfill the buffer
        # abs added for safety when seatching gives a negative
        time.sleep(abs(self.delay_time))
    
    def get_data_start(self):
        self.lock.lock()

    def get_data_end(self):
        self.delay_time = self.NR.next(self.total_size)
        self.size = 0
        self.real_size = 0
        self.total_size = 0
        self.lock.unlock()
        print(self.delay_time)
        
    @property
    def target( self ):
        return self.target_size
        
    @target.setter
    def target( self, t ):
        global AppState
        if t >= AppState.fft_size and t <= self.max_size: 
            self.target_size = t
            self.NR.target = t
            
    @property
    def maxsize( self ):
        return self.max_size

class PSD(QtCore.QRunnable):
    # Computes PSD
    def __init__(self,dataclass):
        super().__init__()
        global AppState
        self.psd = np.zeros(AppState.fft_size) # default blank
        self.dataclass = dataclass
        self.lock = QtCore.QMutex() # For adding or pulling data from panadapter
        self.lock.unlock()
        self.loop = True # can change from afar to stop loop
        
        self.NR = newtrap.NewtRap( 0, .01 * FRAME_TIME, AppState.fft_size, self.dataclass.maxsize , self.dataclass.target ) 
    
    def run(self):
        while self.loop:
            target = time.monotonic() + .95 * FRAME_TIME # refresh 10/sec 
            self.update()
            end = time.monotonic()
            if abs(end-target) > .05 * FRAME_TIME:
                self.dataclass.target = int(self.NR.next( target - end ))
                print("Update",self.dataclass.target)
            if end < target:
                # could take more
                time.sleep(target-end)
            else:
                # overscheduled
                pass
            
    def update(self):
        global AppState

        self.dataclass.get_data_start()
        size = self.dataclass.real_size
        chunk = self.dataclass.data[:size]
        
        self.dataclass.get_data_end()
        
        if size < AppState.fft_size:
            return
        
        if AppState.fft_ratio>1:
            f_demod = 1.
            t_total = (1/AppState.panadapter.SampleRate) * size
            t = np.arange(0, t_total, 1 / AppState.panadapter.SampleRate)
            lo = 2**.5 * np.exp(-2j*np.pi*f_demod * t) # local oscillator
            x_mix = chunk*lo
            
            power2 = int(np.log2(AppState.fft_ratio))
            for mult in range(power2):
                x_mix = scipy.signal.decimate(x_mix, 2) # mix and decimate

            sample_freq, spec = scipy.signal.welch(x_mix, AppState.panadapter.SampleRate, window=AppState.fft_tapering, nperseg=AppState.fft_size,  nfft=AppState.fft_size) 
        else:
            sample_freq, spec = scipy.signal.welch(chunk, AppState.panadapter.SampleRate, window=AppState.fft_tapering, nperseg=AppState.fft_size,  nfft=AppState.fft_size)

        # sample freq not used
#        spec = np.roll(spec, AppState.fft_size//2, 0)[FFT_SIZE//2-self.N_WIN//2:AppState.fft_size//2+self.N_WIN//2]
        N_WIN2 = int( .5 * AppState.fft_size / AppState.fft_ratio )
        spec = np.fft.fftshift(spec)[ AppState.fft_size//2-N_WIN2 : AppState.fft_size//2+N_WIN2 ]

        self.lock.lock()
        # get magnitude 
        # convert to dB scale
        self.psd = 20 * np.log10(abs(spec))
        self.lock.unlock()

class ProgramState:
    # holds program "state"
    def __init__( self, panadapter=None, radio_class=None ):
        self._panadapter = panadapter
        self._radio_class = radio_class
        self._resetNeeded = False
        self._Loop = True # for initial entry into loop
        self._soapylist = {}
        self.fft_tapering='hamming'

        self.fft_size = FFT_SIZE
        self.fft_avg = 128
        self.scroll = 1 # waterfall direction
        self.fft_ratio = 2

        self.discover = None
        if Flag_audio:
            sys.stderr.write("\n--------------------------------------------------------\n\tSetup output from PyAudio follows -- usually can be ignored\n")
            self.audio = pyaudio.PyAudio()
            sys.stderr.write("\tEnd of PyAudio setup\n--------------------------------------------------------\n\n")
        else:
            self.audio = None
            
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
        if not panadapter.driver:
            panadapter = WaveformPan()
        if panadapter != self._panadapter:
            self._resetNeeded = True
        self._panadapter = panadapter
        
        # Set Averaging from Sample rate
        self.fft_avg = int( panadapter.SampleRate / self.fft_size / FRAME_RATE )
        
    def SoapyAdd( self, address, port, name ):
        #print("add",address,port,name)
        self._soapylist[(address,port)] = name
        
    def SoapyDel( self, name ):
        #print("soapydel",name)
        self._soapylist = {k:v for k,v in self._soapylist.items() if v != name }
        
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
AppState = ProgramState()

class Waterfall(pg.ImageItem):    
    Colors = {
    'Default' : ([0,.4,1.],[[0,0,90,255], [200,2020,0,255], [255,0,0,255]]) ,
    'Matrix' : ([0., 1.],[[0,0,0,255], [0,255,0,255]]) ,
    'Red Green' : ([0., 0.5, 1.],[[0,0,0,255], [0,255,0,255], [255,0,0,255]]) ,
    'Tropical' : ([0.,.2,.4,.6,.8,1.],[[68,40,153,255],[222,68,252,255],[252,38,99,255],[252,181,38,255],[86,235,49,255],[3,71,7,255]]) ,
    }

    def __init__(self):
        global AppState
        super().__init__()
        
        self.fftwidth = 0
        
        self.minlev = -220
        self.maxlev = -120

        # set colormap
        self.lookuptable('Default')
        self.setLevels([self.minlev, self.maxlev])

        # setup the correct scaling for x-axis
#        self.setLabel('bottom', 'Frequency', units='kHz')
        
#        self.text_leftlim = pg.TextItem("-%.1f kHz"%(bw_hz*self.N_WIN/2.))
#        self.text_leftlim.setParentItem(self.waterfall)
#        self.plotwidget1.addItem(self.text_leftlim)
#        self.text_leftlim.setPos(0, 0)

#        self.text_rightlim = pg.TextItem("+%.1f kHz"%(bw_hz*self.N_WIN/2.))
#        self.text_rightlim.setParentItem(self.waterfall)
#        self.plotwidget1.addItem(self.text_rightlim)
#        self.text_rightlim.setPos(bw_hz*(self.N_WIN-64), 0)

    @QtCore.pyqtSlot(str)
    def lookuptable( self, choice ):
        if choice not in type(self).Colors:
            choice = 'Default'

        (p,c) = type(self).Colors[choice]
        pos = np.array(p)
        color = np.array(c, dtype=np.ubyte)

        cmap = pg.ColorMap(pos, color)
        self.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))

    def init_image(self):
        global AppState

        bw_hz = AppState.panadapter.SampleRate/AppState.fft_size * self.fftwidth/1.e6/AppState.fft_ratio
        self.scale(bw_hz,1)

        self.img_array = -500*np.ones((self.fftwidth//4, self.fftwidth))
        # Plot the grid
        for x in [0, self.fftwidth//2, self.fftwidth-1]:
            if x==0 or x==self.fftwidth-1:
                self.img_array[:,x] = 0
        

    def image_update(self, psd):
        global AppState
        fftwidth = np.size(psd)
        
        if fftwidth != self.fftwidth :
            self.fftwidth = fftwidth
            self.init_image()
        
        # grid
        for x in [0, fftwidth//2, fftwidth-1]:
            psd[x] = 0            

        # roll down one and replace leading edge with new data
        self.img_array[-1:] = psd
        self.img_array = np.roll(self.img_array, -AppState.scroll, 0)


        for i, x in enumerate(range(0, fftwidth-1, (fftwidth//10))):
            if i!=5 and i!=10:
                if AppState.scroll>0:
                    for y in range(5,15):
                        self.img_array[y,x] = 0
                else:
                    for y in range(-10,-2):
                        self.img_array[y,x] = 0

        self.setImage(self.img_array.T, autoLevels=False, opacity = 1.0, autoDownsample=True)

    @QtCore.pyqtSlot()
    def autolevel(self):
        #tmp_array = np.copy(self.img_array[self.img_array>0])
        #tmp_array = tmp_array[tmp_array<250]
        #tmp_array = tmp_array[:]

        #minminlev = np.percentile(tmp_array, 1)
        #self.minlev = np.percentile(tmp_array, 20)
        #self.maxlev = np.percentile(tmp_array, 99.3)
        #print( self.minlev, self.maxlev )
        [ self.minlevel, self.maxlevel ] = np.percentile( self.img_array[self.img_array<0],[2,98])
        self.setLevels([self.minlev, self.maxlev])

        return self.minlev, self.maxlev
        
    @QtCore.pyqtSlot(float, float)
    def newlevel(self, low, high):
        self.setLevels([low, high])

        return low, high
        
class ApplicationDisplay(QtWidgets.QMainWindow):
    # Display class
    # define a custom signal
    soapy_list_signal = QtCore.pyqtSignal()
    soapy_remote_pan_signal = QtCore.pyqtSignal(dict)
    soapy_pan_signal = QtCore.pyqtSignal(dict)
    rtl_pan_signal = QtCore.pyqtSignal(int)
    fft_change_signal = QtCore.pyqtSignal() # change in one of the FFT size parameters

    refresh = int(FRAME_TIME*1000) # default refresh timer in msec

    def __init__(self ):
        # Comes in with Panadapter set and radio_class set.
        global AppState
        
        self.radio_class = AppState.radio_class # The radio
        
        # configure device
        self.panadapter = AppState.panadapter # the PanClass panadapter
        self.changef( self.radio_class.IF )

        super(ApplicationDisplay, self).__init__()
        
        self.N_WIN = 1024  # How many pixels to show from the FFT (around the center)
        self.N_AVG = AppState.fft_avg

        self.init_ui()
        self.StatusBarText.setText(f'Panadapter: <B>{self.panadapter.name}</B> Radio: <B>{self.radio_class.make} {self.radio_class.model}<\B>')
        self.qt_connections()

        pg.setConfigOptions(antialias=False)

        # Modeless windows
        self.ffttaper = FFTTaperingMenu()
        self.fftsize = FFTSizeMenu()
        
        #self.init_image()
        self.makeMenu()
        
        self.soapy_list_signal.connect(self.remakePanMenu)
        self.soapy_remote_pan_signal.connect(self.setSoapyRemote)
        self.soapy_pan_signal.connect(self.setSoapy)
        self.rtl_pan_signal.connect(self.setRtlsdr)
        self.fft_change_signal.connect(self.fft_change)

        if AppState.discover:
            AppState.discover.SoapyRegister(self.soapy_list_signal)

        # Show window
        self.resize(QtWidgets.QApplication.desktop().availableGeometry().width(),500) 
        self.show()
        
        # Data repository
        self.dataclass = Data()
        self.dataclass.new_complex()

        # Start reader in a separate thread
        self.data_reader = DataReader( self.dataclass )
        QtCore.QThreadPool.globalInstance().start(self.data_reader)
                
        # Start calculator in a separate thread
        self.psd = PSD( self.dataclass )
        QtCore.QThreadPool.globalInstance().start(self.psd)
                
        # Start timer for data collection
        self.timer = QtCore.QTimer() #default
        refresh = type(self).refresh
        self.timer.timeout.connect(self.update)
        print("Timer",refresh)
        self.timer.start(refresh)
        
    def fft_change( self ):
        if AppState.panadapter.Mode == 'Stream':
            # Stream
            AppState.panadapter.Stream( self.stream_read_signal, AppState.fft_avg*AppState.fft_size )
        self.N_WIN = int(AppState.fft_size / AppState.fft_ratio)

    def changef(self, F_SDR):
        global AppState
        AppState.panadapter.SetFrequency( F_SDR )
    
    def close(self):
        global AppState
        #AppState.panadapter.Close()
        
    def makeWaterfall( self, panel ):
        self.waterfall = Waterfall()
        
        panel.addItem(self.waterfall)
        panel.hideAxis("left")
        #self.plotwidget1.hideAxis("bottom")

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

        panmenu = menu.addMenu('&Panadapter')
        self.makePanMenu(panmenu)
        
        AppState.panadapter.Menu( menu, self )
        
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

        spectrummenu = menu.addMenu('&Spectrum')

        a = QtWidgets.QAction('FFT &Taper',self)
        spectrummenu.addAction(a)
        a.triggered.connect(lambda state, s=self: self.ffttaper.Menu_open(s))

        a = QtWidgets.QAction('FFT &Size',self)
        spectrummenu.addAction(a)
        a.triggered.connect(lambda state, s=self: self.fftsize.Menu_open(s))

        a = QtWidgets.QAction('&Autolevel',self)
        spectrummenu.addAction(a)
        a.triggered.connect(self.on_autolevel_clicked)
        
        m = spectrummenu.addMenu('&Colors')
        for c in Waterfall.Colors:
            a = QtWidgets.QAction('&'+c,self)
            a.triggered.connect(lambda state, c=c:self.waterfall.lookuptable(c))
            m.addAction(a)

    def remakePanMenu( self ):
        self.panmenu.clear()
        self.makePanMenu( self.panmenu )


    def makePanMenu( self, menu ):
        global AppState

        self.panmenu = menu

        m = QtWidgets.QAction('&Rescan sources',self)
        m.triggered.connect(self.remakePanMenu)
        menu.addAction(m)
        
        
        if AppState.discover:
            network = menu.addMenu("&Network devices")
            for ((address,port),name) in AppState.soapylist.items():
                #print("Pan",name)
                m = QtWidgets.QAction('{}\t\t{} : {}'.format(name,address,port),self)
                m.triggered.connect(lambda state,a=address,p=port,n=name: self.soapy_remote_pan_signal.emit({'address':a,'port':str(p),'name':n}))
                network.addAction(m)
            
        if Flag_audio:
            audio = menu.addMenu("&Audio devices")
            for (index,(name,rate)) in AppState.audiolist.items():
                #print("Pan",name)
                m = QtWidgets.QAction('{}. {}\t{}'.format(index,name,rate),self)
                m.triggered.connect(lambda state,i=index: self.setAudio(i))
                audio.addAction(m)

        if Flag_USB:
            dev = list(usb.core.find( find_all = True, idVendor = 0x0bda, idProduct = 0x2838 ))
            if len(dev) < 2:
                m = QtWidgets.QAction('RTLSDR',self)
                m.triggered.connect( lambda state,index=0: self.rtl_pan_signal.emit(index) )
                menu.addAction(m)
            else:
                rtl = menu.addMenu('&RTLSDR devices')
                for d in range(len(dev)):
                    m = QtWidgets.QAction('{}. {}\t{}',format(d,dev[d].iProduct,dev[d].iSerial),self)
                    m.triggered.connect(lambda state,index=d: self.rtl_pan_signal.emit(index) )
                    rtl.addAction(m) 

        if True:
            m = QtWidgets.QAction('Math &Waveforms',self)
            m.triggered.connect(self.setWaveform)
            menu.addAction(m)
            menu.addSeparator()
            
        if True:
            m = QtWidgets.QAction('Manual Entry',self)
            m.triggered.connect(lambda state, s=self: Manual(s).exec())
            menu.addAction(m)
            menu.addSeparator()
            
    def setManual( self ):
        global AppState
        
        # Enclosing Dialog
        dlg = QtWidgets.QDialog( self )
        dlg.setWindowTitle( "Manual Panadapter Entry" )
        dlg.resize(300,300)
        
        # Local Tab
        localtab = self.LocalTab(dlg)
        
        # Network Tab
        nettab = self.NetTab(dlg)
                
        # Tabbing structure 
        tab = QtWidgets.QTabWidget( dlg )
        tab.addTab( nettab, "Network" )
        tab.addTab( localtab, "Direct" )

        dlg.exec()
    
    def LocalTab(self,dlg):
        # Local Tab
        localtab = QtWidgets.QTabWidget()
        lst = {} # Local Tab subtabs
        for st in ['SoapySDR','USB','RTLSDR']:
            lst[st] = QtWidgets.QWidget()
            localtab.addTab(lst[st],st)
        return localtab

    def NetTab(self,dlg):
        # Network Tab
        nettab = QtWidgets.QFrame()
        nettablay = QtWidgets.QVBoxLayout()

        #Entry fields
        nettab1 = QtWidgets.QWidget()
        nettab1 = QtWidgets.QGroupBox("Network Address Entry")
        layout = QtWidgets.QFormLayout()
        layout.addRow(QtWidgets.QLabel("Host:"), QtWidgets.QLineEdit(inputMask='000.000.000.000'))
        layout.addRow(QtWidgets.QLabel("Port:"), QtWidgets.QLineEdit(inputMask='00000;',text='55132'))
        layout.addRow(QtWidgets.QLabel("Options:"), QtWidgets.QLineEdit())
        nettab1.setLayout(layout)
        nettablay.addWidget(nettab1)
        
        # Button fields
        nettab2 = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        nettab2.accepted.connect(dlg.accept)
        nettab2.rejected.connect(dlg.reject)
        nettablay.addWidget(nettab2)

        nettab.setLayout(nettablay)
        return nettab


    def setWaveform( self ):
        global AppState
        AppState.panadapter = WaveformPan()
        if AppState.resetNeeded:
            self.Loop(True)

    @QtCore.pyqtSlot(dict)
    def setSoapyRemote( self, dictionary):
        global AppState
        #print("Set SoapyRemote",dictionary)
        try: 
            AppState.panadapter = SoapyRemotePan(dictionary )
            if AppState.resetNeeded:
                self.Loop(True)
        except:
            pass

    @QtCore.pyqtSlot(dict)
    def setSoapy( self, dictionary):
        global AppState
        #print("Set Soapy",dictionary)
        try: 
            AppState.panadapter = SoapyPan(dictionary )
            if AppState.resetNeeded:
                self.Loop(True)
        except:
            pass

    @QtCore.pyqtSlot(int)
    def setRtlsdr( self, index ):
        global AppState
        #print("RTLSDR index=",index)
        try:
            newpan = RTLSDR(index=index)
            AppState.panadapter = newpan
        except:
            pass
        if AppState.resetNeeded:
            self.Loop(True)

    def setAudio( self, index ):
        # use Audio input
        global AppState
        AppState.panadapter = AudioPan(index)
        if AppState.resetNeeded:
            self.Loop(True)

    def makeSpectrum( self, panel ):
        self.spectrum_plot = panel.plot()
        panel.setYRange(-250, -100, padding=0.)
        #panel.showGrid(x=True, y=True)

        panel.hideAxis("left")
        panel.setLabel("bottom",None,'Hz')
        #panel.hideAxis("bottom")

    def init_ui(self):
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle('PEPYSCOPE - IS0KYB')
        
        self.split = QtWidgets.QSplitter()
        self.split.setOrientation(QtCore.Qt.Vertical)
        
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.split)

        Panels.clear()
        self.s_pan = Panels( "Spectrogram", self.makeSpectrum, self.split )
        self.w_pan = Panels( "Waterfall", self.makeWaterfall, self.split )

        hbox = QtWidgets.QHBoxLayout()

        self.zoominbutton = QtWidgets.QPushButton("ZOOM IN")
        self.zoomoutbutton = QtWidgets.QPushButton("ZOOM OUT")
        self.modechange = QtWidgets.QPushButton(TransmissionMode.mode().__name__)
        self.invertscroll = QtWidgets.QPushButton("Scroll")
        self.autolevel = QtWidgets.QPushButton("Auto Levels")

        hbox.addWidget(self.zoominbutton)
        hbox.addWidget(self.zoomoutbutton)
        hbox.addWidget(self.modechange)
        hbox.addWidget(self.invertscroll)
        hbox.addStretch()

        hbox.addWidget(self.autolevel)

        vbox.addLayout(hbox)
        self.win.setLayout(vbox)

        self.setGeometry(10, 10, 1024, 512)
        self.setCentralWidget(self.win)
        
        self.StatusBarFrame = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.StatusBarFrame)
        self.StatusBarText = QtWidgets.QLabel()
        self.StatusBarFrame.addWidget(self.StatusBarText)

    def qt_connections(self):
        self.zoominbutton.clicked.connect(self.on_zoominbutton_clicked)
        self.zoomoutbutton.clicked.connect(self.on_zoomoutbutton_clicked)
        self.modechange.clicked.connect(self.on_modechange_clicked)
        self.invertscroll.clicked.connect(self.on_invertscroll_clicked)

    def on_modechange_clicked(self):
        TransmissionMode.next()
        self.modechange.setText(TransmissionMode.mode().__name__)

    def on_autolevel_clicked(self):
        minlev, maxlev = self.waterfall.autolevel()
        self.s_pan.plot.setYRange(minlev, maxlev, padding=0.3)

    def on_invertscroll_clicked(self):
        global AppState
        AppState.scroll *= -1
        self.waterfall.init_image()

    def on_zoominbutton_clicked(self):
        if AppState.fft_ratio<512:
            AppState.fft_ratio *= 2
    
    def on_zoomoutbutton_clicked(self):
        global AppState
        if AppState.fft_ratio>1:
            AppState.fft_ratio /= 2
 
    def update(self):
        global AppState

        self.psd.lock.lock()
        psd = self.psd.psd
        self.psd.lock.unlock()

        # Plot the grid
        self.waterfall.image_update(psd)

#        self.text_leftlim.setPos(0, 0)
#        self.text_leftlim.setText(text="-%.1f kHz"%(bw_hz/2000./self.fft_ratio))
#        #self.text_rightlim.setPos(bw_hz*1000, 0)
#        self.text_rightlim.setText(text="+%.1f kHz"%(bw_hz/2000./self.fft_ratio))

#        self.spectrum_plot.setData(np.arange(0,psd.shape[0]), -psd, pen="g")
        hz = AppState.panadapter.SampleRate/4
        self.spectrum_plot.setData(np.linspace(-hz,hz,psd.shape[0]), psd, pen="g")

        #self.plotwidget2.plot(x=[0,0], y=[-240,0], pen=pg.mkPen('r', width=1))
        #self.plotwidget2.plot(x=[self.N_WIN/2, self.N_WIN//2], y=[-240,0], pen=pg.mkPen('r', width=1))
        #self.plotwidget2.plot(x=[self.N_WIN-1, self.N_WIN-1], y=[-240,0], pen=pg.mkPen('r', width=1))



    def Loop(self, y_n ):
        global AppState
        AppState.Loop=y_n
        QtWidgets.qApp.quit()
        
    def __del__(self):
        # Kill the thread too
        try:
            self.data_reader.loop = False
            self.psd.loop = False
            print("Success")
        except:
            print("Problem")
            pass
        
class DataReader(QtCore.QRunnable):
    def __init__(self, dataclass ):
        super().__init__()
        self.dataclass = dataclass
        self.chunk_size = self.dataclass.chunk_size
        self.loop = True # can change from affar to stop loop

    def run(self):
        global AppState
        if AppState.panadapter.Mode == 'Block':
            while self.loop:
                self.dataclass.add(AppState.panadapter.Read(self.chunk_size))
       
        else:
            # Stream
            AppState.panadapter.Stream( self.dataclass.add, self.chunk_size )


class Panels():
    # manage the displaypanels -- waterfall and spectrogram so far
    # some complicated logic for menu system to never allow no panels, and disable the menu entry that might allow it, to be clearer.
    # actually can handle an arbitrary number of panels
    List = []
    def __init__(self, name, func, split):
        self.name = name
        self._plot = PltWidget() # for custom context window
        func(self._plot)
        split.addWidget(self._plot)
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
            self._plot.setVisible( False )
            self.visible = False
            v = [ p for p in type(self).List if p.visible ]
            if len(v) == 1:
                v[0].menu.setDisabled( True )        
        else:
            self._plot.setVisible( True )
            self.visible = True
            for p in type(self).List:
                p.menu.setDisabled( False )
        
    @property
    def plot( self ):
        return self._plot

class Manual(QtWidgets.QDialog):
    def __init__(self, caller):
        global AppState
        
        super(Manual,self).__init__(caller)
        
        self.caller = caller
        
        # Enclosing Dialog
        self.setWindowTitle( "Manual Panadapter Entry" )
        self.resize(350,400)
        
        # Local Tab
        localtab = self.LocalTab(caller.soapy_pan_signal, self.local_parser)
        
        # Network Tab
        nettab = self.NetTab(caller.soapy_remote_pan_signal, self.network_parser)
                
        # Tabbing structure 
        tab = QtWidgets.QTabWidget(self)
        tab.addTab( nettab, "Network" )
        tab.addTab( localtab, "Direct" )

    def _toggle_local( self, name, driver, extra, qoption ):
        qoption.setText(extra+'=')
        qoption.setCursorPosition(0)
        self.driver = driver
        self.name = name

            
    def LocalTab(self, signal, parser):
        # Local Tab
        group = QtWidgets.QGroupBox("Select SoapySDR Driver")
        vbox = QtWidgets.QVBoxLayout()

        # Predefine
        self.qlocal_option = QtWidgets.QLineEdit(inputMask='annnnnnnnnnnnn=xxxxxxxxxxxxxxxxxxxxxxxxxx')
                        
        scroll = QtWidgets.QScrollArea(group)
        table = QtWidgets.QWidget()
        tbox = QtWidgets.QVBoxLayout()
        for (name,driver,extra) in [
        ( 'AirSpy',         'airspy',   '' ),
        ( 'AirSpy HF+',     'airspyhf', '' ),
        ( 'Audio',          'audio',    'label' ),
        ( 'BladeRF',        'bladerf',  '' ),
        ( 'Epiq Sidekiq',   'sidekiq',  '' ),
        ( 'FunCube Dondle Pro+', 'fcdpp', '' ),
        ( 'Hack RF',        'hackrf',   '' ),
        ( 'Lime LMS7',      'lime',     '' ),
        ( 'NetSDR',         'netsdr',   '' ),
        ( 'Novena RF',      'novena',   '' ),
        ( 'OsmoSDR',        'osmosdr',  '' ),
        ( 'PlutoSDR',       'plutosdr', 'hostname'),
        ( 'Red Pitya',      'redpitya', 'addr' ),
        ( 'RTL-SDR',  'rtlsdr',   '' ),
        ( 'SDR Play',       'sdrplay',  '' ),
        ( 'Skylark Iris',   'iris',     '' ),
        ( 'UHD',            'uhd',      'type' ),
        ]:
            b = QtWidgets.QRadioButton('&'+name,table)
            b.toggled.connect(lambda n=name, d=driver, e=extra, qo=self.qlocal_option: self._toggle_local(n,d,e,qo))
            tbox.addWidget(b)
        table.setLayout(tbox)

        #Scroll Area Properties
        scroll.setFixedHeight(200)
        scroll.setWidget(table)
        
        vbox.addWidget(scroll)
        
        frame = QtWidgets.QFrame()
        form = QtWidgets.QFormLayout()
        
        form.addRow("Options:", self.qlocal_option)

        # Ok Cancel
        form.addRow(self.OkCancel(signal,parser,self.qlocal_option))
        frame.setLayout(form)
        
        vbox.addWidget(frame)
        group.setLayout(vbox)

        return group

    def NetTab(self, signal, parser):
        # Network Tab
        group = QtWidgets.QGroupBox("Network Address Entry")
        form = QtWidgets.QFormLayout()
        
        # Entry fields
        self.qname=QtWidgets.QLineEdit('SoapyRemote')
        form.addRow("Name:", self.qname)

        self.qaddress=QtWidgets.QLineEdit(inputMask='000.000.000.000')
        form.addRow("Host:", self.qaddress)
        
        self.qport=QtWidgets.QLineEdit(inputMask='00000;',text='55132')
        form.addRow("Port:", self.qport)
        
        self.qnet_option = QtWidgets.QLineEdit(inputMask='annnnnnnnnnnnn=xxxxxxxxxxxxxxxxxxxxxxxxxx')
        form.addRow("Options:", self.qnet_option)

        # Ok Cancel
        form.addRow(self.OkCancel(signal,parser,self.qnet_option))
        group.setLayout(form)
        
        return group
        
    def OkCancel( self, signal, parser, qoption ):
        # Button fields
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda s=signal,p=parser, qo=qoption: self.accept(s,p,qo))
        buttons.rejected.connect(self.reject)
        return buttons
        
    def accept( self, signal, parser, qoption ):
        signal.emit( parser() )
        self.close()
    
    def option_parse( self, option ):
        try:
            oo = option.text().split('=',1)
            if oo[0] == '':
                return None
        except:
            return None
        return oo

    def network_parser( self ):
        arg={'address':self.qaddress.text(),'port':self.qport.text(),'name':self.qname.text()}
        oo = self.option_parse(self.qnet_option)
        if oo:
            arg[oo[0]] = oo[1]
        return arg

    def local_parser( self ):
        arg={'driver':self.driver,'name':self.name}
        oo = self.option_parse(self.qlocal_option)
        if oo:
            arg[oo[0]] = oo[1]
        return arg

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
        self.signal_complete = None 
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
        #print('New ServiceBrowser: {}'.format(browser_path.arguments()))
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
            AppState.SoapyAdd( resolved[7],resolved[8],resolved[2] )
            if self.signal_complete:
                self.signal_complete.emit()
        except:
            pass

    @QtCore.pyqtSlot(QtDBus.QDBusMessage)
    def onItemRemove(self, msg):
        global AppState
        AppState.SoapyDel( msg.arguments()[2] )
        if self.signal_complete:
            self.signal_complete.emit()
        
    def SoapyRegister( self, signal_complete ):
        self.signal_complete = signal_complete 

    @QtCore.pyqtSlot(QtDBus.QDBusMessage)
    def onAllForNow(self, msg):
        if self.signal_complete:
            self.signal_complete.emit()

def CommandLine():
    """Setup argparser object to process the command line"""
    cl = argparse.ArgumentParser(description="PyPanadapter - radio panadapter using an PanClass dongle on the IF (intermediate frequency of a radio by Paul H Alfille based on code of Marco Cogoni")
    cl.add_argument("-s","--sdr",help="Panadapter model",choices=[c.__name__ for c in PanClass.List(2)],nargs='?',default="RTLSDR")
    cl.add_argument("-r","--radio",help="Radio model",choices=[r.__name__ for r in Radio.List()],nargs='?',default=Radio.List()[0].__name__)
    cl.add_argument("-i","--if",help="Intermediate frequency -- overwrites radio default",type=float)
    return cl.parse_args()

def main(args):
    global AppState
    args = CommandLine() # Get args from command line

    TransmissionMode.next() # prime mode list

    pan_class = PanClass.Match( args.sdr, 2 )
    if pan_class:
        AppState.panadapter = pan_class()
    else:
        AppState.panadapter = WaveformPan()

    AppState.radio_class = Radio.Match( args.radio )
    
    while AppState.Loop:
        app = QtWidgets.QApplication([])
        if not AppState.discover:
            try:
                AppState.discover = Discoverer(app, '_soapy._tcp')
            except:
                AppState.discover = None
        display = ApplicationDisplay()
        app.exec_()
        display.data_reader.loop = False
        display.psd.loop = False
        display = None
        app = None

def signal_handler( signal, frame ):
    # Signal handler
    # signal.signal( signal.SIGINT, signal.SIG_IGN )
    sys.exit(0)

if __name__ == '__main__':
    # Set up keyboard interrupt handler
    signal.signal(signal.SIGINT, signal_handler )
    # Start program
    sys.exit(main(sys.argv))
