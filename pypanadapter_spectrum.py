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
from time import sleep
import math
import sys
import signal
import argparse # for parsing the command line

from scipy.signal import welch, decimate
import numpy as np
import pyqtgraph as pg

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
    
FFT_SIZE = 2048
FRAME_RATE = 8 # data refesh rate in Hz

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

class Custom(Radio):
    make = "Custom"
    model = "Specified"
    # Intermediate Frquency
    def __init(self,IF):
        self.IF = IF

class PanClass(SubclassManager):
    # Pan adapter device including RTLSDR
    _name = "Panadapter"
    def __init(self):
        self.driver = None
        
    @property
    def N_AVG(self):
        return int(self.SampleRate / FFT_SIZE / FRAME_RATE )
        
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
        
    def Menu( self, menu, parent ):
        return menu.addMenu( "Pan &Configure" )

class PanStreamClass(PanClass):
    # All PanAdapters that stream Data
    _name = "None"
    Mode = 'Stream'

    def __init(self):
        super().__init__()
        
    def Stream( self, update_signal , chunk_size ) :
        self.update_signal = update_signal
        self.chunk_size = chunk_size
        self.StartStream()

    def emitter( self, data ):
        # sends data to Application Display
        self.update_signal.emit( data )
                
class PanBlockClass(PanClass):
    # All PanAdapters that stream Data
    _name = "None"
    Mode = 'Block'

    def __init(self):
        super().__init__()
        
class RTLSDR(PanBlockClass):
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
                self.driver = RtlSdr(serial_number = self.serial)
                self._name = f'RTLSDR serial {serial}'
            elif self.index:
                self.driver = RtlSdr(self.index)
                self._name = f'RTLSDR index {serial}'
            elif self.host:
                self.driver = RtlSdrTcpClient( hostname=self.host, port=self.port ) 
                self._name = f'RTLSDR @{host}:{port}'
            else:
                self.driver = RtlSdr()
                self_name = 'RTLSDR'
        except:
            print("RTLSDR not found")
            self.driver = None
            return
        self.driver.sample_rate = self.SampleRate
        
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
        if not Flag_audio:
            self.driver = None
            return
        if AppState.audio:
            self.driver = None
            try:
                info = AppState.audio.get_device_info_by_index( index)
                self._name = info['name']
                self.SampleRate = info['defaultSampleRate']
                self.driver = 'quiet'
            except:
                print("Could not open audio device")
                
        self.index = index
    
    def Stream( self, update_function , chunk_size ) :
        if not self.driver:
            return
        try:
            self.driver = AppState.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                frames_per_buffer=chunk_size,
                rate=int(self.SampleRate),
                input=True,
                output=False,
                stream_callback = self.audio_callback,
                input_device_index = self.index
                )
        except:
            self.driver = None
            print(self.name,"Could not start audio streaming")
        super().Stream( update_function, chunk_size )

    def StartStream( self ) :
        if self.driver:
            self.driver.start_stream()
        
    def __del__(self):
        if self.driver:
            self.driver.close()
        
    def SetFrequency(self, IF ):
        self.center_freq = IF
        
    def audio_callback( self, in_data, frame_count, time_info, status_flags ):
        self.emitter( np.frombuffer( in_data, 'float32' ) ) 
        return (None, pyaudio.paContinue)

    def Close(self):
        if self.driver:
            #shutdown the stream
            self.driver.deactivateStream(self.rxStream) #stop streaming
            self.driver.closeStream(self.rxStream)
        self.driver = None

class Waveform(SubclassManager):
    _name = "Default Waveform"
    def __init__(self):
        self._size = 0
        self._phase = 0.
        self.new_phase = self._phase
        self._cycles = 1.
        self.new_cycles = self._cycles
        
    @property
    def name(self):
        return type(self)._name
        
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
        
    def makeArray(self):
         self.data = (np.zeros(self._size)+.00000000001) + (np.zeros(self._size)+.0000000001)*1j
       
    def create( self ):
        # default
        pass
            
    def Read( self, size ):
        #print("Size",self._size)
        if (self._size != size) or (self._cycles != self.new_cycles) or (self._phase != self.new_phase):
            print("Create",size,self._cycles,self._phase)
            
            self._size = size
            self._phase = self.new_phase
            self._cycles = self.new_cycles
            
            # tests
            if self._cycles > self._size:
                self._cycles = self._size
            if self._cycles <= 0.:
                self._cycles = 1.
            self._phase %= 360.
            if self._phase < 0.:
                self._phase += 360.
                
            self.new_phase = self._phase
            self.new_cycles = self._cycles
            print("Create",size,self._cycles,self._phase)
                        
            self.makeArray() 
            self.create()
        return self.data
        
class Square(Waveform):
    _name = "Square Waveform"
    def __init__(self):
        super().__init__()
        
    def create(self):
        cycle_length = int(self._size / self._cycles)
        half_cycle_length = cycle_length // 2

        for i in range( 0,self._size,cycle_length ):
            for j in range(i,i+half_cycle_length):
                if j >= self._size:
                    break
                self.data[j] += 1
                
        #phase correction
        np.roll(self.data,int(cycle_length*self._phase/360.))
            
class Sawtooth(Waveform):
    _name = "Sawtooth Waveform"
    def __init__(self):
        super().__init__()
        
    def create(self):
        cycle_length = int(self._size / self._cycles)
        delta = 2.0 / cycle_length
        
        x = -1.
        for i in range( 0,self._size,cycle_length ):
            x=-1.
            for j in range(i,i+cycle_length):
                if j >= self._size:
                    break
                self.data[j] += x
                x += delta
                
        #phase correction
        np.roll(self.data,int(cycle_length*self._phase/360.))

class Triangle(Waveform):
    _name = "Triangle Waveform"
    def __init__(self):
        super().__init__()
        
    def create(self):
        s = Sawtooth()
        s.phase = self._phase
        s.cycles = self._cycles
        a = s.Read( self._size )
        self.data = 2*np.maximum(a,-a)-1

class Digits(Waveform):
    _name = "Digit Waveform"
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
                
        #phase correction
        np.roll(self.data,int(cycle_length*self._phase/360.))

        # normalize
        self.data /= mx
        
class Sine(Waveform):
    _name = "Sine Waveform"
    def __init__(self):
        super().__init__()
        
    def create(self):
        dangle = 2 * np.pi / (self._size*self._cycles)
        angle = 2 * np.pi * (self._phase / 360.)
        for i in range(self._size):
            self.data[i] = np.sin(angle) + np.cos(angle)*1j
            angle += dangle

class Random(Waveform):
    def __init__(self):
        pass
        
    def Read(self,size):
        return 2*(np.random.random(size)+np.random.random(size)*1j)-(1.+1.j)

class WaveformPan(PanBlockClass):
    SampleRate = 2.56E6 
    _name = "Waveform"

    def __init__(self):
        self.driver = True
        self.squarewave = False
        self.modules = [l.__name__ for l in Waveform.List()]
        if not Flag_random:
            self.modules.remove('Random')
        self.phase = 0.
        self.cycles = 1.
        self._waveform_select( self.modules[0] )
        
    def SetFrequency(self, IF ):
        pass
        
    def Read(self,size):
        return self.waveform.Read( int(size) )
        
    def Close(self):
        self.driver = None
    
    def _waveform_select( self, w ):
        self.module_chosen = w
        self.waveform = Waveform.Match( w )()
        self.name = self.waveform.name
        self.waveform.phase = self.phase 
        self.waveform.cycles = self.cycles 
        
    def Phase( self, parent ):
        print(parent)
        p, ok = QtWidgets.QInputDialog.getDouble( parent, 'Phase Angle', 'Enter in degrees:', self.phase, 0., 360., 2 )
        if ok:
            self.waveform.phase = p
            self.phase = p
        
    def Cycles( self, parent ):
        print(parent)
        c, ok = QtWidgets.QInputDialog.getDouble( parent, 'Cycles', 'Enter repeats:', self.cycles, 0.1 )
        if ok:
            self.waveform.cycles = c
            self.cycles = c
        
    def Menu( self, menu, parent ):
        this_menu = super().Menu(menu, parent )
        
        waveform = this_menu.addMenu('&Waveform')
        waveaction = QtWidgets.QActionGroup(waveform)
        for w in self.modules:
            m = QtWidgets.QAction('&'+w,waveaction,checkable=True,checked=(w==self.module_chosen))
            m.triggered.connect(lambda state, x=w: self._waveform_select(x))
            waveform.addAction(m)
        
        phase = QtWidgets.QAction('&Phase...',parent)
        phase.triggered.connect(lambda state, p=parent: self.Phase(p) )
        this_menu.addAction(phase)
        
        cycles = QtWidgets.QAction('&Cycles...',parent)
        cycles.triggered.connect(lambda state, p=parent: self.Cycles(p) )
        this_menu.addAction(cycles)

#        this_menu.addAction(waveform)
        return this_menu

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
        
 
class ProgramState:
    # holds program "state"
    def __init__( self, panadapter=None, radio_class=None ):
        self._panadapter = panadapter
        self._radio_class = radio_class
        self._resetNeeded = False
        self._Loop = True # for initial entry into loop
        self._soapylist = {}
        self.fft_window='hamming'
        self.discover = None
        if Flag_audio:
            sys.stderr.write("\n--------------------------------------------------------\n\tSetup output from PyAudio follows -- usually can be ignored\n")
            self.audio = pyaudio.PyAudio()
            sys.stderr.write("\tEnd of PyAudio setup\n--------------------------------------------------------\n\n")
        else:
            self.audio = None
            
    def setWindow( self, win ):
        self.fft_window = win

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

class ApplicationDisplay(QtWidgets.QMainWindow):
    # Display class
    #define a custom signal
    block_read_signal = QtCore.pyqtSignal(np.ndarray)
    stream_read_signal = QtCore.pyqtSignal(np.ndarray)
    soapy_list_signal = QtCore.pyqtSignal()
    soapy_remote_pan_signal = QtCore.pyqtSignal(dict)
    soapy_pan_signal = QtCore.pyqtSignal(dict)
    rtl_pan_signal = QtCore.pyqtSignal(int)

    refresh = 50 # default refresh timer in msec

    def __init__(self ):
        # Comes in with Panadapter set and readio_class set.
        global AppState
        
        self.radio_class = AppState.radio_class # The radio
        
        # configure device
        self.panadapter = AppState.panadapter # the PanClass panadapter
        self.changef( self.radio_class.IF )

        super(ApplicationDisplay, self).__init__()
        
        self.N_WIN = 1024  # How many pixels to show from the FFT (around the center)
        self.N_AVG = AppState.panadapter.N_AVG
        self.fft_ratio = 2.

        self.init_ui()
        self.StatusBarText.setText(f'Panadapter: <B>{self.panadapter.name}</B> Radio: <B>{self.radio_class.make} {self.radio_class.model}<\B>')
        self.qt_connections()

        pg.setConfigOptions(antialias=False)

        self.scroll = -1
        
        self.init_image()
        self.makeMenu()

        self.block_read_signal.connect(self.update)
        self.stream_read_signal.connect(self.read_callback)
        self.soapy_list_signal.connect(self.remakePanMenu)
        self.soapy_remote_pan_signal.connect(self.setSoapyRemote)
        self.soapy_pan_signal.connect(self.setSoapy)
        self.rtl_pan_signal.connect(self.setRtlsdr)

        if AppState.discover:
            AppState.discover.SoapyRegister(self.soapy_list_signal)

        # Show window
        self.show()
        
        if AppState.panadapter.Mode == 'Block':
            # Start timer for data collection
            self.timer = QtCore.QTimer() #default
            self.refresh = type(self).refresh
            self.timer.timeout.connect(self.read)
            self.timer.start(self.refresh)
        else:
            # Stream
            AppState.panadapter.Stream( self.stream_read_signal, self.N_AVG*FFT_SIZE )
                
    def read(self):
        global AppState
        # Block mode only
        if TransmissionMode.changed():
            self.changef(TransmissionMode.mode().freq(self.radio_class))
        self.block_read_signal.emit(AppState.panadapter.Read(self.N_AVG*FFT_SIZE))
            
    def read_callback(self,chunk):
        # Stream mode only
        if TransmissionMode.changed():
            self.changef(TransmissionMode.mode().freq(self.radio_class))
        self.update( chunk )
            

    def changef(self, F_SDR):
        global AppState
        AppState.panadapter.SetFrequency( F_SDR )
    
    def close(self):
        global AppState
        #AppState.panadapter.Close()
        
    def Loop(self, y_n ):
        global AppState
        AppState.Loop=y_n
        self.qApp.quit()

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
        bw_hz = AppState.panadapter.SampleRate/float(FFT_SIZE) * float(self.N_WIN)/1.e6/self.fft_ratio
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

        fftmenu = menu.addMenu('&Spectrum')
        win_type = fftmenu.addMenu('FFT &window')
        win_action_group = QtWidgets.QActionGroup(win_type)
        for w in [
            'barthann',
            'bartlett',
            'blackmanharris',
            'blackman',
            'bohman',
            'boxcar',
            'flattop',
            'hamming',
            'hann',
            'parzen',
            'nuttall',
            'triang'
            ]:
            m = QtWidgets.QAction('&'+w,win_action_group,checkable=True,checked=(AppState.fft_window==w))
            m.triggered.connect(lambda state, win=w: AppState.setWindow(win))
            win_type.addAction(m)
        # make sure at least one is chosen
        if not win_action_group.checkedAction():
            AppState.setWindow('hamming')
            
        

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
        
        self.StatusBarFrame = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.StatusBarFrame)
        self.StatusBarText = QtWidgets.QLabel()
        self.StatusBarFrame.addWidget(self.StatusBarText)

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

    def on_avg_decrease_clicked(self):
        if self.N_AVG>1:
            self.N_AVG /= 2


    def on_modechange_clicked(self):
        TransmissionMode.next()
        self.modechange.setText(TransmissionMode.mode().__name__)

    def on_autolevel_clicked(self):
        tmp_array = np.copy(self.img_array[self.img_array>0])
        tmp_array = tmp_array[tmp_array<250]
        tmp_array = tmp_array[:]
        #print( tmp_array.shape )

        self.minminlev = np.percentile(tmp_array, 99)
        self.minlev = np.percentile(tmp_array, 80)
        self.maxlev = np.percentile(tmp_array, 0.3)
        #print( self.minlev, self.maxlev )
        self.waterfall.setLevels([self.minlev, self.maxlev])

        self.s_pan.plot.setYRange(-self.minminlev, -self.maxlev, padding=0.3)

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
        t_total = (1/AppState.panadapter.SampleRate) * FFT_SIZE * self.N_AVG
        t = np.arange(0, t_total, 1 / AppState.panadapter.SampleRate)
        lo = 2**.5 * np.exp(-2j*np.pi*f_demod * t) # local oscillator
        x_mix = x*lo
        
        power2 = int(np.log2(ratio))
        for mult in range(power2):
            x_mix = decimate(x_mix, 2) # mix and decimate

        return x_mix 

    def update(self, chunk):
        # update the displays with the new data (chunk)
        global AppState
        bw_hz = AppState.panadapter.SampleRate/float(FFT_SIZE) * float(self.N_WIN)
        self.win.setWindowTitle('PEPYSCOPE - IS0KYB - N_FFT: %d, BW: %.1f kHz' % (FFT_SIZE, bw_hz/1000./self.fft_ratio))

        if self.fft_ratio>1:
            chunk = self.zoomfft(chunk, self.fft_ratio)

        sample_freq, spec = welch(chunk, AppState.panadapter.SampleRate, window=AppState.fft_window, nperseg=FFT_SIZE,  nfft=FFT_SIZE)
        spec = np.roll(spec, FFT_SIZE//2, 0)[FFT_SIZE//2-self.N_WIN//2:FFT_SIZE//2+self.N_WIN//2]
        
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

class Panels():
    # manage the displaypanels -- waterfall and spectrogram so far
    # some complicated logic for menu system to never allow no panels, and disable the menu entry that might allow it, to be clearer.
    # actually can handle an arbitrary number of panels
    List = []
    def __init__(self, name, func, split):
        self.name = name
        self._plot = pg.PlotWidget()
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
#        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
#        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#        scroll.setWidgetResizable(True)
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
