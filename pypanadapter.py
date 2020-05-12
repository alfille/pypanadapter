from rtlsdr import *
from time import sleep
import math
import numpy as np
from scipy.signal import welch, decimate
import pyqtgraph as pg
#import pyaudio
#from PyQt4 import QtCore, QtGui
from PyQt5 import QtCore, QtWidgets

#FS = 1.0e6 # Sampling Frequency of the RTL-SDR card (in Hz) # DON'T GO TOO LOW, QUALITY ISSUES ARISE
N_AVG = 32 # averaging over how many spectra

class Radio:
        pass

class TS180S(Radio):
    name = "Kenwood TS-180S"
    # Intermediate Frquency
    IF = 8.8315E6

class X5105(Radio):
    name = "Xiegu X5105"
    # Intermediate Frquency
    IF = 70.455E6
    
class SDR:
    def Close(self):
        pass
    
class RTLSDR(SDR):
    SampleRate = 2.56E6  # Sampling Frequency of the RTL-SDR card (in Hz) # DON'T GO TOO LOW, QUALITY ISSUES ARISE
    name = "RTL-SDR"
    def __init__(self):
        self.sdr = RtlSdr()
        self.sdr.sample_rate = self.SampleRate
        
    def SetFrequency(self, IF ):
        if IF < 30.E6 :
            #direct sampling needed (ADC Q vs 1=ADC I)
            self.sdr.set_direct_sampling(2)
        else:
            self.sdr.set_direct_sampling(0)
            
        self.sdr.center_freq = IF
        
    def Read(self,size):
        # IQ inversion to correct low-high frequencies
        return np.flip(self.sdr.read_samples(size))
        
    def Close(self):
        self.sdr.close()

class PanAdapter():
    def __init__(self, sdr, radio = TS180S ):
        self.mode = 0 # LSB or USB
        
        self.radio = radio # The radio
        
        # configure device
        self.sdr = sdr # the SDR panadapter
        self.sdr.SetFrequency( self.radio.IF )

        self.widget = SpectrogramWidget(sdr)
                
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

class SpectrogramWidget(pg.PlotWidget):

    #define a custom signal
    read_collected = QtCore.pyqtSignal(np.ndarray)

    def __init__(self,sdr):
        super(SpectrogramWidget, self).__init__()

        self.sdr = sdr

        self.init_ui()
        self.qt_connections()
        self.waterfall = pg.ImageItem()
        self.plotwidget1.addItem(self.waterfall)
    
        self.N_FFT = 2048 # FFT bins
        self.N_WIN = 1024  # How many pixels to show from the FFT (around the center)
        self.fft_ratio = 2.

        self.mode = 0 # USB=0, LSB=1: defaults to USB
        self.scroll = -1
        
        self.minlev = 220
        self.maxlev = 140

        self.init_image()

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

        # set colormap
        self.waterfall.setLookupTable(lut)
        self.waterfall.setLevels([self.minlev, self.maxlev])

        # setup the correct scaling for x-axis
        self.bw_hz = FS/float(self.N_FFT) * float(self.N_WIN)/1.e6/self.fft_ratio
        self.waterfall.scale(self.bw_hz,1)
        self.setLabel('bottom', 'Frequency', units='kHz')
        
        self.text_leftlim = pg.TextItem("-%.1f kHz"%(self.bw_hz*self.N_WIN/2.))
        self.text_leftlim.setParentItem(self.waterfall)
        self.plotwidget1.addItem(self.text_leftlim)
        self.text_leftlim.setPos(0, 0)

        self.text_rightlim = pg.TextItem("+%.1f kHz"%(self.bw_hz*self.N_WIN/2.))
        self.text_rightlim.setParentItem(self.waterfall)
        self.plotwidget1.addItem(self.text_rightlim)
        self.text_rightlim.setPos(self.bw_hz*(self.N_WIN-64), 0)

        self.plotwidget1.hideAxis("left")
        self.plotwidget1.hideAxis("bottom")

        self.hideAxis("top")
        self.hideAxis("bottom")
        self.hideAxis("left")
        self.hideAxis("right")
        
        self.read_collected.connect(self.update)

        self.win.show()

    def init_image(self):
        self.img_array = 250*np.ones((self.N_WIN, self.N_WIN))
        # Plot the grid
        for x in [0, self.N_WIN//2, self.N_WIN-1]:
            if x==0 or x==self.N_WIN-1:
                self.img_array[:,x] = 0
            else:
                self.img_array[:,x] = 0


    def init_ui(self):
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle('PEPYSCOPE - IS0KYB')
        
        vbox = QtWidgets.QVBoxLayout()
        #self.setLayout(vbox)

        self.plotwidget1 = pg.PlotWidget()
        vbox.addWidget(self.plotwidget1)

        hbox = QtWidgets.QHBoxLayout()

        self.zoominbutton = QtWidgets.QPushButton("ZOOM IN")
        self.zoomoutbutton = QtWidgets.QPushButton("ZOOM OUT")
        self.modechange = QtWidgets.QPushButton("USB")
        self.invertscroll = QtWidgets.QPushButton("Scroll")
        self.autolevel = QtWidgets.QPushButton("Auto Levels")

        hbox.addWidget(self.zoominbutton)
        hbox.addWidget(self.zoomoutbutton)
        hbox.addWidget(self.modechange)
        hbox.addWidget(self.invertscroll)
        hbox.addStretch()

        hbox.addWidget(self.autolevel)

        #vbox.addStretch()
        vbox.addLayout(hbox)
        self.win.setLayout(vbox)

        self.win.setGeometry(10, 10, 1024, 512)
        self.win.show()

    def qt_connections(self):
        self.zoominbutton.clicked.connect(self.on_zoominbutton_clicked)
        self.zoomoutbutton.clicked.connect(self.on_zoomoutbutton_clicked)
        self.modechange.clicked.connect(self.on_modechange_clicked)
        self.invertscroll.clicked.connect(self.on_invertscroll_clicked)
        self.autolevel.clicked.connect(self.on_autolevel_clicked)

    def on_modechange_clicked(self):
        if self.mode == 0:
            self.modechange.setText("LSB")
        elif self.mode == 1:
            self.modechange.setText("USB")
        self.mode += 1
        if self.mode>1:
            self.mode = 0


    def on_autolevel_clicked(self):
        self.minlev = np.percentile(self.img_array, 95)-20
        self.maxlev = np.percentile(self.img_array, 5)-80
        print( self.minlev, self.maxlev )
        self.waterfall.setLevels([self.minlev, self.maxlev])


    def on_invertscroll_clicked(self):
        self.scroll *= -1
        self.init_image()

    def on_zoominbutton_clicked(self):
        if self.fft_ratio<128:
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
        self.bw_hz = FS/float(self.N_FFT) * float(self.N_WIN)
        self.win.setWindowTitle('PEPYSCOPE - IS0KYB - N_FFT: %d, BW: %.1f kHz' % (self.N_FFT, self.bw_hz/1000./self.fft_ratio))

        if self.fft_ratio>1:
            chunk = self.zoomfft(chunk, self.fft_ratio)

        sample_freq, spec = welch(chunk, FS, window="hamming", nperseg=self.N_FFT,  nfft=self.N_FFT)
        spec = np.roll(spec, self.N_FFT//2, 0)[self.N_FFT//2-self.N_WIN//2:self.N_FFT//2+self.N_WIN//2]
        
        # get magnitude 
        psd = abs(spec)
        # convert to dB scale
        psd = -20 * np.log10(psd)

        # Plot the grid
        for x in [0, self.N_WIN//2, self.N_WIN-1]:
            psd[x] = 0            

        # roll down one and replace leading edge with new data
        self.img_array[-1:] = psd
        self.img_array = np.roll(self.img_array, -1*self.scroll, 0)

        for i, x in enumerate(range(0, self.N_WIN-1, (self.N_WIN)//10)):
            if i!=5 and i!=10:
                if self.scroll>0:
                    for y in range(5,15):
                        self.img_array[y,x] = 0
                elif self.scroll<0:
                    for y in range(-10,-2):
                        self.img_array[y,x] = 0


        self.waterfall.setImage(self.img_array.T, autoLevels=False, opacity = 1.0, autoDownsample=True)

        self.text_leftlim.setPos(0, 0)
        self.text_leftlim.setText(text="-%.1f kHz"%(self.bw_hz/2000./self.fft_ratio))
        #self.text_rightlim.setPos(self.bw_hz*1000, 0)
        self.text_rightlim.setText(text="+%.1f kHz"%(self.bw_hz/2000./self.fft_ratio))

if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    radio = TS180S()
    print(radio.name)

    try:
        pan = PanAdapter(sdr, radio )
    except:
        print("Couldn't create the PanAdapter device\n")
        raise

    t = QtCore.QTimer()
    t.timeout.connect(rtl.update_mode)
    t.timeout.connect(rtl.read)
    t.start(10) # max theoretical refresh rate 100 fps

    app.exec_()
    pan.close()
