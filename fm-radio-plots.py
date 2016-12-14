from rtlsdr import RtlSdr  
import numpy as np  
import scipy.signal as signal
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import matplotlib.pyplot as plt


sdr = RtlSdr()

F_station = int(100.3e6)   # Rutgers Radio  
F_offset = 250000         # Offset to capture at  
# We capture at an offset to avoid DC spike
Fc = F_station - F_offset # Capture center frequency  
Fs = int(1140000)         # Sample rate  
N = int(8192000)            # Samples to capture  

# configure device
sdr.sample_rate = Fs      # Hz  
sdr.center_freq = Fc      # Hz  
sdr.gain = 'auto'

# Read samples
samples = sdr.read_samples(N)

# Clean up the SDR device
sdr.close()  
del(sdr)

# Convert samples to a numpy array
x1 = np.array(samples).astype("complex64")

path = "/usr/share/fonts/truetype/roboto/hinted/Roboto-Light.ttf"
prop = font_manager.FontProperties(fname=path)
mpl.rcParams['font.family'] = prop.get_name()

plt.specgram(x1, NFFT=2048, Fs=Fs)  
plt.title("Input signal",)  
plt.xlabel("Time (s)")  
plt.ylabel("Frequency (Hz)") 
plt.ylim(-Fs/2, Fs/2)  
plt.xlim(0,len(x1)/Fs) 
plt.ticklabel_format(style='plain', axis='y' )   
plt.savefig("./plots/x1_spec.png", bbox_inches='tight', pad_inches=0.5)  
#plt.show()

fc1 = np.exp(-1.0j*2.0*np.pi* F_offset/Fs*np.arange(len(x1)))  
# Now, just multiply x1 and the digital complex expontential
x2 = x1 * fc1  

plt.specgram(x2, NFFT=2048, Fs=Fs)  
plt.title("Downshifted input signal")  
plt.xlabel("Time (s)")  
plt.ylabel("Frequency (Hz)")  
plt.ylim(-Fs/2, Fs/2)  
plt.xlim(0,len(x2)/Fs)  
plt.ticklabel_format(style='plain', axis='y' )  
plt.savefig("./plots/x2_spec.png", bbox_inches='tight', pad_inches=0.5)  
plt.close()  
#plt.show()  

# An FM broadcast signal has  a bandwidth of 200 kHz
f_bw = 200000  
dec_rate = int(Fs / f_bw)  
x4 = signal.decimate(x2, dec_rate)  
# Calculate the new sampling rate
Fs_y = Fs/dec_rate  

plt.specgram(x4, NFFT=2048, Fs=Fs_y)  
plt.title("Decimated input signal")  
plt.xlabel("Time (s)")  
plt.ylabel("Frequency (Hz)")  
plt.ylim(-Fs_y/2, Fs_y/2)  
plt.xlim(0,len(x4)/Fs_y)  
plt.ticklabel_format(style='plain', axis='y' )  
plt.savefig("./plots/x4_spec.png", bbox_inches='tight', pad_inches=0.5)  
plt.close()  
#plt.show() 

### Polar discriminator
y5 = x4[1:] * np.conj(x4[:-1])  
x5 = np.angle(y5)   

# The de-emphasis filter
# Given a signal 'x5' (in a numpy array) with sampling rate Fs_y
d = Fs_y * 75e-6   # Calculate the # of samples to hit the -3dB point  
x = np.exp(-1/d)   # Calculate the decay between each sample  
b = [1-x]          # Create the filter coefficients  
a = [1,-x]  
x6 = signal.lfilter(b,a,x5)  

w, h = signal.freqz(b, a)
plt.plot(w, 20 * np.log10(abs(h)), '#ffd200',linewidth = 3)
plt.xlim(0, 3.142) 
plt.ylabel('Amplitude (dB)')
plt.xlabel('Frequency (rad/sample)')
plt.title("De-emphasis filter") 
plt.savefig("./plots/deemphasis.png", bbox_inches='tight', pad_inches=0.5)  
plt.close()  
#plt.show()

# An FM broadcast signal has  a bandwidth of 200 kHz
f_bw = 200000  
n_taps = 64  
# Use Remez algorithm to design filter coefficients
b = signal.remez(n_taps, [0, f_bw, f_bw+(Fs/2-f_bw)/4, Fs/2], [1,0], Hz=Fs)  

w, h = signal.freqz(b, 1)
plt.plot(w, 20 * np.log10(abs(h)), '#91acff', linewidth = 3)
plt.xlim(0, 3.142) 
plt.ylabel('Amplitude (dB)')
plt.title("Decimate filter") 
plt.xlabel('Frequency (rad/sample)')
plt.savefig("./plots/decimate.png", bbox_inches='tight', pad_inches=0.5)  
plt.close()  
#plt.show()
