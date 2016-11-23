from rtlsdr import RtlSdr  
import numpy as np  
import scipy.signal as signal
import pyaudio


# The radio station frequency
F_station = int(100.3e6)   	# Rutgers Radio  
F_offset = 250000			# Offset to capture at         
Fc = F_station - F_offset 	# Capture center frequency  

# Sample properties
Fs = int(1140000)		# Sample rate (different from the audio sample rate)
T = int(8)				# Seconds to capture
N = int(8192000)  		# Samples to capture 

# Configure software defined radio
sdr = RtlSdr()
sdr.sample_rate = Fs      # Hz  
sdr.center_freq = Fc      # Hz  
sdr.gain = 'auto'          
           
# An FM broadcast signal has  a bandwidth of 200 kHz
f_bw = 200000  
n_taps = 64 

# Use Remez algorithm to design filter coefficients used for downscaling in frequency.
lpf = signal.remez(n_taps, [0, f_bw, f_bw+(Fs/2-f_bw)/4, Fs/2], [1,0], Hz=Fs) 

# Read samples
samples = sdr.read_samples(N)

# Convert samples to a numpy array
x1 = np.array(samples).astype("complex64")

# To mix the data down, generate a digital complex exponential 
# (with the same length as x1) with phase -F_offset/Fs
fc1 = np.exp(-1.0j*2.0*np.pi* F_offset/Fs*np.arange(len(x1)))

# Now, just multiply x1 and the digital complex exponential
x2 = x1 * fc1  

# Downsample the signal
x3 = signal.lfilter(lpf, 1.0, x2)

# Decimate the signal
dec_rate = int(Fs / f_bw)  
x4 = x3[0::dec_rate] 

# Calculate the new sampling rate
Fs_y = Fs/dec_rate  

# Polar discriminator
y5 = x4[1:] * np.conj(x4[:-1])  
x5 = np.angle(y5)  

# The de-emphasis filter
# Given a signal 'x5' (in a numpy array) with sampling rate Fs_y
d = Fs_y * 75e-6   # Calculate the # of samples to hit the -3dB point  
x = np.exp(-1/d)   # Calculate the decay between each sample  
b = [1-x]          # Create the filter coefficients  
a = [1,-x]  
x6 = signal.lfilter(b,a,x5)  

# Find a decimation rate to achieve audio sampling rate between 44-48 kHz
audio_freq = 44100.0  
dec_audio = int(Fs_y/audio_freq)  
Fs_audio = Fs_y / dec_audio

# Decimate to audio
x7 = signal.decimate(x6, dec_audio)  

# Scale audio to adjust volume
x7 *= 10000 / np.max(np.abs(x7)) 

# Save to file as 16-bit signed single-channel audio samples
x7.astype("int16").tofile("wbfm-mono.raw") 










# Clean up the SDR device
sdr.close()  
del(sdr)

