from __future__ import division, print_function
from rtlsdr import RtlSdr  
import numpy as np  
import scipy.signal as signal
import pyaudio
import struct
import time
import cProfile, pstats, io
from threading import Thread, Lock, Condition
from Queue import Queue


pr = cProfile.Profile()

# This queue stores unprocessed blocks of radio samples
queue = []
condition = Condition()
lock = Lock()

# These store the delays for various filters in radio processing
delays = []
block_angle = 0
#pr.enable()

class SDR(Thread):
	'''
	This is a producer class that samples radio signal blocks and places them
	in a global queue
	'''
	# This queue stores unprocessed blocks of radio samples
	queue = []
	condition = Condition()
	lock = Lock()

	# These store the delays for various filters in radio processing
	delays = []
	block_angle = 0
	def __init__(self, startFc, Fs, blocksize, sample_width):
		super(SDR, self).__init__()

		self.blocksize = blocksize
		self.sample_width = sample_width
		self.sdr = RtlSdr()
		self.sdr.sample_rate = Fs      # Hz  
		self.sdr.center_freq = startFc     # Hz  
		self.sdr.gain = 'auto' 

	def run(self):
		global queue
		t = time.time()
		while True:
			
			samples = self.sdr.read_bytes(num_bytes = 2*self.blocksize)
			condition.acquire()
			print('Gathered a block at', time.time()-t)	
			queue.append(samples)
			condition.notify()
			condition.release()
		self.sdr.stop()

class RadioSamplesProcessor(Thread):
	'''
	A consumer thread. This thread processes the radio samples into audio and passes
	them to the PyAudio object to be played by the audio output device.
	'''

	def __init__(self, blocksize, Fs, sample_width, f_offset):
		super(RadioSamplesProcessor, self).__init__()

		# Radio sampling rate
		self.Fs = Fs

		self.sample_width = sample_width
		self.blocksize = blocksize
		self.f_offset = f_offset

		# An FM broadcast signal has  a bandwidth of 200 kHz
		self.f_bw = 200000 

		# Calculate the sampling rate for first decimation. This is done to focus 
		# on the radio frequencies
		self.dec_rate = int(self.Fs / self.f_bw)  
		self.Fs_y = self.Fs/self.dec_rate   

		# Find a decimation rate to achieve audio sampling rate for audio between 44-48 kHz. This
		# is used in the second decimation.
		self.audio_freq = 48000.0  
		self.dec_audio = int(self.Fs_y/self.audio_freq)  
		self.Fs_audio = int(self.Fs_y / self.dec_audio)

		# Block angle difference
		self.block_angle_increment =  (float(self.blocksize*self.f_offset)/self.Fs \
			- np.floor(self.blocksize*self.f_offset/self.Fs)) * 2.0 * np.pi

		# Deemphasis filter coefficients and initial delay
		self.deemphasis_coefficents = self.deemphasis_filter()
		global delays
		delays.append(self.deemphasis_coefficents[2])

		# Open audio stream
		self.channels = 1		# For now we'll focus on getting one channel to work
		self.p = pyaudio.PyAudio()
		self.stream = self.p.open(format = self.p.get_format_from_width(self.sample_width),
		                 channels = self.channels,
		                 rate = self.Fs_audio,
		                 input = False,
		                 output = True,
		                 frames_per_buffer = 7000)

	def deemphasis_filter(self):
		'''	Defines the deemphasis filter used in FM demodulation  
		'''

		d = self.Fs_y * 75e-6   # Calculate the # of samples to hit the -3dB point  
		x = np.exp(-1/d)   # Calculate the decay between each sample  
		b = [1-x]          # Create the filter coefficients  
		a = [1,-x]
		zi = signal.lfilter_zi(b, a)
		return [b, a, zi]

	def process_to_audio(self, samples, deemphasis_delay, block_angle):
		''' 
		Processes binary samples from the sdr into audio samples to be played by the
		audio device.
		'''

		# Convert to a numpy array of complex IQ samples
		iq = np.empty(len(samples)//2, 'complex')
		iq.real, iq.imag = samples[::2], samples[1::2]
		iq /= (255/2)
		iq -= (1 + 1j)
		x1 = iq

		# Multiply x1 and the digital complex exponential to shift back from the offset frequency
		fc1 = np.exp((-1.0j*(2.0*np.pi* self.f_offset/self.Fs)*\
			np.arange(self.blocksize)) + 1.0j*block_angle)
		x2 = x1 * fc1  	
		block_angle += self.block_angle_increment

		# Decimate the signal to focus on the radio frequencies of interest
		x3 = signal.decimate(x2, self.dec_rate, ftype = 'fir', zero_phase = True)  

		# Demodulate the IQ samples
		y4 = x3[1:] * np.conj(x3[:-1])  
		x4 = np.angle(y4)  

		# The deemphasis filter
		(b, a, zi) = self.deemphasis_coefficents
		zi_deemphasis = deemphasis_delay
		x5, zi_deemphasis = signal.lfilter(b, a, x4, zi = zi_deemphasis)  

		# Decimate to audio
		x6 = signal.decimate(x5, self.dec_audio, ftype = 'fir', zero_phase = True)  

		# Scale audio to adjust volume
		x6 *= int(10000 / np.max(np.abs(x6))) 

		# Clip to avoid overflow
		x6 = np.clip(x6, (-2**(self.sample_width*8-1)), (2**(self.sample_width*8-1) - 1))
		x6 = x6.astype(int)

		return (x6, zi_deemphasis, block_angle)

	def play_to_speaker(self, audio_samples):
		''' Send bytes to the buffer of the pyaudio object to be played by the audio device
		'''
		assert self.sample_width == 2 # Otherwise 'h' is not applicable
		output_string = struct.pack('h' * len(audio_samples), *audio_samples)
		self.stream.write(output_string)
		return

	def run(self):
		''' This method is called by Thread.start()
		'''
		global queue, block_angle, delays
		while True:
			condition.acquire()
			print('Processing a block at:', time.time())
			if not len(queue):
				#print("Nothing in queue, consumer is waiting")
				condition.wait()
				#print("Producer added something to queue and notified the consumer")
			samples = queue.pop(0)
			deemphasis_delay = delays[0]
			condition.release() 

			audio_samples, deemphasis_delay, block_angle_current = self.process_to_audio(
				samples, deemphasis_delay, block_angle)

			self.play_to_speaker(audio_samples)

			
			delays[0] = deemphasis_delay
			block_angle = block_angle_current
			
		print('I dont know what an infinite loop is. ')      

class Radio:
	''' This is the main class of the program
	'''
	def __init__(self, station = int(100.3e6) ):
		# The station parameters
		self.f_station = int(100.3e6)   			# The radio station frequency
		self.f_offset = 250000						# Offset to capture at         
		self.fc = self.f_station - self.f_offset 	# Capture center frequency  

		self.sample_width = 2

		# Sampling parameters for the rtl-sdr
		self.Fs = 1140000     			    # Sample rate (different from the audio sample rate)
		self.blocksize = 128*1024 		    # Samples to capture per bloc


		# Configure software defined radio thread/class
		self.sdr = SDR(self.fc, self.Fs, self.blocksize, self.sample_width) 

		# Configure radio samples processors thread/class
		self.processor = RadioSamplesProcessor(self.blocksize, self.Fs, self.sample_width, self.f_offset)

           
	def get_radio_samples(self):
		''' For sampling of a single block in synchronous mode
		'''
		assert type(self.sdr) is RtlSdr
		samples = self.sdr.read_samples(self.blocksize)
		return samples


	def clip(self, width, array):
		''' 
		Use to prevent overflow when converting to bytes. Currently unused 
		so it might get deleted.
		'''
		python_array = []		# The expected input is a numpy array
		for i in range(len(array)):
			if array[i] > (2**(8*width-1)-1):
				python_array.append(2**(8*width-1)-1)
			elif array[i] < (-(2**(8*width-1))):
				python_array.append(-(2**(8*width-1)))
			else:
				python_array.append(int(array[i]))
		return python_array

	def play_a_block(self):
		'''	If in synchronous mode, use the method to play one sample block.
		'''
		assert type(self.sdr) is RtlSdr
		samples = self.get_radio_samples()
		output_string = self.process_to_audio(samples)
		self.play_to_speaker(output_string)

	def play(self):
		''' 
		Start the `sdr` thread that samples the air and the `processor` thread that
		processes the samples to audio.
		'''
		self.sdr.start()
		self.processor.start()

	def close(self):
		# Clean up the SDR device
		self.sdr.close()  
		del(self.sdr)

		# Close the audio interface
		self.stream.stop_stream()	
		self.stream.close()		
		self.p.terminate()


radio = Radio()
radio.play()
#radio.close()

'''
pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
'''
