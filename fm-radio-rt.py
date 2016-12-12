from __future__ import division, print_function
from rtlsdr import RtlSdrAio  
import numpy as np  
import scipy.signal as signal
import pyaudio
import struct
import time
import asyncio
import cProfile, pstats, io
import threading
from threading import Thread, Lock, Condition
from queue import Queue


pr = cProfile.Profile()

# This queue stores unprocessed blocks of radio samples
queue =  Queue(10)

# This queue stores audio samples
audio_queue = Queue(10)

# This queue stores filtered audio samples
filtered_audio_queue = Queue(10)

# These threading modules are used to manage concurrency
condition = Condition()
lock = Lock()

# These store delays for various filters in radio processing
delays = []
block_angle = 0

#pr.enable()

class SDR:
	'''
	This is a producer class that samples radio signal blocks and places them
	in a global queue
	'''

	def __init__(self, startFc, Fs, blocksize, sample_width):
		#super(SDR, self).__init__()

		self.blocksize = blocksize
		self.sample_width = sample_width
		self.sdr = RtlSdrAio()
		self.sdr.sample_rate = Fs      # Hz  
		self.sdr.center_freq = startFc     # Hz  
		self.sdr.gain = 'auto'

	def run(self, n_blocks):
		loop = asyncio.get_event_loop()
		asyncio.set_event_loop(loop) 
		loop.run_until_complete(self.stream_samples(n_blocks))

	async def stream_samples(self, n_blocks):
		global queue
		blocks_so_far = 0
		async for samples in self.sdr.stream(num_samples_or_bytes = 2*self.blocksize, format = 'bytes'):
			#print('Gathered a block')
			queue.put(samples)
			blocks_so_far+=1
			if blocks_so_far>=n_blocks:
				break
		await self.sdr.stop()

class AudioSamplesProcessor(Thread):
	''' This thread plays the audio samples from the audio queue
	'''

	def __init__(self, sample_width, audioFs, audio_buffer_length):
		super(AudioSamplesProcessor, self).__init__()

		self.sample_width = sample_width
		self.p = pyaudio.PyAudio()
		self.audio_buffer_length = audio_buffer_length
		self.stream = self.p.open(format = self.p.get_format_from_width(self.sample_width),
		             channels = 1,
		             rate = audioFs,
		             input = False,
		             output = True,
		             frames_per_buffer = self.audio_buffer_length)	
		self.previous_block = []

	def run(self):
		global audio_queue
		while True:
			audio_samples = audio_queue.get()
			audio_queue.task_done()

			# Used to stop the thread
			if audio_samples is None:
				break

			# Fill the previous block with zeros on the first run
			if not(len(self.previous_block)):
				self.previous_block = np.zeros((len(audio_samples),),)

			audio_samples = self.audio_filter(audio_samples, self.previous_block, 'robo')
			# Write to bytes
			output_string = struct.pack('h' * len(audio_samples), *audio_samples)

			# Write stream
			self.stream.write(output_string)

			# Keep the previous block
			self.previous_block = audio_samples
			
			#print('Task done')
		self.stream.stop_stream()

	def audio_filter(self,audio_block, previous_block, type = 'nofilter'):
		if type == 'robot':
			audio_block = self.robot(audio_block, previous_block)
		audio_block = np.clip(audio_block, (-2**(self.sample_width*8-1)), (2**(self.sample_width*8-1) - 1))
		audio_block = audio_block.astype(int)
		return audio_block
		


	def stft(self, previous_block, current_block, Nfft, R):
		'''
		Short-time Fourier transform performed on two consecutive audio-blocks
		This function does sfft to the current block, the second half of the previous block + the first
		half of the current block and the current block This is because we have chosen a
		a 50% overlap between the blocks in sfft.
		'''

		# Convert the inputs into numpy arrays
		overlapping_block = np.concatenate((previous_block[R/2:R], current_block[0:R/2]))

		# Create the window
		n = np.array(range(1,R+1))+0.5
		window = np.cos(np.pi*n/R-np.pi/2)

		# Do windowed fft on the previous block
		X1 = previous_block*window
		X1 = np.fft.fft(X1,Nfft)

		# Do windowed fft on the overlapping block
		X2 = overlapping_block*window
		assert len(X2) == R
		X2 = np.fft.fft(X2,Nfft)

		# Do windowed fft on the current block
		X3 = current_block*window
		X3 = np.fft.fft(X3,Nfft)
		return X1, X2, X3

	def istft(self, first_block, second_block, third_block, R):
		''' Inverse short-time Fourier transform on the three blocks returned by sfft()
		'''

		# Create the window
		n = np.array(range(1,R+1))+0.5
		window = np.cos(np.pi*n/R-np.pi/2)


		# Find the issft
		Y1 = np.fft.ifft(first_block, R)
		Y2 = np.fft.ifft(second_block, R)
		Y3 = np.fft.ifft(third_block, R)
		y1 = Y1*window
		y2 = Y2*window
		y3 = Y3*window
		return np.real(y1), np.real(y2), np.real(y3)

	def robot(self, current_audio_block, previous_block):

		# Number of bins in sfft
		Nfft = 512
		R = int(Nfft)

		# We only need a sample of the previous block
		previous_block = previous_block[-1-R+1:]

		# For the filter to work we need to separate the current_block into a number of 
		# smaller blocks
		n_blocks = int(len(current_audio_block)/R) + 2

		# Pad the end with zeros which will be removed, if necessary 
		extra_samples = R - len(current_audio_block)%R

		# We also add `R` more samples because it is necessary for the robot transformation to work
		current_audio_block = np.concatenate((current_audio_block, np.zeros((extra_samples+R,))))

		processed_block = np.array([])
		
		for i in range(n_blocks):
			current_block = current_audio_block[i*R:i*R+R]


			# Since we are doing a 50% sfft, two consecutive blocks,
			# are needed to do the robot processing
			# We pass the old input block and the new input block
			# to the sfft
			first_block_ft, second_block_ft, third_block_ft = self.stft(previous_block, current_block, Nfft, R)

			# Set phase to zero in STFT-domain
			first_block_ft = np.absolute(first_block_ft)
			second_block_ft = np.absolute(second_block_ft)
			third_block_ft = np.absolute(third_block_ft)

			# Synthesize the new signal
			first_block, second_block, third_block = self.istft(first_block_ft, second_block_ft, third_block_ft, R)

			# Take the second half of the first block and pad the rest with zeros
			first_block = np.concatenate((first_block[R/2:R], np.zeros((R/2),)),)

			# Take the first half of the third block and pad the reset with zeros
			third_block = np.concatenate((np.zeros((R/2),), third_block[0:R/2] ),)

			# Obtain the output from the overlapping regions of the three blocks
			robot = np.add(first_block, second_block)
			robot = np.add(robot, third_block)

			# Scale the amplitude
			robot *= int(32000 / np.max(np.abs(robot))) 

			# Add to the cummulative processed block
			processed_block = np.concatenate((processed_block, robot))

			# Update the previous block
			previous_block = current_block

		# Remove the zeros we added earlier
		processed_block = processed_block[0:len(processed_block) - R - extra_samples]

		return processed_block

	def close():
		stream.close()
		p.terminate()
	

class RadioSamplesProcessor(Thread):
	'''
	A consumer thread. This thread processes the radio samples into audio and passes
	them to the PyAudio object to be played by the audio output device.
	'''

	def __init__(self, blocksize, Fs, sample_width, f_offset, audioFs, ID):
		super(RadioSamplesProcessor, self).__init__()

		# Radio sampling rate
		self.Fs = Fs

		self.sample_width = sample_width
		self.blocksize = blocksize
		self.f_offset = f_offset

		self.ID = ID

		# An FM broadcast signal has  a bandwidth of 200 kHz
		self.f_bw = 200000 

		# Calculate the sampling rate for first decimation. This is done to focus 
		# on the radio frequencies
		self.dec_rate = int(self.Fs / self.f_bw)  
		self.Fs_y = self.Fs/self.dec_rate   

		# Find a decimation rate to achieve audio sampling rate for audio between 44-48 kHz. This
		# is used in the second decimation.
		self.audio_freq = audioFs 
		self.dec_audio = int(self.Fs_y / self.audio_freq)  
		self.Fs_audio = int(self.Fs_y / self.dec_audio)

		# Block angle difference
		self.block_angle_increment =  (float(self.blocksize*self.f_offset)/self.Fs \
			- np.floor(self.blocksize*self.f_offset/self.Fs)) * 2.0 * np.pi

		# Deemphasis filter coefficients and initial delay
		self.deemphasis_coefficents = self.deemphasis_filter()
		global delays
		delays.append(self.deemphasis_coefficents[2])


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
		global delays
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
		x3 = signal.decimate(x2, self.dec_rate, n = 22, ftype = 'fir', zero_phase = True)  

		# Demodulate the IQ samples
		y4 = x3[1:] * np.conj(x3[:-1])  
		x4 = np.angle(y4)  

		# The deemphasis filter
		(b, a, zi) = self.deemphasis_coefficents
		zi_deemphasis = deemphasis_delay
		delays[0] = zi_deemphasis
		x5, zi_deemphasis = signal.lfilter(b, a, x4, zi = zi_deemphasis)  

		# Decimate to audio
		x6 = signal.decimate(x5, self.dec_audio, n = 22, ftype = 'fir', zero_phase = True)  

		# Scale audio to adjust volume
		x6 *= int(32000 / np.max(np.abs(x6))) 

		# Clip to avoid overflow


		return (x6, zi_deemphasis, block_angle)

	def run(self):
		''' This method is called by Thread().start()
		'''
		global queue, block_angle, delays
		global audio_queue
		while True:
			#print('Thread',  threading.current_thread(), 'working')
			samples = queue.get()
			queue.task_done()
			if samples is None:
				audio_queue.put(None)
				break
			
			deemphasis_delay = delays[0]
			delays[0] = self.deemphasis_coefficents[2]
			now = time.time()
			audio_samples, deemphasis_delay, block_angle_current = self.process_to_audio(
				samples, deemphasis_delay, block_angle)
			#delays[0] = deemphasis_delay
			#block_angle = block_angle_current
			#lock.acquire()
			
			audio_queue.put(audio_samples)
			#print('')
			#print('Thread',  threading.current_thread(), 'done')
			#lock.release()
			#print('Placed audio samples' )	
			
			
			#print('It took',time.time()-now,'seconds to process the audio samples.')
			#print('Samples are', float(len(audio_samples))/48000.00,'secondslong.')
			#print(len(audio_samples))
			  



class Radio:
	''' This is the main class of the program
	'''
	def __init__(self, station = int(100.3e6) ):
		# The station parameters
		self.f_station = int(100.3e6)   			# The radio station frequency
		self.f_offset = 250000						# Offset to capture at         
		self.fc = self.f_station - self.f_offset 	# Capture center frequency  

		self.sample_width = 2
		self.audio_buffer_length = 10500*3*2
		self.n_blocks = 60

		# Sampling parameters for the rtl-sdr
		self.Fs = 1140000     			    # Sample rate (different from the audio sample rate)
		self.blocksize = 128*1024*2*2 		    # Samples to capture per bloc
		self.audioFs = 40000

		# Configure software defined radio thread/class
		self.sdr1 = SDR(self.fc, self.Fs, self.blocksize, self.sample_width) 

		# Configure radio samples processors thread/class
		self.radio_processor1 = RadioSamplesProcessor(self.blocksize, self.Fs, self.sample_width, self.f_offset, self.audioFs, 1)
		self.radio_processor2 = RadioSamplesProcessor(self.blocksize, self.Fs, self.sample_width, self.f_offset, self.audioFs, 2)
		#self.radio_processor3 = RadioSamplesProcessor(self.blocksize, self.Fs, self.sample_width, self.f_offset, self.audioFs, 3)

		# Define audio processor
		self.audio_processor = AudioSamplesProcessor(self.sample_width, self.audioFs, self.audio_buffer_length)

		self.radio_threads = [self.radio_processor1, self.radio_processor2]
           
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

	def stop(self):
		# Stop all consumer threads
		for i in range(self.radio_threads):
			queue.put(None)
		for thread in self.all_threads:
			thread.join()

	def play(self):
		''' 
		Start the `sdr` thread that samples the air, the `radio_processor` thread that
		processes the samples to audio and the `radio_processor` that plays samples.
		'''
		for thread in self.radio_threads:
			thread.start()

		self.audio_processor.start()

		self.sdr1.run(self.n_blocks)
		self.stop()


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
