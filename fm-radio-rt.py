from rtlsdr import RtlSdrAio  
import numpy as np  
import scipy.signal as signal
import pyaudio
import struct
import time
import math
import asyncio
import cProfile, pstats, io
import struct


pr = cProfile.Profile()
#pr.enable()

class Radio:
	def __init__(self, station = int(100.3e6) ):
		# The station parameters
		self.f_station = int(100.3e6)   			# The radio station frequency
		self.f_offset = 250000						# Offset to capture at         
		self.fc = self.f_station - self.f_offset 	# Capture center frequency  

		# Sampling parameters
		self.Fs = int(1140000)     			# Sample rate (different from the audio sample rate)
		self.blocksize = int(128*1024) 		# Samples to capture per block
		self.n_blocks = 10000

		# An FM broadcast signal has  a bandwidth of 200 kHz
		self.f_bw = 200000  
		self.n_taps = 64 

		# Calculate the sampling rate after first decimation. It is not necessary
		# to have such a high sampling rate for an audio signal
		self.dec_rate = int(self.Fs / self.f_bw)  
		self.Fs_y = self.Fs/self.dec_rate  

		# Find a decimation rate to achieve audio sampling rate between 44-48 kHz. This
		# is used in the second decimation.
		self.audio_freq = 48000.0  
		self.dec_audio = int(self.Fs_y/self.audio_freq)  
		self.Fs_audio = int(self.Fs_y / self.dec_audio)


		# Configure software defined radio
		self.sdr = RtlSdrAio()
		self.sdr.sample_rate = self.Fs      # Hz  
		self.sdr.center_freq = self.fc      # Hz  
		self.sdr.gain = 'auto'  

		# Downsample filter
		self.downsample_coefficents = self.downsample_filter()

		# Deemphasis filter
		self.deemphasis_coefficents = self.deemphasis_filter()

		# Block angle difference
		self.block_angle = 0
		self.block_angle_increment =  (float(self.blocksize*self.f_offset)/self.Fs \
			- np.floor(self.blocksize*self.f_offset/self.Fs)) * 2.0 * np.pi

		# Open audio stream
		self.samp_width = 2
		self.channels = 1		# For now we'll focus on getting one channel to work
		self.p = pyaudio.PyAudio()
		self.stream = self.p.open(format = self.p.get_format_from_width(self.samp_width),
		                 channels = self.channels,
		                 rate = self.Fs_audio,
		                 input = False,
		                 output = True,
		                 frames_per_buffer = 7000)

	def deemphasis_filter(self):
		# The de-emphasis filter
		d = self.Fs_y * 75e-6   # Calculate the # of samples to hit the -3dB point  
		x = np.exp(-1/d)   # Calculate the decay between each sample  
		b = [1-x]          # Create the filter coefficients  
		a = [1,-x]
		zi = signal.lfilter_zi(b, a)
		return [b, a, zi]

	def downsample_filter(self):
		# Use Remez algorithm to design filter coefficients used for downsampling in frequency.
		b = signal.remez(self.n_taps, [0, self.f_bw, self.f_bw+(self.Fs/2-self.f_bw)/4, self.Fs/2], [1,0], Hz=self.Fs) 				
		a = [1.0]
		zi = signal.lfilter_zi(b, a)
		return [b, a, zi]
           
	def get_radio_samples(self):
		assert type(self.sdr) is RtlSdr
		samples = self.sdr.read_samples(self.blocksize)
		return samples

	async def process_to_audio(self, samples, downsample_coefficents, deemphasis_coefficents, block_angle):
		global elapsed_decimate1, elapsed_convert, elapsed_down1, elapsed_shift, elapsed_demodulate, elapsed_deemphasis2, elapsed_string
		# Convert samples to a numpy array
		#x1 = np.array(samples).astype("complex64")
		x1 = samples
		# Now, just multiply x1 and the digital complex exponential
		fc1 = np.exp((-1.0j*(2.0*np.pi* self.f_offset/self.Fs)*\
			np.arange(self.blocksize)) + 1.0j*block_angle)
		x2 = x1 * fc1  	
		block_angle += self.block_angle_increment


		# Downsample the signal
		(b, a, zi_downsample) = downsample_coefficents
		#x3, zi_downsample = signal.lfilter(b, a, x2, zi = zi_downsample)

		# Decimate the signal
		#x4 = x3[0::self.dec_rate] 
		x4 = signal.decimate(x2, self.dec_rate, ftype = 'fir', zero_phase = True)  

		# Polar discriminator
		y5 = x4[1:] * np.conj(x4[:-1])  
		x5 = np.angle(y5)  

		# The deemphasis filter
		(b, a, zi_deemphasis) = deemphasis_coefficents
		x6, zi_deemphasis = signal.lfilter(b, a, x5, zi = zi_deemphasis)  

		# Decimate to audio
		x7 = signal.decimate(x6, self.dec_audio, ftype = 'fir', zero_phase = True)  

		# Scale audio to adjust volume
		x7 *= int(10000 / np.max(np.abs(x7))) 

		# Clip to avoid overflow
		x7 = self.clip(self.samp_width, x7)

		return (x7, zi_downsample, zi_deemphasis, block_angle)

	async def play_to_speaker(self, audio_samples):
		# Send bytes to the buffer of the pyaudio object
		assert self.samp_width == 2 # Otherwise 'h' is not applicable
		output_string = struct.pack('h' * len(audio_samples), *audio_samples)
		self.stream.write(output_string)
		return

	def clip(self, width, array):
		# Use to prevent overflow when converting to bytes
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
		# If in synchronous mode, use the method to play one sample block.
		assert type(self.sdr) is RtlSdr
		samples = self.get_radio_samples()
		output_string = self.process_to_audio(samples)
		self.play_to_speaker(output_string)

	def play(self):
		# Play blocks in a stream using the asynchronous mode
		assert type(self.sdr) is  RtlSdrAio
		loop = asyncio.get_event_loop()
		loop.run_until_complete(self.stream_samples())

	async def stream_samples(self):
		assert type(self.sdr) is RtlSdrAio
		# Stream and process the required number of blocks to audio, then send the 
		# the output string to the output speaker
		blocks_so_far = 0
		async for samples in self.sdr.stream(num_samples_or_bytes = self.blocksize, format = 'samples'):
			# Process to audio and update the residues
			audio_samples, self.downsample_coefficents[2], self.deemphasis_coefficents[2], self.block_angle = \
				await self.process_to_audio(samples, self.downsample_coefficents,
				 self.deemphasis_coefficents, self.block_angle)
			# Play to speaker
			await self.play_to_speaker(audio_samples)
			# Done with this block
			blocks_so_far += 1
			# We have streamed all of the block
			if blocks_so_far >= self.n_blocks:
				break	
		await self.sdr.stop()

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
radio.close()

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())