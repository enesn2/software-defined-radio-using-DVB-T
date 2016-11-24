from rtlsdr import RtlSdrAio  
import numpy as np  
import scipy.signal as signal
import pyaudio
import struct
import time
import asyncio

elapsed_decimate1 = 0
elapsed_convert = 0
elapsed_read = 0
elapsed_down1 = 0
elapsed_shift = 0
elapsed_demodulate = 0
elapsed_deemphasis2 = 0
elapsed_string = 0


class Radio:
	def __init__(self, station = int(100.3e6) ):
		# The station parameters
		self.f_station = int(100.3e6)   			# The radio station frequency
		self.f_offset = 250000						# Offset to capture at         
		self.fc = self.f_station - self.f_offset 	# Capture center frequency  

		# Sampling parameters
		self.Fs = int(1140000)     			# Sample rate (different from the audio sample rate)
		self.blocksize = int(16384) 		# Samples to capture per block
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

		# To mix the data down, generate a digital complex exponential 
		# (with the same length as x1) with phase -F_offset/Fs
		self.fc1 = np.exp(-1.0j*2.0*np.pi* self.f_offset/self.Fs*np.arange((self.blocksize)))

		# Configure software defined radio
		self.sdr = RtlSdrAio()
		self.sdr.sample_rate = self.Fs      # Hz  
		self.sdr.center_freq = self.fc      # Hz  
		self.sdr.gain = 'auto'  

		# Open audio stream
		self.samp_width = 2
		self.channels = 1		# For now we'll focus on getting one channel to work
		self.p = pyaudio.PyAudio()
		self.stream = self.p.open(format = self.p.get_format_from_width(self.samp_width),
		                 channels = self.channels,
		                 rate = self.Fs_audio,
		                 input = False,
		                 output = True)

	def deemphasis_filter(self):
		# The de-emphasis filter
		d = self.Fs_y * 75e-6   # Calculate the # of samples to hit the -3dB point  
		x = np.exp(-1/d)   # Calculate the decay between each sample  
		b = [1-x]          # Create the filter coefficients  
		a = [1,-x] 
		return (b, a)

	def downsample_filter(self):
		# Use Remez algorithm to design filter coefficients used for downscaling in frequency.
		b = signal.remez(self.n_taps, [0, self.f_bw, self.f_bw+(self.Fs/2-self.f_bw)/4, self.Fs/2], [1,0], Hz=self.Fs) 				
		return b
           
	def get_radio_samples(self):
		assert type(self.sdr) is RtlSdr
		global elapsed_read
		t = time.time()
		samples = self.sdr.read_samples(self.blocksize)
		elapsed_read = time.time() - t
		return samples

	async def process_to_audio(self, samples):
		global elapsed_decimate1, elapsed_convert, elapsed_down1, elapsed_shift, elapsed_demodulate, elapsed_deemphasis2, elapsed_string
		# Convert samples to a numpy array
		t = time.time()
		x1 = np.array(samples).astype("complex64")
		elapsed_convert = time.time() - t

		# Now, just multiply x1 and the digital complex exponential
		t = time.time()
		x2 = x1 * self.fc1  	
		elapsed_shift = time.time() - t

		# Downsample the signal
		t = time.time()
		b = self.downsample_filter()
		x3 = signal.lfilter(b, 1.0, x2)
		elapsed_down1 = time.time() - t

		# Decimate the signal
		t = time.time()
		x4 = x3[0::self.dec_rate] 
		elapsed_decimate1 = time.time() - t

		# Polar discriminator
		t = time.time()
		y5 = x4[1:] * np.conj(x4[:-1])  
		x5 = np.angle(y5)  
		elapsed_demodulate = time.time() - t

		# The deemphasis filter
		t = time.time()
		b, a = self.deemphasis_filter()
		x6 = signal.lfilter(b,a,x5)  
		elapsed_deemphasis2 = time.time() - t

		# Decimate to audio
		t = time.time()
		x7 = signal.decimate(x6, self.dec_audio)  
		elapsed_decimate2 = time.time() - t

		# Scale audio to adjust volume
		x7 *= int(10000 / np.max(np.abs(x7))) 

		# Clip to avoid overflow
		x7 = self.clip(self.samp_width, x7)

		# Convert values to binary string
		t = time.time()
		assert self.samp_width == 2 # Otherwise 'h' is not applicable
		output_string = struct.pack('h' * len(x7), *x7)
		elapsed_string = time.time() - t
		return output_string

	async def play_to_speaker(self, output_string):
		# Send to the buffer of pyaudio object
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
			# Process to audio
			output_string = await self.process_to_audio(samples)
			# Play to speaker
			await self.play_to_speaker(output_string)
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







print("Read time:", str(elapsed_read))
print("Convert to array time:", str(elapsed_convert))
print("Shift frequency time:", str(elapsed_shift))
print("Downscale one time:", str(elapsed_down1))
print("Decimate one time:", str(elapsed_decimate1 ))
print("Demodulate time:", str(elapsed_demodulate))
print("Deemphasis filter time:", str(elapsed_deemphasis2))
print("Convert to string time:", str(elapsed_string))