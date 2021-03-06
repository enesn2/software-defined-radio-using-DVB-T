# Description
This code utilizes USB DVB-T dongles based on the Realtek RTL2832U chipset to stream FM radio broadcasts in real time. In addition, the code has the capability to add audio effects to the original audio. Currently, two audio effects are implemented, "robotization" and "whisperization". The program was written for Python 3 and it cannot be ran in Python 2.

# Usage
Download the source files and install the necessary libraries by running: 
```
sudo bash setup.py
```
in the source folder. To play the broadcast of an FM radio station run:
```
python3 fm-radio-rt.py --station=<frequency in MHz>
```
This will play the broadcast for two minutes. To change the play time add the argument `--time=<number of seconds>`.

Applying the 'robot' audio effect to a broadcast:
```
python3 fm-radio-rt.py --station=<frequency in MHz> --audio-effect=robot
```
To apply the 'whisper' audio effect to a broadcast add `--audio-effect=whisper` instead.

If the audio stutter occurs while running, try decreasing the audio sampling rate to a lower one such as 24000 by adding `--audio-fs=24000`. Note that in the current version of the code other sampling rates may cause distortions in the audio.

# Dependencies
+ Windows/Linux/OSX
+ Python 3.3+
+ pyrtlsdr
+ SciPy
+ NumPy
+ PyAudio

The code has only been tested on Linux thus far, but in theory it should work across all operating systems if the dependencies are present. The `setup.sh` script only works with Linux.