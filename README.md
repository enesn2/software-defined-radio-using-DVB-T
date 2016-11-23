# Software defined radio
To setup the necessary packages on your computer run
```
bash fm-radio-setup.sh  
```
To capture a recording run
```
python fm-radio.py
```
This will produce the file wbfm-mono.raw containing the audio. To run the file use
```
aplay wbfm-mono.raw -r 45600 -f S16_LE -t raw -c 1  
```
You might need to install the sox library first using 
```
sudo apt-get install sox
```