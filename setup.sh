#!/bin/bash

apt-get update  # update list of available software  
apt-get -y install git cmake libusb-1.0-0-dev python python-pip python-dev
apt-get -y install python-scipy python-numpy python-matplotlib


# Remove other RTL-SDR driver, if it is loaded
modprobe -r dvb_usb_rtl28xxu

git clone https://github.com/steve-m/librtlsdr  
cd librtlsdr  
mkdir build  
cd build  
cmake ../  
make  
make install  
ldconfig  

cd

pip install pyrtlsdr  
pip install pyaudio  

wget https://raw.githubusercontent.com/keenerd/rtl-sdr-misc/master/heatmap/flatten.py


