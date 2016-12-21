#!/bin/bash

sudo apt-get update  # update list of available software  
sudo apt-get -y install git cmake libusb-1.0-0-dev python3 python3-pip python3-dev
sudo apt-get -y install python3-scipy python3-numpy

cd

git clone https://github.com/steve-m/librtlsdr  
cd librtlsdr  
mkdir build  
cd build  
cmake ../  
make  
sudo make install  
ldconfig  

cd

git clone https://github.com/roger-/pyrtlsdr.git
cd pyrtlsdr
sudo python3 setup.py install

sudo pip3 install pyaudio  


