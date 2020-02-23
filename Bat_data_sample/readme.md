# Real-life Bats recordings

For the task of localization of bats in 3D space, based only on their sound waves, using Deep Learning tools, the SELDnet has been tested on bats sounds recorded in multichannel audio files. 

In order to use the pre-trained SELDnet on the bat sound files and their corresponding location labels, the bat experiment documentation files had been modified to the same formats of those from the datasets of the original paper. 
The 46 multichannel audio files of the bats recording (recorded during trials) has been modified into a 4 multichannel audio files, while every channel representing the microphones on each room wall. Furthermore, the bat locations labels at every time (3D Cartesian coordinates x, y, z coordinates) has been modified as well and normalized according to the room size.
