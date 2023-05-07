# GAN-heightmap-generation

This repository is a 1.5 year long university project. Its main goal is to create a Generative Adversarial Network to create heightmaps to be used in Unreal Engine. Unreal Engine code will generate a terrain from the provided heightmap with use of common tools used for procedural terrain generation.

GAN will be trained on data provided by [NASA Visible Earth](https://visibleearth.nasa.gov/).

https://github.com/mgraves236/GAN-heightmap-generation/blob/master/map.png


![Wolrd heightmap](https://github.com/mgraves236/GAN-heightmap-generation/blob/master/map.png "Wolrd heightmap")

## Data preprocessing

Files `crop.py` and `remove.py` are responsible for data preprocessing. The first one creates a grid made by 256x256 pixels elements and crops the world map image according to it. Elements that are almost uniform in color are redundant to the training process, thus `remove.py` removes them.
