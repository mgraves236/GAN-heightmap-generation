# GAN-heightmap-generation

This repository is a 1.5 year long university project. Its main goal is to create a Generative Adversarial Network to create heightmaps to be used in Unreal Engine. Unreal Engine code will generate a terrain from the provided heightmap with the use of common tools used for procedural terrain generation.

GAN will be trained on data provided by [NASA Visible Earth](https://visibleearth.nasa.gov/).

https://github.com/mgraves236/GAN-heightmap-generation/blob/master/map.png


![World heightmap](https://github.com/mgraves236/GAN-heightmap-generation/blob/master/map-lower.png "World heightmap")

## Data preprocessing

Files `crop.py` and `remove.py` are responsible for data preprocessing. The first one creates a grid made by 256x256 pixels elements and crops the world map image according to it. Elements that are almost uniform in color are redundant to the training process, thus `remove.py` removes them.
From 14,112 cropped images, only 5,088 are utilized in the training process.

## GAN implementation

The model is implemented in `train.py`. It consists of two simple sequential neural networks which act as a discriminator and a generator. 
The generator takes a noise vector as an input and modifies the data to achieve the results most similar to the original data. Its architecture is presented below:
![Generator architecture](https://github.com/mgraves236/GAN-heightmap-generation/blob/master/gen.png "Generator architecture")

The discriminator's task is to recognize the real examples and the generated ones. It was constructed as a simple 6-layers neural network:

![Discriminator architecture](https://github.com/mgraves236/GAN-heightmap-generation/blob/master/dis.png "Discriminator architecture")

The networks were trained for 35,000 epochs with a batch size of 256 with Adam.

## Postprocessing 

The output images from the generator required further processing by applying shape blur with a circular kernel. Examples of generated images are shown below:

![Result images](https://github.com/mgraves236/GAN-heightmap-generation/blob/master/res.png "Result images")

## Loading into Unreal Engine
Generated heightmaps were loaded into Unreal Engine. An automaterial was implemented using blueprints to automatically color the loaded terrain:

![Blueprint of automaterial](https://github.com/mgraves236/GAN-heightmap-generation/blob/master/ue.png "Blueprint of automaterial")

The final result of the project is the colored terrain generated by AI:

![Final result](https://github.com/mgraves236/GAN-heightmap-generation/blob/master/ue2.png "Final result")

