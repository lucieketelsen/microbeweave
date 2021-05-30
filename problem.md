# Problem

How to work creatively within computational workflows? 

This project considered ways of 'contaminating' traditional hand weaving samples with microbial growth patterns.  

The project is composed of several 'problems':

* compile original dataset from existing images
* create images from microscopy captures
* derive image frames from time lapse microscopy video
* clean, resize and format images as datasets
* segment and transform microscopy frames \(data augmentation\) to expand dataset size
* increase resolution or map features of frame segments derived from microscopy
* Train a model to generate novel images based on inputs and outputs

Many of these problems can be solved with pre-existing computer vision or machine learning tools, including libraries and modules specifically created for dealing with microscopy captures. 

What is challenging in this case, however, is creating weave samples that are influenced by microscopy captures, such that a model is capable of generating novel, hybrid images. This is different to deep dream or style transfer techniques, which wouldn't be suited to this kind of creative problem. 

Initially, I thought transfer learning might be appropriate. I attempted 'image painting' using Tensorflow \(please see 'Preparatory Explorations' for an explanation of what was attempted here\). 

The final project output uses UGATIT - an unsupervised GAN for image to image translation. 

