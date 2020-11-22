---
layout: post
published: true
title: Understanding Image style transfer using Convolutional Neural Networks
---

Main goal of this post is to explain Gatys et al (2016) work on Image style transfer using CNN’s in easier terms.

Code for generating all images in this notebook can be found at [github](https://github.com/raviteja-ganta/Neural-style-transfer-using-CNN)

### Contents:

1) What is Style transfer?

2) Content of an Image

3) Style of an Image

4) Understanding output of CNN’s

5) Cost function

6) Gram matrix

### 1) What is Style transfer?

First of all, what is style transfer between images? I will try to explain it with the example below

![F-1](https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/master/images/NS_fig1.png "Fig. 1: Style transfer: Target image looks like content image painted with style of style image") 
*Fig. 1: Style transfer: Target image looks like content image painted with style of style image*

We have content image which is a stretch of buildings across a river. We also have a style image which is a painting. Main idea behind style transfer is to transfer the ‘style’ of style image to the content image so that the target images looks like buildings and river painted in style of artwork(style image). We can clearly see that content is preserved but looks like buildings and water are painted.

To do this we need to extract content from content image, style from style image and combine these two to get our target image. But before that, lets understand what exactly content and style of an image are.

### 2) Content of an Image:
Content can be thought as objects and arrangements in an image. In our current case, content is literally content in the image with out taking in to account texture and color of pixels. So in our above examples content is just houses, water and grass irrespective of colors.

### 3) Style of an Image:
We can think of style as texture, colors of pixels. At same time it doesn’t care about actual arrangement and identity of different objects in that image.

Now that we have understanding of what content and style of image are, lets see how can we get them from the image.

But before that its important to understand what CNN’s are learning. It gives us clear idea when we talk about extracting style from image.

### 4) Understanding output of CNN’s:
I will be using trained Convnet used in paper Zeiler and Fergus., 2013, Visualizing and understanding convolutional networks and visualize what hidden units in different layers are computing. Input to the below network is ImageNet data spread over 1000 categories.

![F-2](https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/master/images/Neural_style_transfer/NS_fig2.png) 

Lets start with a hidden unit in layer 1 and find out the images that maximize that units activation. So we pass our training set through the above network and figure out what is the image that maximizes that units activation. Below are the image patches that activated randomly chosen 9 different hidden units of layer 1

<img align="left" src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/master/images/Neural_style_transfer/NS_fig3.png">
Fig. 3 (a) gives sense that hidden units in layer 1 are mainly looking for simple features like edges or shades of color. For example, first hidden unit(Row1/Col1) is getting activated for all 9 images whenever it see an slant edge. Same way Row2/Col1 hidden unit is getting activated when it sees orange shade in input image.<br/>







ddfdfasdf
