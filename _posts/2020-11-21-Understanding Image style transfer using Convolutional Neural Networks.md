---
layout: post
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

![N](https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/master/images/NS_fig1.png) 

*Fig. 1: Style transfer: Target image looks like content image painted with style of style image*

