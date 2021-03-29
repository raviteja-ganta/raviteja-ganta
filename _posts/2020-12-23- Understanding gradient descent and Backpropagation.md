---
layout: post
title:  "Understanding gradient descent and Backpropagation"
tags: [ Tips, Neural Networks]
featured_image_thumbnail: assets/images/Transformers/tf_1_thumbnail.jpg
featured_image: assets/images/Transformers/tf_2.jpg
featured: true
hidden: true
---


Gradient descent and backpropagation are workhorse behind training neural networks and understanding what's happeing inside these algorithms is atmost importance
for efficient learning. This post gives in depth explanation of gradient descent and Backpropagation for training neural networks.


Below are the contents:

1) Notation to represent neural network
2) Forward propagation



1) Notation to represent a neural network:

I will be using a 2 layer neural network as shown below through out this post as our running example.

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Backprop/bp1.png" />
</p>

*w<sub>jk</sub><sup>[l]</sup>* - Weight for connection from k<sup>th</sup> neuron in layer (l-1) to j<sup>th</sup> neuron in layer l

*b<sub>j</sub><sup>[l]</sup>* - Bias of j<sup>th</sup> neuron in layer l

*a<sub>j</sub><sup>[l]</sup>* - Activation of j<sup>th</sup> neuron in layer l

2) Forward propagation:

Below is the calculation that happens as part of forward propagation in single neuron

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Backprop/bp2.png" />
</p>

So the activation at any neuron in layer l can be written as 

*a<sub>j</sub><sup>[l]</sup>* = *f*(<img src="https://render.githubusercontent.com/render/math?math=\displaystyle \sum_{k}"> 

*a<sub>j</sub><sup>[l]</sup>* = <img src="https://render.githubusercontent.com/render/math?math=f">(<img src="https://render.githubusercontent.com/render/math?math=\displaystyle \sum_{k}"> 



