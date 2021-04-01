---
layout: post
mathjax: true
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

<img src="https://render.githubusercontent.com/render/math?math=\LARGE a_j^[l]"> = <img src="https://render.githubusercontent.com/render/math?math=\LARGE f"><img src="https://render.githubusercontent.com/render/math?math=\LARGE ("><img src="https://render.githubusercontent.com/render/math?math=\displaystyle \sum_{k}"> <img src="https://render.githubusercontent.com/render/math?math=\LARGE w_jk^[l] a_k^[l-1]">  

Matrix representation:

For Neural network in Fig 1 above,

*w<sup>[l]</sup>* - Weight matrix for layer l

We use equation 1 from above to calulate activations for every layer but all calculations are done using matrix mulitiplcations as they are very fast

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Backprop/bp3.png" />
</p>


Main goal of backpropagation is to understand how varying individiual weights of network changes the errors made by the network. But how do we calculate error made by the nextwork. We use term called cost function to get estimate of error. Cost gives estimate of how bad or good our network predictions are. This is function of actual label and predicted value

<img src="https://render.githubusercontent.com/render/math?math=\LARGE C = f(y^i,yhat)"> 
where <img src="https://render.githubusercontent.com/render/math?math=\LARGE y^i"> is the actual label or truth of data point *i* and yhat is the prediction from neural network.


