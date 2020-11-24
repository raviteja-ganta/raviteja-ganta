---
layout: post
published: true
title: Distilling the Knowledge in a Neural Network
---
 

How can we compress and transfer knowledge from bigger model or ensemble of models(which were trained on very large datasets to extract structure from data) to a single small
model with out much dip in performance?


But why do we want to do this? Why we need smaller model when bigger model or ensemble model is already giving great results on test data?


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/master/images/Distill_knowledge/dk_1.png" />
</p>



At training time we typically train large/ensemble of models because the main goal is to extract structure from very large datasets. We could also be applying maythings like dropout, data augmentation at train times to feed these large models all kinds of data.



But at prediction time our objective is totally different. We want to get results as quickly as possible. So using bigger/ensemble of models is very expensive and will hinder deployment to large number of users. So, now the question is how can we compress knowledge from this bigger model in to smaller model which can be easily deployed.



Geoffrey Hinton, Oriol Vinyals and Jeff Dean from google through their [paper](https://arxiv.org/pdf/1503.02531.pdf) came up with different kind of training called **distillation** to transfer this knowledge to the smaller model. This is the same technique which hugging face used in their [Distill BERT](https://arxiv.org/pdf/1910.01108.pdf) implementation.



<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/master/images/Distill_knowledge/dk_3.png" />
</p>



Usually in Machine learning, model that learns to discriminate between large number of classes, the main training objective is to maximize average log probability of correct answer. For example, take example of MNIST dataset where goal is to classify an image as whether its 1 or 2 or ... 9. So if actual image is 2 then objective of any model is to maximize **P(its 2/image)** (which can be read as probability that particular image is 2 given the image). But model also gives probabilities to all incorrect answers even though those probabilities are very small.



<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/master/images/Distill_knowledge/dk_4.png" />
</p>




