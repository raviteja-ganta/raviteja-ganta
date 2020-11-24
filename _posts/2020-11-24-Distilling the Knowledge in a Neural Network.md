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


At prediction time using bigger/ensemble of models is very expensive and will hinder deployment to large number of users. So, now the question is how can we compress knowledge from this bigger model in to smaller model which can be easily deployed.


Geoffrey Hinton, Oriol Vinyals and Jeff Dean from google through their [paper](https://arxiv.org/pdf/1503.02531.pdf) came up with different kind of training called **distillation** to transfer this knowledge to the smaller model. This is the same technique which hugging face used in their [Distill BERT](https://arxiv.org/pdf/1910.01108.pdf) implementation.


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/master/images/Distill_knowledge/dk_3.png" />
</p>

