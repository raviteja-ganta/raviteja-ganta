---
layout: post
published: true
title: Sentiment Analysis using BERT and hugging face
---

This article talks about how can we use pretrained language model BERT to do transfer learning on most famous task in NLP - Sentiment Analysis 

Transfer learning is very popular in deep learning but mostly confined to computer vision. Transfer learning in NLP is not very popular until recently(thanks to pretrained
language models). But this usage went up so fast starting 2018 after paper on BERT was released. BERT stands for Bidirectional Encoder Representations from Transformers. We can think of this as a language models which looks at both left and right context when prediciting current word. This is what they called masked language modelling(MLM). Additionally BERT also use 'next sentence prediction' task in addition to MLM during pretraining. We can use this pretrained BERT model for transfer learning on downstream tasks like our Sentiment Analysis.

Sentiment Analysis is very popular application in NLP where goal is to find/classify emotions in subjective data. For example given a restaurent review by customer, using sentiment analysis we can understand what customer thinks of the restaurent(whether he likes or not). Before the rise of transfer learning in NLP, RNN's like LSTM's/GRU's are widely used for sentiment analysis. These 
