---
layout: post
published: true
title: Sentiment Analysis using BERT and hugging face
---

This article talks about how can we use pretrained language model BERT to do transfer learning on most famous task in NLP - Sentiment Analysis 

Transfer learning is very popular in deep learning but mostly confined to computer vision. Transfer learning in NLP is not very popular until recently(thanks to pretrained
language models). But this usage went up so fast starting 2018 after paper on BERT was released. BERT stands for Bidirectional Encoder Representations from Transformers. We can think of this as a language models which looks at both left and right context when prediciting current word. This is what they called masked language modelling(MLM). Additionally BERT also use 'next sentence prediction' task in addition to MLM during pretraining. We can use this pretrained BERT model for transfer learning on downstream tasks like our Sentiment Analysis.

Sentiment Analysis is very popular application in NLP where goal is to find/classify emotions in subjective data. For example given a restaurent review by customer, using sentiment analysis we can understand what customer thinks of the restaurent(whether he likes or not). Before the rise of transfer learning in NLP, RNN's like LSTM's/GRU's are widely used for sentiment analysis to build from scratch. These gave decent results, but what if we can use pretrained unsurpervised models which already have lot of information on how language is structrued(because they were pretrained on massive unlabelled data) for our use case just by adding one additional layer on top of it and just fine tune total model for the task at hand. BERT showed that if we do it this way, we can save lot of time and also get state of art results even with smaller training data.

Below we can see how finetuning is done

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/master/images/Bert_sentiment/Bs_f1.png" />
</p>

Input traning sentence is passed through pretrained BERT model and on top of BERT we add one layer of Feed forward NN with softmax for our sentiment classification. Final hidden state corresponding to [CLS] token is used as the aggregate sequence representation for classification. According to paper, final hidden state is of 768 dimensions but for illustration I used 4 dimensions.

For this project, I used smaller vesion of BERT called DistillBERT. Huggingface leveraged knowledge distillation during pretraning phase and reduced size of BERT by 40% while retaining 97% of its language understanding capabilities and being 60% faster. 

I tested with both base BERT(BERT has two versions BERT base and BERT large) and DistillBERT and found that peformance dip is not that great when using DistillBERT but training time decreased by 50%.




