---
layout: post
title:  "Word Embeddings"
tags: [ Tips, Natural Language Processing, Neural Networks  ]
featured_image_thumbnail: assets/images/Transformers/tf_1_thumbnail.jpg
featured_image: assets/images/Transformers/tf_2.jpg
---


Its obvious that any deep learning model needs to have some sort of numeric input to work with. In computer vision its not a problem as we have pixel values as inputs but in
Natural language processing we have text which is not a numeric input. One common way to convert text to numbers is to use one hot representation. For example, let's say we have
10,000 word vocablary, so any word can be represented by a vector of length 10,000 with 0's everywhere except a one '1' in words position in vocablary. 


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/word embeddings/we_1.png" />
</p>


### Problem with One-Hot encoding:

Problem with One-Hot encoding is that vector will be all zeros except one unique index for each word as shown above in fig 1. Representing words in this way leads to substantial data sparsity and usually means that we may need more data in order to successfully train statistical models.

Apart from above issue main problem is of identifying similar words using one-hot encoding vectors. Lets see what I mean by this with help of 2 sentences. *Examples are from Deep learning specilalization - CourseEra.

Let's say we have a trained language model and one of the traning examples is *I want a glass of orange **Juice***. This language model can easily predict Juice as last word by looking at word *orange*. Let's say our language model never seen word *apple* before. So, When our language model encounters sentence *I want a glass of apple* to predict next word, it will be able to predict word *Juice* if representation for both words *orange* and *apple* are similar. But in our case using one-hot representation relation between apple and orange is not any closer(similar) as the relationship between any of the words *a* or *an* or *zebra* etc. It's because dot product between any 2 different one-hot vector is zero. So it doesn't know that *apple* and *orange* are much more similar than let's say *apple* and *an* or *apple* and *king*. But the point is *apple* and *orange* are fruits and the representations for these should be similar.

So, we want semantically similar words to be mapped to nearby points, thus making the representation carry useful information about the word actual meaning. We call this new representation as **Word Embeddings**.







