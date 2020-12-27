---
layout: post
title:  "Word Embeddings"
tags: [ Tips, Natural Language Processing, Neural Networks  ]
featured_image_thumbnail: assets/images/Transformers/tf_1_thumbnail.jpg
featured_image: assets/images/Transformers/tf_2.jpg
---


Its obvious that any deep learning model needs to have some sort of numeric input to work with. In computer vision its not a problem as we have pixel values as inputs but in
Natural language processing we have text which is not a numeric input. One common way to convert text to numbers is to use one hot representation. For example, let's say we have
10,000 word vocablary, so any word can be represented by a vector of length 10,000 with 0's everywhere except one '1' in words position in vocablary. 
