---
layout: post
title:  "Reformer - The efficient Transformer"
tags: [ Tips, Natural Language Processing, Neural Networks  ]
featured_image_thumbnail: assets/images/Transformers/tf_1_thumbnail.jpg
featured_image: assets/images/Transformers/tf_2.jpg
featured: true
hidden: true
---


### Transforme Complexity:

Recent trend in Natural Language Processing is to use Transformer Architecture for wide range of NLP tasks including but just not limited to Machine translation, Question-Answering,
Named entity recognition, Text Summarization. It takes advantage of parallelization and relies on attention mechanism to draw global dependencies between input and output. 
It was shown to successfully outperform LSTM's and GRU's on wide range of tasks.

In my recent blog [post](https://raviteja-ganta.github.io/attention-is-all-you-need-transformers) I tried explainig Transformer architecture for machine translation. For more 
detail on the complete transformer operation, have a look at it.

Transformer models are also used on increasingly long sequences. Up to 11 thousand tokens of text in a single example were processed in (Liu et al., 2018).

Transformers peformed well on short sequences but it will easily runs out of memory when run on long sequences. Lets understand why

* Attention on sequence of length L takes L<sup>2 time and memory. For example, if we are doing self attention of two sentences of length L then we need to compare each word in
first sentence with each word in second sentence which is L X L comparision = L<sup>2. This is just for one layer of transformer.

* Recent transformer models have more that one layer. For example recent GPT3 model from OpenAI has 96 layers. So number of comarsions and memory requirement goes up by number of layers.

This is not a problem for shorter sequence lengths but take an example given in the [Reformer paper](https://arxiv.org/pdf/2001.04451.pdf)
