---
layout: post
title:  "Reformer - The efficient Transformer"
tags: [ Tips, Natural Language Processing, Neural Networks  ]
featured_image_thumbnail: assets/images/Transformers/tf_1_thumbnail.jpg
featured_image: assets/images/Transformers/tf_2.jpg
---


### Transformer Complexity:

Recent trend in Natural Language Processing is to use Transformer Architecture for wide range of NLP tasks including but just not limited to Machine translation, Question-Answering, Named entity recognition, Text Summarization. It takes advantage of parallelization and relies on attention mechanism to draw global dependencies between input and output. It was shown to successfully outperform LSTM's and GRU's on wide range of tasks.

In my recent blog [post](https://raviteja-ganta.github.io/attention-is-all-you-need-transformers) I tried explaining Transformer architecture for machine translation. For more 
detail on the complete transformer operation, have a look at it.

Transformer models are also used on increasingly long sequences. Up to 11 thousand tokens of text in a single example were processed in (Liu et al., 2018) but problem is transformers peformed well on short sequences but it will easily runs out of memory when run on long sequences. Lets understand why

* Attention on sequence of length L takes L<sup>2 time and memory. For example, if we are doing self attention of two sentences of length L then we need to compare each word in
first sentence with each word in second sentence which is L X L comparision = L<sup>2. This is just for one layer of transformer.
  
* We also need to store activations from each layer of forward so that they can be used during backpropagation

* Recent transformer models have more that one layer. For example recent GPT3 model from OpenAI has 96 layers. So number of comarsions and memory requirement goes up by number of layers.

This is not a problem for shorter sequence lengths but take an example given in the [Reformer paper](https://arxiv.org/pdf/2001.04451.pdf). 0.5B parameters per layer = 2GB of memory, Activations for 64k tokens(long sequence) with embed_size = 1024 and batch_size = 8 requires another 2GB of memory per layer. If we have only one layer we can easily fit entire computations in memory. But recent transformer models have more than one layer(GPT3 has 96 layers). So total memory requirement would be aroudn 96 * 4GB ~ 400 GB which is impossible to fit on single GPU. So what we can do to efficiently run transformer architecture on **long sequences**. Here comes *Reformer*.


### Overview of Reformer:

Reformer solves the memory constraints of transformer model by adding 

* Locality sensitive hashing attention

* Reversible residual networks

* Chunked feed forward layers

Lets understand each component in detail.


### Locality sensitive hashing attention(LSH attention):

Before we start LSH attention, lets discuss briefly how standard attention works in transformers. For detailed information have a look at my [post](https://raviteja-ganta.github.io/attention-is-all-you-need-transformers) on transformers.

We have query and key of dimension d







