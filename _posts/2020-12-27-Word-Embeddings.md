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


### Word Embeddings helps transfer learning:

As we have seen above in example, even though our training data for language modelling does not have word *apple*, if representation for *apple* is similar to word *orange* (both are fruits so we expect most of characteristics of them are same, so similar representation) then our language model is more probable to predict next word as *juice* in sentence *I want a glass of apple*.


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/word embeddings/we_2.png" />
</p>


### How word embeddings look:

Word embeddings are continuous, vector space representations of words. Word embedding vectors usually have dimensions around 300(in general embed_size). So each word in vocablury are represented by a vector of size 300. But what does each value of this vector of embed_size represent? We can think of them as representing certain characteristics like age, food, gender, size, royal etc. Let's go through an example to understand this in detail. Even though below example shows that embeddings are represented by characterstics like age, food, gender etc., embeddings which we learn wont have an easy interpretation like component one is *age*, component two is *food* etc., Important point to understand is that embeddings learn certain characteristics which will be very similar for similar words and different for dissimilar words.


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/word embeddings/we_3.png" />
</p>


If we look at word embeddings of words *Apple* and *orange*, they look almost identical. So dot product of these two vectors would be high.

Now that we have decent understanding of word embeddings, let's see how we can produce these vectors of embed_size size for every word in vocablary.


### Generating Word Embeddings:

Below are architectures we will go through

* Word2Vec Skip-Gram model

* Word2Vec Skip-Gram model with negative sampling



### Word2Vec Skip-Gram model


Main idea behind Word2Vec Skip-Gram model is that *words that show up in similar context will have similar representation*. Context = Words that are likely to appear around them.

Lets understand this with an example:

1) I play cricket every alternate day

2) Children are playing football across the street

3) Doctor advised me to play tennis every morning.


Above, if we see words *cricket*, *football* and *tennis* all have word *play* surrounding them(similar context). So these 3 words will have similar word embedding vectors which is justifiable as these 3 words are names of sport.


For any word, we can think of context as surrounding words. So our task is to predict a randomly selected surrounding word from the word of interest. Lets understand this with an example *Children are playing football across the street*. Let's pick our word of interest as word *football* and *playing* as the surrounding word. So we give word *football* to our neural network(which we are going to train and build) and this neural network should predict with high probability the word *playing*. But context can include surrounding words that come before and also after and not just one word. So authors of [paper](https://arxiv.org/pdf/1301.3781.pdf) used window size to select surrounding context words as shown below.


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/word embeddings/we_4.png" />
</p>


#### Model architecture:

Below is the model architecture we will be using


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/word embeddings/we_5.png" />
</p>


Each training input as show in fig 3 will be first represented as one hot vectors that go in to model as shown in fig 4. Then we have embedding layer with number of neurons = embed_size(usually this is set to 300) and output layer will be 10,000 neuron softmax(here vocablary size = 10,000) that outputs probability of each word in vocablary.


For example, when word *football* is sent as input to trained Word2Vec model above, then final layer will output high probabilities for the words *are*, *playing*, *across*, *the* as these are output words for input *football* in training set.


Model is trained like any other neural network where we update coefficients using backpropagation and loss function.

#### Hidden layer

Hidden layer is where all the magic happens. Coefficients of this hidden layer are the actual word embeddings.

For example, let's say we are learning word embeddings of dimension 300 for every word. Then coefficients or weight matrix of hidden embedding layer will be of size 10000(vocab_size) X 300(embed_size). Output of hidden layer will be of size 300(embed_size) X 1. This output will pass through output layer with softmax to generate 10000 values.


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/word embeddings/we_6.png" />
</p>


Instead of matrix multiplication as shown above, we can directly use embedding weight matrix as look up table. Each row in embedding weight matrix gives embedding vector for a word(result would be same) in the vocabulary.


#### Intuition


As we have seen above, main idea behind Word2Vec Skip-Gram model is that

> Words that show up in similar context will have similar representation

So if two words have similar context(i.e., same words surrounding them) then our model needs to output similar output for these 2 words. For example, in our 3 sentences, words *cricket*, *football*, *tennis* have similar context as word *playing* is the surrounding word. Generally speaking if we take large text corpora there is very high chance that these 3 words(cricket, football, tennis) will have *playing* as surrounding word. So our model has to output *playing* as output with higher probability for these 3 words. This can only happen if embedding vectors for these 3 words are similar because embedding vectors are the one that go in to output softmax layer as shown in Fig 4






