---
layout: post
title:  "OpenAI GPT"
tags: [ Tips, Natural Language Processing, Neural Networks  ]
featured_image_thumbnail: assets/images/Transformers/we_0_thumbnail.png
featured_image: assets/images/Transformers/we_0.png
---


In NLP, labelled data for training any task such as textual entailment, question answering, semantic similarity assessment, named entity recognition is very scarce. This makes it
very challenging for algorithms to give state of art results on these tasks. We have same kind of problem in computer vision and deep learning community started using transfer learning 
as a way to transfer knowledge learnt from huge amount of generic data to the task at hand. But transfer learning in NLP is relatively new one. Most compelling evidence of transfer
learning in NLP has been the extensive use of pre-trained [word embeddings](https://raviteja-ganta.github.io/Word-Embeddings) to improve performance on a range of NLP tasks. 
These approaches, however, mainly transfer word-level information which may not be sufficient. How can we transfer higher-level semantics than word level information with long range sequences?

OpenAI in their [GPT paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) demonstrated that this is possible by first pre-training of language model on diverse corpus of unlabeled text followed by fine-tuning on specific task. 


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/GPT/gpt_1.png" />
</p>








