# LM Without Entity Knowledge

[![python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)


# Project Purpose

We want to investigate LM trained on anonymized data(without entities) can perform comparable as standard LM trained.

# Methodoly

In order to realize this study, we have followed the differents sets.

- Collect data inorder to pretrain a model : CNN and Daily mail 
- Anonymized the dataset with NER model by detecting enties and replaced them with placeholder
- Train a tokenizer in our case we have chosen a Unigram Tokenizer
- Select the transformer architecture
- Pretrain the model by choosing adapt hyperparameter for a good training and convergence
- Fine- Tune the model without entity on task who doesn't require entity knowledge like Glue benchmark an Question answering (Squad v1)

# Pretraining model

<p align="center">
  <img src="images/Layer 5 (2).png" alt="Rating" width="400"/>
</p>


## Data for pretraining
The CNN Daily dataset [available here](https://huggingface.co/datasets/abisee/cnn_dailymail?row=26) is a rich collection of diverse news articles covering various topics, offering complex linguistic structures and named entities. It is highly valuable for training language models to perform tasks like summarization, question answering, and information extraction, as it mirrors real-world language use.

## Data preprocessing : Dataset Anonymisation
 
In our case we have chosen the **Flair NER model** [available here](https://huggingface.co/flair/ner-english-ontonotes-large) which employs stacked Bidirectional Long Short-Term Memory (BiLSTM) networks to capture long-range dependencies in both forward and backward directions, which enhances the model’s contextual understanding . Finally, the incorporation of a Conditional Random Field (CRF) layer at the end allows the model to capture
sequence-level tag dependencies, ensuring that the predicted labels form a valid sequence consistent with the training data


<p align="center">
  <img src="images/bilstm_crf.png" alt="Rating" width="400"/>
</p>





