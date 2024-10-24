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
  <img src="images/Layer 5 (2).png" alt="Pretraining workflow" width="400"/>
</p>


## Data for pretraining
The CNN Daily dataset [available here](https://huggingface.co/datasets/abisee/cnn_dailymail?row=26) is a rich collection of diverse news articles covering various topics, offering complex linguistic structures and named entities. It is highly valuable for training language models to perform tasks like summarization, question answering, and information extraction, as it mirrors real-world language use.

## Data preprocessing : Dataset Anonymisation
 
In our case we have chosen the **Flair NER model** [available here](https://huggingface.co/flair/ner-english-ontonotes-large) which employs stacked Bidirectional Long Short-Term Memory (BiLSTM) networks to capture long-range dependencies in both forward and backward directions, which enhances the model’s contextual understanding . Finally, the incorporation of a Conditional Random Field (CRF) layer at the end allows the model to capture
sequence-level tag dependencies, ensuring that the predicted labels form a valid sequence consistent with the training data

<p align="center">
  <img src="images/bilstm_crf.png" alt="Architecture of a BiLSTM-CRF mode" width="400"/>
</p>

Specifically, when the Named Entity Recognition (NER) process identifies an entity $\( e_i \)$ within a document, it is replaced as follows:

$$
\begin{equation}
f(e_i) = \text{"ent"} + ID(e_i),
\end{equation}
$$

where $\( e_i \)$ is the identified entity in the document and $\( ID(e_i) \)$ is a unique integer assigned to $\( e_i \)$ within that document.


## Training Unigram tokenizer

The Unigram model aims to find a vocabulary $\( V \)$ and subword sequence $\( s = (s_1, s_2, \dots, s_n) \)$ for a given input text $\( T \)$ that maximizes the likelihood:

$$
\begin{equation}
P(T) = \prod_{i=1}^{n} P(s_i),
\end{equation}
$$

where $\( P(s_i) \)$ is the probability of the subword unit $\( s_i \)$.

## GPT2 architecture


The GPT-2 model consists of several key components :

$$
\begin{enumerate}
    \item \textbf{Embedding Layer}: Transforms tokens into continuous vectors.
    \begin{itemize}
        \item  \textbf{Token Embeddings}: Each token \( t_i \) is mapped to an embedding \( \mathbf{e}_i \in \mathbb{R}^d \).
        \item \textbf{Position Embeddings}: Adds positional information to maintain sequence order.
    \end{itemize}
    
    \item \textbf{Transformer Blocks}: Stacked layers consisting of:
    \begin{itemize}
       \item  \textbf{Masked Multi-Head Self-Attention}: 
     \[
     \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V
     \]
   
   \item \textbf{Feed-Forward Neural Network (FFN)}:
     \[
     \text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
     \]
     \item \textbf{Layer Normalization}: Normalizes inputs to stabilize training.
 \end{itemize}
 
    \item \textbf{Output Layer}: Projects hidden states back to the token vocabulary for next-token prediction.
\end{enumerate}

$$

<p align="center">
  <img src="images/Full_GPT_architecture.png" alt="GPT2 Architecture" width="400"/>
</p>


