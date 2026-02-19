# Deep Sequence Modeling

Deep learning becomes significantly more powerful when we move from static inputs to sequential data. In many real-world problems, data is not independent — it evolves over time. Words in a sentence, notes in music, stock prices, biological sequences, and audio signals all carry meaning through order and temporal dependency.

Sequence modeling is about learning patterns where the prediction at time t depends not only on the current input, but also on what came before.

---

## From Static Models to Sequential Thinking

In a standard feedforward neural network, we assume inputs are independent. We pass an input vector through layers and produce an output. There is no memory of previous inputs.

However, consider predicting the next word in a sentence:

"This morning I took my cat for a ___"

To predict the next word correctly, the model must remember earlier words such as "cat" and "morning". This introduces the need for memory.

This is where sequence models begin.

---

## Recurrent Neural Networks (RNNs)

The key idea of an RNN is simple yet profound: introduce a hidden state that acts as memory.

At each time step t:

h_t = f(Wx_t + Uh_{t-1})
y_t = g(Vh_t)

Where:

* x_t is the input at time t
* h_t is the hidden state (memory)
* y_t is the output
* W, U, V are weight matrices

The hidden state carries information forward through time. The same weights are reused at every time step.

Conceptually:

Input_t → [RNN Cell] → Output_t
↑
Memory (h_{t-1})

Unrolling through time reveals that the RNN is effectively a deep network across time steps with shared parameters.

![alt text](Deep_Sequence_Modeling\support_image\image.png)
---

## Training RNNs: Backpropagation Through Time (BPTT)

Training RNNs requires extending backpropagation across time.

Instead of propagating error backward through layers only, we now propagate error backward across time steps.

Loss is computed at each time step:

Total Loss = sum over t of L_t

Gradients must flow from later time steps back to earlier ones.

This process is called Backpropagation Through Time (BPTT).

However, repeated multiplication of gradients across many time steps introduces instability.

Two major issues arise:

* Vanishing gradients: gradients shrink toward zero.
* Exploding gradients: gradients grow uncontrollably large.

Both make learning long-term dependencies difficult.

---

## Long Short-Term Memory (LSTM)

To address gradient instability, more advanced recurrent architectures were developed.

LSTM networks introduce gated mechanisms that regulate memory flow:

* Forget gate
* Input gate
* Output gate

These gates allow the network to selectively retain or discard information.

This significantly improves the model’s ability to learn long-range dependencies while stabilizing training.

---

## Representing Words Numerically: Embeddings

Neural networks operate on numbers. Therefore, words must be converted into vector representations.

Step 1: Build a vocabulary of all possible words.
Step 2: Assign each word an index.
Step 3: Map the index to a vector representation.

Two main approaches:

One-Hot Encoding:
Each word maps to a sparse vector with a single 1 and the rest 0.

Learned Embeddings:
Words map to dense vectors in a lower-dimensional space. Similar words occupy nearby positions in this embedding space.

Embeddings allow neural networks to capture semantic relationships.

---

## Limitations of RNNs

Despite their elegance, RNNs have limitations:

1. Fixed-size hidden state creates a memory bottleneck.
2. Sequential computation prevents full parallelization.
3. Long-range dependencies remain difficult to model.

These limitations motivated a shift away from recurrence.

---

## Attention Mechanism

Attention introduces a fundamentally different approach.

Instead of compressing all history into a single hidden state, attention allows each element in a sequence to directly reference all other elements.

Core components:

Query (Q)
Key (K)
Value (V)

Similarity is computed as:

Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V

Process:

1. Compute similarity between query and key.
2. Scale and normalize with softmax.
3. Use resulting weights to combine values.

This allows the model to "focus" on important parts of the sequence dynamically.

---

## Positional Encoding

Since attention removes recurrence, explicit position information must be added.

Positional encodings inject information about word order into embeddings, allowing the model to retain sequence structure.

---

## Transformers

The Transformer architecture, introduced in 2017 in "Attention Is All You Need," eliminates recurrence entirely.

A Transformer consists of:

* Multi-Head Self-Attention
* Feedforward Layers
* Residual Connections
* Layer Normalization

Multi-head attention allows the model to learn different types of relationships simultaneously.

Transformers enable:

* Full parallelization
* Better long-range dependency modeling
* Scalability to massive datasets

Modern Large Language Models (LLMs) such as GPT are built on Transformer architectures.

---

## Applications of Sequence Modeling

Sequence models power:

* Next-word prediction
* Language modeling
* Machine translation
* Music generation
* Speech recognition
* Vision Transformers
* Biological sequence modeling

Sequence modeling has expanded deep learning beyond static pattern recognition into generative and predictive intelligence.

---

# Concept Graph: Evolution of Sequence Modeling

```
                   [Static Neural Networks]
                             |
                             v
                  [Need for Memory Over Time]
                             |
                             v
                        [RNN]
                             |
        ---------------------------------------------
        |                                           |
   Backpropagation                          Gradient Issues
   Through Time                         (Vanishing / Exploding)
        |                                           |
        v                                           v
       LSTM                                  Memory Bottleneck
                                                     |
                                                     v
                                            Parallelization Limits
                                                     |
                                                     v
                                                 [Attention]
                                                     |
                       -----------------------------------------------
                       |                                             |
               Query-Key-Value                               Positional Encoding
                       |                                             |
                       v                                             v
                    Self-Attention  --------------------->  Order Awareness Restored
                       |
                       v
                 Multi-Head Attention
                       |
                       v
                   [Transformer]
                       |
                       v
              Large Language Models (GPT, etc.)
```

This graph shows the conceptual transition from recurrence-based memory to attention-based global reasoning, ultimately leading to scalable Transformer architectures.
