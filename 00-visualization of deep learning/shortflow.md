Neural Network
│
├── Architecture
│     ├── ANN (MLP)
│     ├── CNN
│     ├── RNN / LSTM / GRU
│     └── Transformer
│
├── Forward Pass
│     ├── Linear Transform         W·x + b
│     │
│     ├── Normalization
│     │     ├── BatchNorm          CNN · ANN
│     │     ├── LayerNorm          Transformer
│     │     └── RMSNorm            LLaMA · Mistral · modern LLMs
│     │
│     ├── Activation Functions
│     │     ├── Hidden Layers      ReLU · Leaky ReLU · ELU · GELU · Swish
│     │     ├── RNN / Gates        Tanh · Sigmoid
│     │     └── Output Layer       Sigmoid (binary) · Softmax (multi-class)
│     │
│     └── Dropout                  training only · zeros random neurons
│
├── Loss Function
│     ├── Regression               MSE · MAE · Huber · Log-Cosh
│     ├── Classification           BCE · Cross Entropy · Focal · Hinge
│     ├── Sequence                 Cross Entropy · NLL
│     └── Generative               KL Divergence · Minimax · Wasserstein
│
├── Backpropagation
│     ├── Standard                 ANN · CNN  (chain rule through layers)
│     ├── BPTT                     RNN · LSTM · GRU  (chain rule through time)
│     ├── Through Attention        Transformer  (chain rule through attention matrix)
│     └── Gradient Clipping        clip_grad_norm_  → prevents explosion
│
├── Optimizer
│     ├── SGD family               SGD · Momentum · Nesterov
│     ├── Adaptive                 Adam · AdamW · RMSProp · AdaGrad
│     ├── Modern                   Lion · Adafactor · LAMB
│     └── Weight Decay             L2 penalty applied here during update
│
├── Regularization                 (conceptual group — executes across multiple steps)
│     ├── Dropout                  → lives in Forward Pass
│     ├── Weight Decay             → lives in Optimizer
│     └── Early Stopping           → wraps entire training loop
│
├── Hyperparameter Tuning          (wraps the entire loop)
│     ├── Learning rate
│     ├── Batch size
│     ├── Architecture depth
│     ├── Dropout rate
│     └── Optimizer choice
│
└── Evaluation / Inference
      ├── Metrics
      │     ├── Classification     Accuracy · Precision · Recall · F1 · ROC-AUC
      │     ├── Regression         MAE · MSE · RMSE · R²
      │     ├── Language Models    Perplexity · BLEU · ROUGE
      │     └── Generation         FID (images) · MOS (audio)
      └── Inference
            ├── Greedy             always pick highest probability token
            ├── Beam Search        keep top-k candidates at each step
            └── Sampling           temperature · top-k · top-p (nucleus)