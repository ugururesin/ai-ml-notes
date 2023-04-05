### DEEP LEARNING ALGORITHMS

| Algorithm | Problem Type | Pros | Cons | Failure Modes | Remedies |
| --- | --- | --- | --- | --- | --- |
| Neural Networks | Regression and Classification | Can handle complex relationships and a large number of input features | Slow to train, sensitive to hyperparameters, requires a large amount of labeled data | Overfitting | Regularization, Transfer learning, Data augmentation, Early stopping |
| Convolutional Neural Networks | Image Classification | Can handle large amounts of image data, can extract features from images | Slow to train, requires a large amount of labeled data, sensitive to hyperparameters | Overfitting | Regularization, Transfer learning, Data augmentation, Early stopping |
| Recurrent Neural Networks | Sequence Prediction | Can handle sequential data, can maintain context over time | Slow to train, requires a large amount of labeled data, sensitive to hyperparameters | Overfitting, Vanishing gradients | Regularization, Transfer learning, Data augmentation, Early stopping, Gradient Clipping |
| Long Short-Term Memory (LSTM) | Sequence Prediction | Can handle sequential data, can maintain context over long sequences | Slow to train, requires a large amount of labeled data, sensitive to hyperparameters | Overfitting | Regularization, Transfer learning, Data augmentation, Early stopping |
| Autoencoders | Unsupervised Learning | Can be used for feature extraction and dimensionality reduction | Can be slow to train, requires a large amount of labeled data | Overfitting | Regularization, Transfer learning, Data augmentation, Early stopping |
| Generative Adversarial Networks | Unsupervised Learning | Can generate new data samples that are similar to the original dataset | Slow to train, prone to instability | Mode Collapse | Regularization, Transfer learning, Data augmentation, Early stopping |
| Transformer Networks | Sequence Prediction | Can handle sequential data, can maintain context over time | Slow to train, requires a large amount of labeled data, sensitive to hyperparameters | Overfitting | Regularization, Transfer learning, Data augmentation, Early stopping |
| Q-Learning | Reinforcement Learning | Can be used to learn optimal policies for decision-making tasks | Can be slow to converge, sensitive to hyperparameters | Overfitting, Slow convergence | Regularization, Transfer learning, Data augmentation, Early stopping |
| SARSA | Reinforcement Learning | Can be used to learn optimal policies for decision-making tasks | Can be slow to converge, sensitive to hyperparameters | Overfitting, Slow convergence | Regularization, Transfer learning, Data augmentation, Early stopping |

### TOKENIZATION FOR NLP
| Method | Example | Advantages | Disadvantages |
|--------|---------|------------|---------------|
| Bag-of-Words | ["The cat in the hat", "The dog in the yard"] | - Simple and easy to implement <br> - Works well for small datasets | - Ignores word order and context <br> - Does not capture relationships between words |
| TF-IDF | ["The cat in the hat", "The dog in the yard"] | - Considers word frequency and document frequency <br> - More informative representation than Bag-of-Words | - Can still miss important context <br> - Requires tuning of parameters |
| Word Embeddings | ["The cat in the hat", "The dog in the yard"] | - Captures relationships between words <br> - Able to represent new, unseen words | - Can be computationally expensive <br> - May not perform well for rare words |
