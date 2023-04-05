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
| Tokenization Method | Example | Advantages | Disadvantages |
| --- | --- | --- | --- |
| Whitespace tokenizer | "This is a sentence." | Simple, easy to implement | Does not handle punctuation, special characters well |
| Punctuation tokenizer | "This is a sentence." | Handles punctuation well | May split contractions and compound words |
| Word tokenizer | "This is a sentence." | Handles contractions and compound words well | May split words with hyphens or slashes |
| Regex tokenizer | "This is a sentence." | Highly customizable | Requires knowledge of regular expressions |
| Sentence tokenizer | "This is a sentence. This is another sentence." | Splits text into sentences | May struggle with abbreviations or titles |
| Byte Pair Encoding tokenizer | "This is a sentence." | Handles rare or out-of-vocabulary words | Can result in large vocabularies |
| Sentencepiece tokenizer | "This is a sentence." | Handles rare or out-of-vocabulary words, highly customizable | Requires additional libraries, may be slower |

