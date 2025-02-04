# Sentiment Analysis of Tweets Using LSTM

## Overview
This project implements a deep learning model using a Bidirectional Long Short-Term Memory (BiLSTM) network for sentiment analysis of tweets. The model classifies tweets into three sentiment categories: Positive, Neutral, and Negative. The dataset undergoes preprocessing, feature extraction using GloVe embeddings, and classification using a BiLSTM network.

## Table of Contents
- [Preprocessing](#preprocessing)
- [Embeddings](#embeddings)
- [Model Architecture](#model-architecture)
- [Model Configuration](#model-configuration)
- [Sentiment Prediction](#sentiment-prediction)
- [Regularization](#regularization)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Preprocessing
To ensure data consistency and remove noise, the following preprocessing steps were applied:
- **Lowercasing:** Converted all text to lowercase.
- **Removing URLs:** Eliminated links using regex.
- **Removing Mentions:** Cleared Twitter mentions (e.g., `@username`).
- **Removing Special Characters:** Retained only alphanumeric characters and spaces.
- **Whitespace Stripping:** Removed extra spaces for better formatting.

## Embeddings
GloVe (Global Vectors for Word Representation) embeddings were used to convert text into numerical feature vectors:
- **Embedding Source:** Pretrained GloVe vectors (`glove.6B.100d.txt`).
- **Embedding Dimension:** 100-dimensional feature vectors.
- **Mapping:** Each word in the vocabulary is mapped to a pretrained vector.
- **Static Embeddings:** Embeddings were set to non-trainable to retain pretrained knowledge.

## Model Architecture
The BiLSTM model consists of the following layers:
1. **Embedding Layer:** Converts input tokens into 100-dimensional vectors using the GloVe embedding matrix.
2. **SpatialDropout1D:** Applies dropout across the embedding sequence to reduce overfitting.
3. **Bidirectional LSTM Layers:**
   - First Layer: 256 LSTM units, processing sequences in both forward and backward directions.
   - Second Layer: 128 LSTM units summarizing the processed sequence.
4. **Fully Connected Layers:**
   - Dense layer with 128 neurons and ReLU activation.
   - Output layer with 3 neurons and softmax activation for sentiment classification.
5. **Regularization:** Dropout layers with a 50% dropout rate were included after LSTM and dense layers.

## Model Configuration
- **Loss Function:** Sparse Categorical Crossentropy (for multi-class classification).
- **Optimizer:** Adam optimizer.
- **Metrics:** Accuracy was tracked during training.

## Sentiment Prediction
The model assigns a sentiment label based on the output probabilities from the softmax layer:
- **Positive (Label: 2)**
- **Neutral (Label: 1)**
- **Negative (Label: 0)**

Predicted sentiments are saved to a CSV file (`test_predictions.csv`) for further analysis.

## Regularization
To enhance generalization and prevent overfitting, the following techniques were used:
- **Dropout Layers:** Applied after LSTM and dense layers.
- **SpatialDropout1D:** Reduces spatial correlations in the embedding sequences.
- **Early Stopping:** Training halts when validation loss does not improve for 3 consecutive epochs.

## Evaluation Metrics
The model was evaluated on a validation dataset with the following performance metrics:
- **Accuracy:** 88.8%
- **Precision:** 89.0%
- **Recall:** 88.7%
- **F1 Score:** 88.6%

These results demonstrate strong classification performance and effective handling of class imbalance through oversampling and model regularization.

## Results
Final predictions for the test dataset are stored in `test_predictions.csv` for external evaluation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-lstm.git
   cd sentiment-analysis-lstm
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download GloVe embeddings:
   ```bash
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   ```

## Usage
1. **Preprocess Data:** Run the preprocessing script.
   ```bash
   python preprocess.py
   ```
2. **Train the Model:** Execute the training script.
   ```bash
   python train.py
   ```
3. **Predict Sentiments:** Run the prediction script.
   ```bash
   python predict.py
   ```

## Contributing
Contributions are welcome! Feel free to submit issues and pull requests.

## License
This project is licensed under the MIT License.

