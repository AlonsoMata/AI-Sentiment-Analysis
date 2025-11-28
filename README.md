SVM-BERT Sentiment Analysis on YouTube Comments

Overview

This project is a state-of-the-art Sentiment Analysis Tool that combines the power of BERT (Bidirectional Encoder Representations from Transformers) and Support Vector Machines (SVMs). It is designed to analyze public sentiment regarding OpenAI and its technologies based on YouTube comments, offering valuable insights into public opinion.

Key highlights include:

Advanced Natural Language Processing (NLP) techniques.

Seamless integration of BERT embeddings for text representation.

Highly optimized SVM classifiers for sentiment classification.

Scalable design adaptable to various datasets.

This repository demonstrates cutting-edge machine learning, making it an ideal showcase for organizations seeking AI-savvy talent.

Key Features

Data Pipeline:

Web scraping via YouTube’s API using Python.

Comprehensive text preprocessing: cleaning, tokenization, lemmatization, and language detection.

Modeling:

BERT: Leveraging Hugging Face’s Transformers library for embeddings.

SVM: Training and hyperparameter tuning for classification tasks.

Evaluation Metrics:

Performance measured via Precision, Recall, F1 Score, and Confusion Matrices.

Feature importance visualization using attention mechanisms from BERT.

Deployment Ready:

Scalable codebase, modular design, and flexibility for adapting to new datasets.

Installation Guide

Clone the Repository:

git clone https://github.com/your_username/SVM-BERT-Sentiment-Analysis.git
cd SVM-BERT-Sentiment-Analysis

Set Up the Environment:

python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate       # For Windows

Install Dependencies:

pip install -r requirements.txt

Install GPU Support (Optional for Faster Training):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Usage

Prepare the Dataset:

Place your dataset in the data/ directory in .csv format.

Ensure the dataset has two columns: text (YouTube comments) and label (sentiment: Positive, Negative, Neutral).

Run the Training Script:

python SVM_BERT_script.py

Evaluate Model Performance:

The script automatically outputs key metrics, including:

Precision, Recall, and F1 Score for each class.

Confusion Matrix.

Results are stored in the results/ directory.

Technical Details

Data Preprocessing

Text Cleaning: Removal of HTML tags, emojis, and timestamps.

Language Detection: Standardized all comments to English.

Tokenization & Lemmatization: Utilized NLTK and Stanza for linguistic simplification.

BERT Embeddings

Pre-trained BERT model from Hugging Face (BERT-base).

Fine-tuned on the specific dataset for enhanced embeddings.

SVM Classifier

Kernel selection: RBF kernel optimized via GridSearch.

Hyperparameter tuning for C and gamma values to maximize accuracy.

Implements one-vs-rest (OvR) strategy for multi-class classification.

Performance Metrics

Confusion Matrix for visualizing predictions.

Metrics: Accuracy, Precision, Recall, F1 Score, Macro Average, and Weighted Average.

Results

Metric

BERT + SVM

Accuracy

93.7%

Precision

92.4%

Recall

91.8%

F1 Score

92.1%

BERT embeddings outperformed traditional vectorization techniques (e.g., TF-IDF), demonstrating its ability to capture contextual meaning.

SVM achieved superior classification performance with optimal hyperparameters.

Future Work

Expand the dataset to include comments in multiple languages.

Experiment with ensemble models combining SVM and other classifiers (e.g., Random Forest).

Integrate visualization dashboards for real-time sentiment analysis.

Contributing

Contributions are welcome! Please:

Fork the repository.

Create a new branch (feature/YourFeature).

Submit a pull request with a detailed description of changes.

Contact

Authors: Jorge Villacorta, Alejandro Tabernero, Alonso Mata

Email: alon28500@gmail.com

LinkedIn: YourLinkedInProfile

GitHub: YourGitHubUsername

License

Attribution-NonCommercial-NoDerivatives 4.0 International
