# Fake Message Detector

A machine learning-based solution to detect fake/spam messages on social media platforms using LSTM and TF-IDF models.

![Fake Message Detection](https://img.shields.io/badge/ML-Spam%20Detection-blue)
![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)

## Overview

This project implements a web application that can detect fake or spam messages using natural language processing and deep learning techniques. The system analyzes text content to determine whether it's legitimate or potentially spam/fake with high accuracy.

## Features

- **Multiple Classification Models**: 
  - LSTM neural network for sequence analysis
  - TF-IDF + Logistic Regression for comparison
  - Ensemble method combining both approaches

- **Interactive Web UI**:
  - Real-time message analysis
  - Prediction confidence scores
  - History tracking of previous detections
  - Statistics dashboard

- **Text Processing**:
  - Advanced NLP preprocessing 
  - URL, emoji, and special character handling
  - Stop word removal and lemmatization

## Project Structure

```
├── data/                    # Dataset files
│   ├── twitter_spam_data.csv    # Raw dataset
│   └── processed_twitter_spam.csv # Preprocessed dataset
├── models/                  # Trained model files
│   ├── lstm_model.h5        # LSTM neural network model
│   ├── tokenizer.pickle     # Text tokenizer
│   ├── tfidf_vectorizer.pickle # TF-IDF vectorizer
│   └── lr_model.pickle      # Logistic regression model
├── src/                     # Source code
│   ├── preprocessing.py     # Text cleaning and preprocessing
│   ├── main.py              # Model training script
│   ├── predict.py           # Prediction functionality
│   └── data_helper.py       # Dataset utilities
├── static/                  # Web assets
│   ├── css/                 # Stylesheets
│   └── js/                  # JavaScript files
├── templates/               # HTML templates
├── run_flask.py             # Flask web application
├── run.py                   # Main runner script with CLI
└── README.md                # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/BaverYldz/SpamHandler-DL.git
cd SpamHandler-DL

# Install Git LFS (if not installed)
# For Windows: https://git-lfs.github.com/
# For macOS: brew install git-lfs
# For Ubuntu/Debian: sudo apt install git-lfs

# Set up Git LFS
git lfs install

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The project uses a Twitter spam dataset with approximately 5,500 messages labeled as spam (1) or legitimate (0). The dataset includes various features such as:
- Message text
- Class label (spam/legitimate)

## Usage

### 1. Data Preparation

```bash
python src/data_helper.py
```

### 2. Training Models

```bash
python src/main.py
```

### 3. Running the Web Application

```bash
python run_flask.py
```
Then open your browser and navigate to: http://localhost:5000

### 4. Using the CLI Runner

```bash
python run.py
```
Choose from the interactive menu options to prepare data, train models, or launch the web app.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| LSTM | 87% | 84% | 81% | 82.5% |
| TF-IDF + LR | 85% | 82% | 80% | 81% |
| Ensemble | 88% | 85% | 82% | 83.5% |

## Technologies Used

- **Python** - Core programming language
- **TensorFlow/Keras** - Deep learning for LSTM model
- **NLTK** - Natural language processing
- **Scikit-learn** - Machine learning algorithms and metrics
- **Flask** - Web application framework
- **HTML/CSS/JavaScript** - Frontend interface

## Future Improvements

- Implement BERT transformer models for improved accuracy
- Add support for multiple languages (currently English-focused)
- Develop a REST API for integration with other applications
- Enable real-time social media monitoring

## Using Git LFS for Large Files

This project uses Git Large File Storage (LFS) to handle large files like datasets and models.

### Setting up Git LFS

1. Install Git LFS from https://git-lfs.github.com/
2. In your cloned repository, run:
   ```
   git lfs install
   ```
3. All large files are already tracked in .gitattributes

### Working with Large Files

- Git LFS files appear as normal files in your working directory
- Git LFS automatically handles these files during git operations
- If you add new large files, track them with:
  ```
  git lfs track "*.your_extension"
  git add .gitattributes
  ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
