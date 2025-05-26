import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from preprocessing import clean_text

class SpamPredictor:
    def __init__(self):
        """
        Initialize the spam predictor with trained models
        """
        self.lstm_model = None
        self.tokenizer = None
        self.tfidf_vectorizer = None
        self.lr_model = None
        self.max_len = 100  # Same as in training
        self.is_ready = False
        
        # Load models if they exist
        self.load_models()
    
    def load_models(self):
        """
        Load trained models from disk
        """
        try:
            # Use absolute paths for model loading
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'models')
            
            if not os.path.exists(models_dir):
                print(f"Models directory not found at {models_dir}")
                return
                
            lstm_path = os.path.join(models_dir, 'lstm_model.h5')
            tokenizer_path = os.path.join(models_dir, 'tokenizer.pickle')
            tfidf_path = os.path.join(models_dir, 'tfidf_vectorizer.pickle')
            lr_path = os.path.join(models_dir, 'lr_model.pickle')
            
            missing_files = []
            
            # Load LSTM model if it exists
            if os.path.exists(lstm_path):
                self.lstm_model = tf.keras.models.load_model(lstm_path)
                print("LSTM model loaded successfully")
            else:
                missing_files.append('LSTM model')
            
            # Load tokenizer if it exists
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as handle:
                    self.tokenizer = pickle.load(handle)
                print("Tokenizer loaded successfully")
            else:
                missing_files.append('Tokenizer')
            
            # Load TF-IDF and LR model if they exist
            if os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as handle:
                    self.tfidf_vectorizer = pickle.load(handle)
                print("TF-IDF vectorizer loaded successfully")
            else:
                missing_files.append('TF-IDF vectorizer')
                
            if os.path.exists(lr_path):
                with open(lr_path, 'rb') as handle:
                    self.lr_model = pickle.load(handle)
                print("Logistic Regression model loaded successfully")
            else:
                missing_files.append('Logistic Regression model')
            
            # Check if we have at least one complete model system
            has_lstm_system = self.lstm_model is not None and self.tokenizer is not None
            has_tfidf_system = self.tfidf_vectorizer is not None and self.lr_model is not None
            
            self.is_ready = has_lstm_system or has_tfidf_system
            
            if not self.is_ready:
                print("Warning: No complete model system is available")
                print(f"Missing components: {', '.join(missing_files)}")
                print("Please run main.py to train the models first")
            else:
                print("At least one model is ready for predictions")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_ready = False
    
    def predict_lstm(self, text):
        """
        Make prediction using the LSTM model
        """
        if self.lstm_model is None or self.tokenizer is None:
            return None
        
        try:
            # Clean the text
            cleaned_text = clean_text(text)
            
            # Convert to sequence
            sequence = self.tokenizer.texts_to_sequences([cleaned_text])
            
            # Pad sequence
            padded_sequence = pad_sequences(sequence, maxlen=self.max_len)
            
            # Make prediction
            prediction = self.lstm_model.predict(padded_sequence)[0][0]
            
            # Return prediction and label
            is_spam = bool(prediction > 0.5)
            confidence = float(prediction) if is_spam else float(1 - prediction)
            
            return {
                'is_spam': is_spam,
                'confidence': confidence,
                'model': 'LSTM'
            }
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
            return None
    
    def predict_tfidf(self, text):
        """
        Make prediction using TF-IDF and Logistic Regression model
        """
        if self.tfidf_vectorizer is None or self.lr_model is None:
            return None
        
        try:
            # Clean the text
            cleaned_text = clean_text(text)
            
            # Transform using TF-IDF
            tfidf_vector = self.tfidf_vectorizer.transform([cleaned_text])
            
            # Make prediction
            prediction = self.lr_model.predict(tfidf_vector)[0]
            probabilities = self.lr_model.predict_proba(tfidf_vector)[0]
            
            # Return prediction and label
            is_spam = bool(prediction == 1)
            confidence = float(probabilities[1]) if is_spam else float(probabilities[0])
            
            return {
                'is_spam': is_spam,
                'confidence': confidence,
                'model': 'TF-IDF + LR'
            }
        except Exception as e:
            print(f"Error in TF-IDF prediction: {e}")
            return None
    
    def predict(self, text, model='all'):
        """
        Make predictions using specified model(s)
        """
        if not self.is_ready:
            return {
                'error': 'Models not loaded properly. Please train the models first by running main.py.',
                'status': 'not_ready'
            }
        
        results = {}
        
        if model == 'lstm' or model == 'all':
            lstm_result = self.predict_lstm(text)
            if lstm_result is not None:
                results['lstm'] = lstm_result
        
        if model == 'tfidf' or model == 'all':
            tfidf_result = self.predict_tfidf(text)
            if tfidf_result is not None:
                results['tfidf'] = tfidf_result
        
        # If both models available, create a combined result
        if 'lstm' in results and 'tfidf' in results:
            # Simple ensemble - if both agree, use that result, otherwise use the one with higher confidence
            lstm_spam = results['lstm']['is_spam']
            tfidf_spam = results['tfidf']['is_spam']
            
            if lstm_spam == tfidf_spam:
                combined_spam = lstm_spam
                combined_confidence = (results['lstm']['confidence'] + results['tfidf']['confidence']) / 2
            elif results['lstm']['confidence'] > results['tfidf']['confidence']:
                combined_spam = lstm_spam
                combined_confidence = results['lstm']['confidence']
            else:
                combined_spam = tfidf_spam
                combined_confidence = results['tfidf']['confidence']
            
            results['combined'] = {
                'is_spam': combined_spam,
                'confidence': combined_confidence,
                'model': 'Ensemble (LSTM + TF-IDF)'
            }
        
        if not results:
            return {
                'error': 'No predictions were made. Check if models are properly loaded or if text is valid.',
                'status': 'prediction_failed'
            }
            
        return results

if __name__ == "__main__":
    # Simple test
    predictor = SpamPredictor()
    
    if not predictor.is_ready:
        print("\nWARNING: Models are not loaded. Please run main.py first to train the models.")
        exit(1)
    
    # Test messages
    test_messages = [
        "Hello, how are you doing today?",
        "URGENT: You've WON a FREE PRIZE! Call now to claim your £1000 reward!",
        "Meeting at 3pm tomorrow in the conference room",
        "CONGRATULATIONS! You are selected for FREE entry to win £250,000.00 cash! Call now 09064019788"
    ]
    
    for message in test_messages:
        print(f"\nMessage: {message}")
        results = predictor.predict(message)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            continue
            
        for model_name, result in results.items():
            print(f"Model: {model_name}")
            print(f"Is Spam: {'Yes' if result['is_spam'] else 'No'}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
