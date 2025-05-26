import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

def train_lstm_model():
    """
    Train LSTM model for spam detection
    """
    try:
        # Use absolute paths instead of relative paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models')
        data_dir = os.path.join(base_dir, 'data')
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"Created models directory at {models_dir}")

        # Load preprocessed data
        print("Loading preprocessed data...")
        data_path = os.path.join(data_dir, 'processed_twitter_spam.csv')
        
        if not os.path.exists(data_path):
            print(f"Error: Preprocessed data not found at {data_path}. Please run preprocessing.py first.")
            return
            
        df = pd.read_csv(data_path)
        
        # Check if dataset is loaded correctly
        if df.empty:
            print("Dataset is empty.")
            return
        
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Verify columns
        required_columns = ['class', 'clean_text']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found in the dataset.")
                return
        
        # Handle any remaining NaN values
        if df['clean_text'].isna().sum() > 0:
            print(f"Warning: Found {df['clean_text'].isna().sum()} missing values in 'clean_text'. Filling with empty strings.")
            df['clean_text'] = df['clean_text'].fillna("")
        
        # Extract features and target
        X = df['clean_text'].values
        y = df['class'].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Prepare text data for LSTM
        max_words = 5000  # Max vocabulary size
        max_len = 100     # Max sequence length
        
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)
        
        # Save the tokenizer
        tokenizer_path = os.path.join(models_dir, 'tokenizer.pickle')
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer saved to {tokenizer_path}")
        
        # Convert texts to sequences
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences to ensure same length
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
        
        # Build LSTM model
        print("Building and training LSTM model...")
        model = Sequential()
        model.add(Embedding(max_words, 128, input_length=max_len))
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        print(model.summary())
        
        # Train the model
        history = model.fit(
            X_train_pad, y_train,
            epochs=5,
            batch_size=64,
            validation_split=0.1,
            verbose=1
        )
        
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        
        # Make predictions
        y_pred_prob = model.predict(X_test_pad)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall: {recall*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        model_path = os.path.join(models_dir, 'lstm_model.h5')
        model.save(model_path)
        print(f"Model saved successfully to {model_path}")
        
        # Also train a simpler model using TF-IDF for comparison
        train_tfidf_model(X_train, X_test, y_train, y_test, models_dir)
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()

def train_tfidf_model(X_train, X_test, y_train, y_test, models_dir):
    """
    Train a TF-IDF based model for comparison
    """
    from sklearn.linear_model import LogisticRegression
    
    try:
        print("\nTraining TF-IDF + Logistic Regression model for comparison...")
        
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        # Save the vectorizer
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pickle')
        with open(vectorizer_path, 'wb') as handle:
            pickle.dump(tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"TF-IDF vectorizer saved to {vectorizer_path}")
        
        # Train logistic regression
        lr_model = LogisticRegression()
        lr_model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred_lr = lr_model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred_lr)
        precision = precision_score(y_test, y_pred_lr)
        recall = recall_score(y_test, y_pred_lr)
        
        print(f"TF-IDF + LR Model - Accuracy: {accuracy*100:.2f}%")
        print(f"TF-IDF + LR Model - Precision: {precision*100:.2f}%")
        print(f"TF-IDF + LR Model - Recall: {recall*100:.2f}%")
        
        # Save the model
        lr_model_path = os.path.join(models_dir, 'lr_model.pickle')
        with open(lr_model_path, 'wb') as handle:
            pickle.dump(lr_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Logistic Regression model saved to {lr_model_path}")
            
    except Exception as e:
        print(f"Error during TF-IDF model training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_lstm_model()