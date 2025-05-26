import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_text(text):
    """
    Clean and preprocess text data
    """
    # Check for NaN or non-string values
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (#topic)
    text = re.sub(r'#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def preprocess_dataset():
    """
    Load, clean, and preprocess the Twitter spam dataset
    """
    try:
        # Initialize data directories
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created data directory at {data_dir}")
        
        # Load dataset
        print("Loading dataset...")
        input_path = os.path.join(data_dir, 'twitter_spam_data.csv')
        
        if not os.path.exists(input_path):
            print(f"Dataset not found at {input_path}. Please make sure the file exists.")
            
            # Try to find an alternative file
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    print(f"Found CSV file: {file}. Will try to use this file instead.")
                    input_path = os.path.join(data_dir, file)
                    break
            
            if not os.path.exists(input_path):
                print("No CSV files found. Please run data_helper.py first.")
                return
        
        df = pd.read_csv(input_path)
        
        # Check if dataset is empty
        if df.empty:
            print("Dataset is empty.")
            return
        
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Display first few rows to understand the structure
        print("First few rows of the dataset:")
        print(df.head())
        
        # Check for column names and rename if needed
        if 'class' not in df.columns and 'label' in df.columns:
            df = df.rename(columns={'label': 'class'})
        
        # Make sure required columns exist
        if 'class' not in df.columns or 'text' not in df.columns:
            print(f"Error: Required columns 'class' and 'text' not found in the dataset.")
            print(f"Available columns: {df.columns.tolist()}")
            return
        
        # Convert 'class' to numeric if it's not
        if df['class'].dtype == 'object':
            df['class'] = df['class'].apply(lambda x: 1 if x in ['spam', 'fake', '1'] else 0)
        
        # Check for missing values in text column
        missing_texts = df['text'].isna().sum()
        if missing_texts > 0:
            print(f"Warning: Found {missing_texts} missing values in 'text' column.")
            df['text'] = df['text'].fillna("")  # Fill NaN values with empty string
        
        # Apply text cleaning to the 'text' column
        print("Cleaning text data...")
        df['clean_text'] = df['text'].apply(clean_text)
        
        # Save the preprocessed data
        print("Saving preprocessed data...")
        output_path = os.path.join(data_dir, 'processed_twitter_spam.csv')
        df.to_csv(output_path, index=False)
        print(f"Preprocessing complete! Saved to {output_path}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    preprocess_dataset()
