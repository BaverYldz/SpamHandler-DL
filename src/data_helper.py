import os
import pandas as pd
import shutil

def setup_data_directory():
    """
    Create the data directory if it doesn't exist
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    if not os.path.exists(data_dir):
        print(f"Creating data directory at {data_dir}")
        os.makedirs(data_dir)
    else:
        print(f"Data directory already exists at {data_dir}")
    
    return data_dir

def convert_csv_to_dataset(input_path, output_path):
    """
    Convert CSV file to the expected dataset format
    """
    try:
        # Check if the input file exists
        if not os.path.exists(input_path):
            print(f"Input file not found at {input_path}")
            return False
        
        print(f"Loading data from {input_path}...")
        
        # Read the CSV file
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        else:
            # Assuming it's a plaintext file
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                if len(lines) > 0 and ',' in lines[0]:
                    # Try to parse as CSV
                    df = pd.read_csv(input_path)
                else:
                    print("File format not recognized")
                    return False
        
        print(f"Data loaded with shape: {df.shape}")
        
        # Check if the dataset has the required columns
        if 'class' not in df.columns and 'text' not in df.columns:
            print("Dataset missing required columns 'class' and 'text'")
            print("Available columns:", df.columns.tolist())
            
            # Try to identify suitable columns
            text_columns = [col for col in df.columns if 'text' in col.lower()]
            class_columns = [col for col in df.columns if 'class' in col.lower()]
            
            if text_columns and class_columns:
                print(f"Using {text_columns[0]} as 'text' and {class_columns[0]} as 'class'")
                df = df.rename(columns={text_columns[0]: 'text', class_columns[0]: 'class'})
            else:
                # If we can't find appropriate columns, create a minimal dataset
                print("Creating a minimal dataset with the first 2 columns")
                if len(df.columns) >= 2:
                    df = df.iloc[:, :2]
                    df.columns = ['class', 'text']
                else:
                    print("Not enough columns to create a dataset")
                    return False
        
        # Make sure 'class' column is numeric (0 for ham, 1 for spam)
        if df['class'].dtype == 'object':
            df['class'] = df['class'].apply(lambda x: 1 if str(x).strip() in ['1', 'spam', 'yes', 'true'] else 0)
        
        # Save the formatted dataset
        print(f"Saving formatted dataset to {output_path}")
        df.to_csv(output_path, index=False)
        
        print(f"Dataset successfully converted and saved to {output_path}")
        print(f"Dataset contains {len(df)} rows with {len(df[df['class'] == 1])} spam messages")
        
        return True
    
    except Exception as e:
        print(f"Error converting dataset: {e}")
        return False

def prepare_dataset():
    """
    Prepare the dataset for preprocessing
    """
    data_dir = setup_data_directory()
    
    # Check for existing data files
    data_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    
    input_path = None
    for file in data_files:
        if 'twitter_spam_data' in file.lower() or 'spam' in file.lower():
            input_path = os.path.join(data_dir, file)
            break
    
    if not input_path:
        print("No suitable dataset file found in the data directory.")
        print("Available files:", data_files)
        return False
    
    output_path = os.path.join(data_dir, 'twitter_spam_data.csv')
    
    # If the input and output paths are the same, make a backup
    if input_path == output_path:
        backup_path = os.path.join(data_dir, 'twitter_spam_data_backup.csv')
        print(f"Creating backup of original file to {backup_path}")
        shutil.copy2(input_path, backup_path)
    
    result = convert_csv_to_dataset(input_path, output_path)
    
    if result:
        print("Dataset preparation completed successfully.")
        return True
    else:
        print("Failed to prepare the dataset.")
        return False

if __name__ == "__main__":
    prepare_dataset()
