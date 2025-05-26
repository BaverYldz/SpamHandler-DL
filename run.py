import os
import sys
import subprocess
import importlib.util
import webbrowser
import time

def import_or_install(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
def check_requirements():
    """Check if required packages are installed"""
    required_packages = ["pandas", "numpy", "scikit-learn", "nltk", "flask", "tensorflow"]
    for package in required_packages:
        import_or_install(package)
    print("All required packages are installed.")

def check_for_files():
    """Check if necessary files exist"""
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Check if source files exist
    required_files = [
        os.path.join(src_dir, 'preprocessing.py'),
        os.path.join(src_dir, 'main.py'),
        os.path.join(src_dir, 'predict.py')
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Required file {file} not found.")
            return False
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at {data_dir}")
        
    return True

def run_data_helper():
    """Run data helper script to prepare the dataset"""
    data_helper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'data_helper.py')
    
    if not os.path.exists(data_helper_path):
        print("Creating data helper script...")
        with open(data_helper_path, 'w') as f:
            f.write('''# filepath: c:\\Users\\omerb\\Desktop\\FakeDetector\\src\\data_helper.py

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
                lines = content.split('\\n')
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
''')
    
    print("Running data helper script...")
    result = subprocess.run([sys.executable, data_helper_path], capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"Error running data helper script: {result.stderr}")
        return False
    
    return True

def run_preprocessing():
    """Run the preprocessing script"""
    preprocessing_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'preprocessing.py')
    
    print("\nRunning preprocessing script...")
    result = subprocess.run([sys.executable, preprocessing_path], capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"Error running preprocessing script: {result.stderr}")
        return False
    
    return True

def run_main():
    """Run the main training script"""
    main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'main.py')
    
    print("\nRunning model training script...")
    result = subprocess.run([sys.executable, main_path], capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"Error running main script: {result.stderr}")
        return False
    
    return True

def run_flask(open_browser=True):
    """Run the Flask web application"""
    flask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_flask.py')
    
    print("\nStarting Flask web application...")
    
    # Start Flask in a separate process
    flask_process = subprocess.Popen([sys.executable, flask_path])
    
    if open_browser:
        # Wait a bit for Flask to start up
        time.sleep(2)
        try:
            print("Opening web browser to http://localhost:5000")
            webbrowser.open('http://localhost:5000')
        except Exception as e:
            print(f"Failed to open browser: {e}")
    
    try:
        # Wait for user to press Ctrl+C
        flask_process.wait()
    except KeyboardInterrupt:
        print("Stopping Flask application...")
        flask_process.terminate()
        flask_process.wait()
    
    return True

def setup_static_files():
    """Ensure static files and templates are in place"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure directories exist
    os.makedirs(os.path.join(base_dir, 'static', 'css'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'static', 'js'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'templates'), exist_ok=True)
    
    # Check if the template file exists
    if not os.path.exists(os.path.join(base_dir, 'templates', 'index.html')):
        print("Warning: index.html template file not found. Run may not work correctly.")

    # Check for static files
    if not os.path.exists(os.path.join(base_dir, 'static', 'css', 'style.css')):
        print("Warning: CSS file not found. Run may not work correctly.")
    
    if not os.path.exists(os.path.join(base_dir, 'static', 'js', 'main.js')):
        print("Warning: JavaScript file not found. Run may not work correctly.")
    
    return True

def print_menu():
    """Print the main menu"""
    print("\n=== Fake Message Detector ===")
    print("1. Prepare data")
    print("2. Preprocess data")
    print("3. Train model")
    print("4. Run web application")
    print("5. Run full pipeline (1-4)")
    print("0. Exit")
    
    choice = input("Enter your choice (0-5): ")
    return choice

def main():
    """Main function"""
    print("Checking requirements...")
    check_requirements()
    
    if not check_for_files():
        print("Error: Some required files are missing.")
        return
    
    # Ensure all static files are set up
    setup_static_files()
    
    while True:
        choice = print_menu()
        
        if choice == '1':
            run_data_helper()
        elif choice == '2':
            run_preprocessing()
        elif choice == '3':
            run_main()
        elif choice == '4':
            run_flask()
        elif choice == '5':
            run_data_helper()
            run_preprocessing()
            run_main()
            run_flask()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
