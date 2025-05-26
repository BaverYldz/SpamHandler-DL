import os
import sys
import subprocess

def run_model_training():
    """Run just the model training script"""
    # Get the absolute path to the main.py script
    main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'main.py')
    
    # Make sure the models directory exists
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory at {models_dir}")
    
    # Run the model training script
    print("\nRunning model training script...")
    result = subprocess.run([sys.executable, main_path], capture_output=False)
    
    if result.returncode != 0:
        print(f"Error running main script")
        return False
    
    print("\nModel training completed. You can now run the Flask application.")
    return True

if __name__ == "__main__":
    run_model_training()
