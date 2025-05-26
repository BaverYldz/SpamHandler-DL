from flask import Flask, request, render_template, jsonify, url_for
import os
import sys
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our predictor
try:
    from src.predict import SpamPredictor
    predictor = SpamPredictor()
except Exception as e:
    logger.error(f"Error importing predictor: {str(e)}")
    predictor = None

# Initialize Flask app
app = Flask(__name__)

# Stats storage
stats_file = os.path.join(os.path.dirname(__file__), 'stats.json')

def load_stats():
    """Load usage statistics"""
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading stats: {str(e)}")
    return {'total': 0, 'spam': 0, 'ham': 0, 'last_updated': datetime.now().isoformat()}

def save_stats(stats):
    """Save usage statistics"""
    try:
        stats['last_updated'] = datetime.now().isoformat()
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
    except Exception as e:
        logger.error(f"Error saving stats: {str(e)}")

# Initialize or load stats
stats = load_stats()

@app.route('/')
def home():
    """Serve the home page"""
    model_status = "ready" if predictor and predictor.is_ready else "not_ready"
    return render_template('index.html', model_status=model_status)

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on submitted text"""
    # Get text from request
    text = request.form.get('text', '')
    
    if not text:
        return jsonify({
            'error': 'No text provided'
        }), 400
    
    # Check if predictor is available
    if not predictor:
        return jsonify({
            'error': 'Predictor not initialized properly. Please restart the server.',
            'status': 'not_ready'
        }), 503
    
    # Check if models are ready
    if not predictor.is_ready:
        return jsonify({
            'error': 'Models are not loaded properly. Please run training first.',
            'status': 'not_ready'
        }), 503
    
    # Get model preference
    model = request.form.get('model', 'all')
    
    # Make prediction
    try:
        logger.info(f"Making prediction for text: '{text[:30]}...' using model: {model}")
        results = predictor.predict(text, model=model)
        
        # Update stats if prediction was successful
        if not 'error' in results:
            global stats
            stats['total'] += 1
            
            # Find if spam was detected in any result
            is_spam = False
            if 'combined' in results and results['combined']['is_spam']:
                is_spam = True
            elif 'lstm' in results and results['lstm']['is_spam']:
                is_spam = True
            elif 'tfidf' in results and results['tfidf']['is_spam']:
                is_spam = True
                
            if is_spam:
                stats['spam'] += 1
            else:
                stats['ham'] += 1
                
            save_stats(stats)
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions"""
    # Get JSON data
    try:
        data = request.get_json()
    except Exception:
        return jsonify({
            'error': 'Invalid JSON data'
        }), 400
    
    if not data or 'text' not in data:
        return jsonify({
            'error': 'No text provided in request'
        }), 400
    
    # Check if models are ready
    if not predictor or not predictor.is_ready:
        return jsonify({
            'error': 'Models are not loaded properly. Please run training first.',
            'status': 'not_ready'
        }), 503
    
    # Get model preference
    model = data.get('model', 'all')
    
    # Make prediction
    try:
        results = predictor.predict(data['text'], model=model)
        return jsonify(results)
    except Exception as e:
        logger.error(f"API error during prediction: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/status')
def status():
    """Return the status of the models and usage statistics"""
    if not predictor:
        return jsonify({
            'ready': False,
            'models': {},
            'stats': stats
        })
    
    return jsonify({
        'ready': predictor.is_ready,
        'models': {
            'lstm': predictor.lstm_model is not None and predictor.tokenizer is not None,
            'tfidf': predictor.tfidf_vectorizer is not None and predictor.lr_model is not None,
        },
        'stats': stats
    })

@app.route('/reset-stats')
def reset_stats():
    """Reset the usage statistics"""
    global stats
    stats = {'total': 0, 'spam': 0, 'ham': 0, 'last_updated': datetime.now().isoformat()}
    save_stats(stats)
    return jsonify({'status': 'success', 'message': 'Statistics reset successfully'})

# Function to check models directory
def check_models():
    """Check if model files exist and print warning if not"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_files_exist = (
        os.path.exists(os.path.join(models_dir, 'lstm_model.h5')) and 
        os.path.exists(os.path.join(models_dir, 'tokenizer.pickle')) and
        os.path.exists(os.path.join(models_dir, 'tfidf_vectorizer.pickle')) and
        os.path.exists(os.path.join(models_dir, 'lr_model.pickle'))
    )
    
    return model_files_exist

if __name__ == '__main__':
    print("Starting Fake Message Detector Web Application...")
    
    # Create necessary directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.join('static', 'css'), exist_ok=True)
    os.makedirs(os.path.join('static', 'js'), exist_ok=True)
    
    # Check if models exist
    model_files_exist = check_models()
    
    if not model_files_exist:
        print("Warning: Some model files are missing. Please train the models first by running:")
        print("python src/main.py")
        print("\nThe application will still run, but predictions won't work until models are trained.")
    else:
        print("All model files found. Application is ready for predictions.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
