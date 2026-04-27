from flask import Flask, render_template, request, redirect, url_for, session, flash, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import json
import os
from PIL import Image, ImageOps
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import uuid
from datetime import datetime
from functools import wraps
from time import time
import csv
import base64
import uuid
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default-fallback-key-change-this')

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_CACHE_DIR = 'model_cache'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['MODEL_CACHE_DIR'] = MODEL_CACHE_DIR

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Users file
USERS_FILE = 'users.json'

# Rate limiting storage
rate_limits = {}

# Available models (optional feature)
MODELS = {
    'blip-base': "Salesforce/blip-image-captioning-base",
    'blip-large': "Salesforce/blip-image-captioning-large",
}

# Initialize BLIP model for image captioning
print("Loading BLIP model...")
try:
    processor = BlipProcessor.from_pretrained(
        MODELS['blip-base'],
        cache_dir=MODEL_CACHE_DIR
    )
    model = BlipForConditionalGeneration.from_pretrained(
        MODELS['blip-base'],
        cache_dir=MODEL_CACHE_DIR
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check your internet connection and try again.")
    exit(1)

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(filepath):
    """Check if file size is within limits"""
    file_size = os.path.getsize(filepath)
    max_size = 16 * 1024 * 1024  # 16MB
    if file_size > max_size:
        os.remove(filepath)  # Clean up
        return False, f"File too large. Max size: {max_size//(1024*1024)}MB"
    return True, "OK"

def preprocess_image(image_path):
    """Optimize image for processing"""
    try:
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (optional)
        max_size = 800
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save optimized image
        image.save(image_path, optimize=True, quality=85)
        return True
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return False

def generate_multiple_captions(image_path, num_captions=3):
    """Generate multiple caption variations"""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        
        captions = []
        for i in range(num_captions):
            out = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=0.8 + (i * 0.1),  # Vary temperature
                do_sample=True
            )
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
        
        return captions
    except Exception as e:
        print(f"Error generating multiple captions: {e}")
        return ["Unable to generate caption"] * num_captions

def generate_detailed_caption(image_path):
    """Generate longer, more descriptive caption"""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        out = model.generate(
            **inputs,
            max_length=100,
            num_beams=5,
            min_length=20
        )
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating detailed caption: {e}")
        return "Unable to generate detailed caption"

def generate_caption(image_path, caption_type='standard'):
    """Generate caption for the given image"""
    try:
        if caption_type == 'detailed':
            return generate_detailed_caption(image_path)
        elif caption_type == 'multiple':
            return generate_multiple_captions(image_path)
        else:
            # Standard caption
            image = Image.open(image_path).convert('RGB')
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs, max_length=50)
            return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Unable to generate caption for this image."

def save_user_history(username, image_filename, caption, caption_type='standard'):
    """Save image history for user"""
    users = load_users()
    if username in users:
        if 'history' not in users[username]:
            users[username]['history'] = []
        
        users[username]['history'].append({
            'id': str(uuid.uuid4()),
            'image': image_filename,
            'caption': caption if isinstance(caption, str) else ', '.join(caption),
            'caption_type': caption_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        save_users(users)

# Rate limiting decorator
def rate_limit(limit=5, per=3600):  # 5 uploads per hour
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            username = session.get('username')
            if username:
                key = f"{username}:upload"
                now = time()
                
                if key in rate_limits:
                    # Clean old entries
                    rate_limits[key] = [t for t in rate_limits[key] if now - t < per]
                    
                    if len(rate_limits[key]) >= limit:
                        flash(f'Rate limit exceeded. Max {limit} uploads per hour.', 'error')
                        return redirect(url_for('dashboard'))
                    
                    rate_limits[key].append(now)
                else:
                    rate_limits[key] = [now]
            
            return f(*args, **kwargs)
        return wrapped
    return decorator

@app.route('/')
def index():
    """Redirect to login page"""
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        users = load_users()
        
        if username in users and check_password_hash(users[username]['password'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        users = load_users()
        
        if username in users:
            flash('Username already exists!', 'error')
        else:
            # Create new user
            users[username] = {
                'password': generate_password_hash(password),
                'email': email,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'history': []
            }
            save_users(users)
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    """User dashboard"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    users = load_users()
    user_history = users.get(username, {}).get('history', [])
    
    # Sort history by timestamp (newest first)
    user_history.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template('dashboard.html', username=username, history=user_history)

@app.route('/upload', methods=['POST'])
@rate_limit(limit=10, per=3600)  # 10 uploads per hour
def upload_file():
    """Handle image upload and caption generation"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if 'file' not in request.files:
        flash('No file selected!', 'error')
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    caption_type = request.form.get('caption_type', 'standard')
    
    if file.filename == '':
        flash('No file selected!', 'error')
        return redirect(url_for('dashboard'))
    
    if file and allowed_file(file.filename):
        # Secure filename and save
        filename = secure_filename(file.filename)
        # Add unique identifier to prevent overwriting
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Validate file size
        is_valid, message = validate_file_size(filepath)
        if not is_valid:
            flash(message, 'error')
            return redirect(url_for('dashboard'))
        
        # Preprocess image
        preprocess_image(filepath)
        
        # Generate caption based on type
        if caption_type == 'multiple':
            captions = generate_caption(filepath, caption_type='multiple')
            caption_display = captions
            # Save to user history (join multiple captions)
            save_user_history(session['username'], unique_filename, captions, caption_type)
        else:
            caption = generate_caption(filepath, caption_type)
            caption_display = caption
            # Save to user history
            save_user_history(session['username'], unique_filename, caption, caption_type)
        
        return render_template('result.html', 
                             image_filename=unique_filename, 
                             caption=caption_display,
                             caption_type=caption_type)
    
    flash('Invalid file type! Allowed types: png, jpg, jpeg, gif, bmp', 'error')
    return redirect(url_for('dashboard'))

@app.route('/history/<history_id>')
def view_history(history_id):
    """View specific history item"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    users = load_users()
    user_history = users.get(username, {}).get('history', [])
    
    # Find the specific history item
    history_item = next((item for item in user_history if item['id'] == history_id), None)
    
    if history_item:
        caption = history_item['caption']
        if history_item.get('caption_type') == 'multiple':
            caption = caption.split(', ') if isinstance(caption, str) else caption
        
        return render_template('result.html', 
                             image_filename=history_item['image'],
                             caption=caption,
                             caption_type=history_item.get('caption_type', 'standard'),
                             timestamp=history_item['timestamp'])
    else:
        flash('History item not found!', 'error')
        return redirect(url_for('dashboard'))

@app.route('/export_history')
def export_history():
    """Export user history as CSV"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    users = load_users()
    history = users.get(username, {}).get('history', [])
    
    # Create CSV
    output = make_response()
    output.headers['Content-Disposition'] = f'attachment; filename={username}_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    output.headers['Content-type'] = 'text/csv'
    
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'Caption Type', 'Caption', 'Image Filename'])
    
    for item in history:
        writer.writerow([
            item['timestamp'], 
            item.get('caption_type', 'standard'), 
            item['caption'], 
            item['image']
        ])
    
    flash('History exported successfully!', 'success')
    return output

@app.route('/delete_history/<history_id>', methods=['POST'])
def delete_history(history_id):
    """Delete a specific history item"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    users = load_users()
    
    if username in users and 'history' in users[username]:
        # Find and remove the history item
        users[username]['history'] = [
            item for item in users[username]['history'] 
            if item['id'] != history_id
        ]
        save_users(users)
        flash('History item deleted successfully!', 'success')
    
    return redirect(url_for('dashboard'))

@app.route('/clear_all_history', methods=['POST'])
def clear_all_history():
    """Clear all history for the user"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    users = load_users()
    
    if username in users:
        users[username]['history'] = []
        save_users(users)
        flash('All history cleared successfully!', 'success')
    
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    """Logout user"""
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))
@app.route('/capture_image', methods=['POST'])
def capture_image():
    """Handle camera captured image"""

    if 'username' not in session:
        return redirect(url_for('login'))

    image_data = request.form.get('imageData')

    if not image_data:
        flash("No image captured!", "error")
        return redirect(url_for('dashboard'))

    try:
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        # preprocess image
        preprocess_image(filepath)

        # generate caption
        caption = generate_caption(filepath)

        # save history
        save_user_history(session['username'], filename, caption, 'standard')

        return render_template(
            'result.html',
            image_filename=filename,
            caption=caption,
            caption_type='standard'
        )

    except Exception as e:
        print("Camera capture error:", e)
        flash("Error processing captured image", "error")
        return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
