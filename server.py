from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, render_template
from flask_cors import CORS
import os
import sqlite3
import jwt
from datetime import datetime, timedelta
import json
import base64
from PIL import Image
import io
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Secret key for JWT
SECRET_KEY = 'your-secret-key'

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize database
def init_db():
    conn = sqlite3.connect('surveillance.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    
    # Create locations table
    c.execute('''CREATE TABLE IF NOT EXISTS locations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  status TEXT NOT NULL,
                  last_activity DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create alerts table
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  location_id INTEGER,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  type TEXT NOT NULL,
                  description TEXT,
                  FOREIGN KEY (location_id) REFERENCES locations(id))''')
    
    # Insert default admin user if not exists
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                 ('admin', 'admin123'))
    except sqlite3.IntegrityError:
        pass
    
    # Insert default locations if not exists
    locations = [
        ('Main Entrance', 'unlocked'),
        ('Back Door', 'locked'),
        ('Server Room', 'locked'),
        ('Storage Room', 'locked')
    ]
    
    for location in locations:
        try:
            c.execute("INSERT INTO locations (name, status) VALUES (?, ?)", location)
        except sqlite3.IntegrityError:
            pass
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

def generate_token(username):
    payload = {
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload['username']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def process_frame(frame_data, location_id):
    try:
        # Decode base64 image
        image_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        alerts = []
        if len(faces) > 0:
            # Person detected
            conn = sqlite3.connect('surveillance.db')
            c = conn.cursor()
            c.execute("INSERT INTO alerts (location_id, type, description) VALUES (?, ?, ?)",
                     (location_id, 'person_detected', f'{len(faces)} person(s) detected'))
            conn.commit()
            conn.close()
            
            alerts.append({
                'type': 'person_detected',
                'timestamp': datetime.now().isoformat(),
                'location_id': location_id,
                'count': len(faces)
            })
        
        return {
            'faces_detected': len(faces),
            'alerts': alerts
        }
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        conn = sqlite3.connect('surveillance.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                 (username, password))
        user = c.fetchone()
        conn.close()
        
        if user:
            token = generate_token(username)
            return jsonify({'token': token})
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/surveillance')
def surveillance():
    token = request.args.get('token')
    if not token or not verify_token(token):
        return redirect(url_for('login'))
    return render_template('surveillance.html')

@app.route('/api/locations')
def get_locations():
    token = request.args.get('token')
    if not token or not verify_token(token):
        return jsonify({'error': 'Unauthorized'}), 401
    
    current_location = request.args.get('current_location')
    
    conn = sqlite3.connect('surveillance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM locations")
    locations = [{'id': row[0], 'name': row[1], 'status': row[2], 'last_activity': row[3]} for row in c.fetchall()]
    conn.close()
    
    if current_location:
        # Update location status and last activity
        conn = sqlite3.connect('surveillance.db')
        c = conn.cursor()
        c.execute("UPDATE locations SET status = 'active', last_activity = CURRENT_TIMESTAMP WHERE id = ?", 
                 (current_location,))
        conn.commit()
        conn.close()
    
    return jsonify(locations)

@app.route('/api/process_frame', methods=['POST'])
def process_frame_endpoint():
    try:
        token = request.args.get('token')
        if not token or not verify_token(token):
            return jsonify({'error': 'Unauthorized'}), 401

        data = request.get_json()
        if not data or 'image' not in data or 'location_id' not in data:
            return jsonify({'error': 'Missing required data'}), 400
        
        result = process_frame(data['image'], data['location_id'])
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Failed to process frame'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    token = request.args.get('token')
    if not token or not verify_token(token):
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('surveillance.db')
    c = conn.cursor()
    c.execute("SELECT a.*, l.name as location_name FROM alerts a JOIN locations l ON a.location_id = l.id ORDER BY a.timestamp DESC LIMIT 10")
    alerts = [{
        'id': row[0],
        'location_id': row[1],
        'timestamp': row[2],
        'type': row[3],
        'description': row[4],
        'location_name': row[5]
    } for row in c.fetchall()]
    conn.close()
    
    return jsonify(alerts)

if __name__ == '__main__':
    app.run(debug=True) 