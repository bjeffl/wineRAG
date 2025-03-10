# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
import uuid
import pandas as pd
from recommendation_engine import ProductRecommendationEngine
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'static/images/products'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# CSV file path
csv_path = 'lcbo_wines_updated.csv'
absolute_path = os.path.abspath(csv_path)
print(f"Looking for CSV at: {absolute_path}")
print(f"File exists: {os.path.exists(absolute_path)}")

# Try to read the CSV file directly to verify it's accessible
try:
    print("Reading CSV file directly:")
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded {len(df)} rows with pandas")
    print(f"Columns: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"First row sample: {df.iloc[0][['title', 'category', 'price']].to_dict()}")
except Exception as e:
    print(f"Error reading CSV directly: {str(e)}")

# Initialize recommendation engine
engine = ProductRecommendationEngine()

# Reset ChromaDB collections to start fresh
print("Resetting ChromaDB collections...")
try:
    engine.client.delete_collection(name="products")
    print("Deleted products collection")
except Exception as e:
    print(f"No products collection to delete or error: {str(e)}")

try:
    engine.client.delete_collection(name="user_preferences")
    print("Deleted user_preferences collection")
except Exception as e:
    print(f"No user_preferences collection to delete or error: {str(e)}")

# Recreate collections
engine.product_collection = engine.client.create_collection(
    name="products",
    metadata={"hnsw:space": "cosine"}
)
engine.user_collection = engine.client.create_collection(
    name="user_preferences",
    metadata={"hnsw:space": "cosine"}
)

# Clear existing products list
engine.products = []

# Load products from CSV
if os.path.exists(csv_path):
    print("Found CSV file, attempting to load...")
    count = engine.load_products_from_csv(csv_path)
    print(f"Loaded {count} wine products from CSV")
else:
    print(f"CSV file not found at: {os.path.abspath(csv_path)}")

# Add sample products if no products were loaded
if len(engine.products) == 0:
    print("No products loaded, adding sample products...")
    sample_products = [
        {
            'name': 'Cabernet Sauvignon',
            'description': 'Bold red wine with rich flavors of blackcurrant, cedar, and tobacco. Full-bodied with firm tannins.',
            'price': 24.99,
            'category': 'Red Wine',
            'tags': 'full-bodied,tannic,blackcurrant,cedar',
            'country': 'France',
            'alcohol_content': '14.5'
        },
        {
            'name': 'Chardonnay',
            'description': 'Versatile white wine with notes of apple, pear, and vanilla. Medium to full-bodied with a buttery finish.',
            'price': 19.99,
            'category': 'White Wine',
            'tags': 'medium-bodied,apple,vanilla,buttery',
            'country': 'United States',
            'alcohol_content': '13.0'
        },
        {
            'name': 'Pinot Noir',
            'description': 'Elegant red wine with cherry, raspberry, and earthy mushroom notes. Light to medium-bodied with silky tannins.',
            'price': 27.99,
            'category': 'Red Wine',
            'tags': 'light-bodied,cherry,raspberry,silky',
            'country': 'France',
            'alcohol_content': '13.5'
        }
    ]
    
    for product in sample_products:
        engine.add_product(product)
    
    print(f"Added {len(sample_products)} sample wine products")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


@app.route('/')
def index():
    # Get or create a user ID
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # Get viewed product IDs from session
    viewed_product_ids = session.get('viewed_product_ids', [])
    
    # Get recommendations
    products = engine.get_recommendations(
        user_id=session['user_id'],
        n_results=3,
        excluded_ids=viewed_product_ids
    )
    
    # Add product IDs to viewed list
    for product in products:
        if product['id'] not in viewed_product_ids:
            viewed_product_ids.append(product['id'])
    
    session['viewed_product_ids'] = viewed_product_ids
    
    # Get user feedback for display
    user_feedback = {}
    if session['user_id'] in engine.feedback.get('users', {}):
        user_feedback = {
            'likes': engine.feedback['users'][session['user_id']]['likes'],
            'dislikes': engine.feedback['users'][session['user_id']]['dislikes']
        }
    
    return render_template('index.html', products=products, user_feedback=user_feedback)


@app.route('/feedback', methods=['POST'])
def product_feedback():
    data = request.json
    product_id = data.get('product_id')
    feedback_type = data.get('feedback')  # 'up' or 'down'
    
    if not product_id or feedback_type not in ['up', 'down']:
        return jsonify({'error': 'Valid product ID and feedback type are required'}), 400
    
    # Process feedback
    engine.add_feedback(session['user_id'], product_id, feedback_type)
    
    # Get next batch of recommendations
    viewed_product_ids = session.get('viewed_product_ids', [])
    new_products = engine.get_recommendations(
        user_id=session['user_id'],
        n_results=3,
        excluded_ids=viewed_product_ids
    )
    
    # Update viewed products
    for product in new_products:
        if product['id'] not in viewed_product_ids:
            viewed_product_ids.append(product['id'])
    
    session['viewed_product_ids'] = viewed_product_ids
    
    return jsonify({
        'success': True,
        'message': 'Feedback recorded successfully',
        'new_products': new_products
    })


@app.route('/reset')
def reset_recommendations():
    session['viewed_product_ids'] = []
    return redirect(url_for('index'))


@app.route('/admin')
def admin():
    products = engine.products
    return render_template('admin.html', products=products)


@app.route('/admin/add', methods=['GET', 'POST'])
def add_product():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        price = request.form.get('price')
        category = request.form.get('category')
        tags = request.form.get('tags')
        
        if not name or not price or not category:
            flash('Name, price, and category are required', 'error')
            return redirect(url_for('add_product'))
        
        # Handle image upload
        image_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                image_path = f"/static/images/products/{unique_filename}"
        
        # Create product object
        product = {
            'name': name,
            'description': description,
            'price': float(price),
            'category': category,
            'tags': tags,
            'image': image_path
        }
        
        # Add to recommendation engine
        engine.add_product(product)
        
        flash('Product added successfully', 'success')
        return redirect(url_for('admin'))
    
    return render_template('add_product.html')


@app.route('/admin/delete/<product_id>', methods=['POST'])
def delete_product(product_id):
    engine.delete_product(product_id)
    flash('Product deleted successfully', 'success')
    return redirect(url_for('admin'))


if __name__ == '__main__':
    app.run(debug=True)