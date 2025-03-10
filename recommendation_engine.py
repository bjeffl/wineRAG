# recommendation_engine.py
import os
import json
import chromadb
import numpy as np
import uuid
import csv
import os

from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from datetime import datetime

class ProductRecommendationEngine:
    def __init__(self, db_path="chroma_db", products_path="data/products.json", feedback_path="data/feedback.json"):
        # Ensure directories exist
        os.makedirs(os.path.dirname(products_path), exist_ok=True)
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
        os.makedirs(db_path, exist_ok=True)
        
        self.db_path = db_path
        self.products_path = products_path
        self.feedback_path = feedback_path
        
        # Initialize embedding model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize collections
        try:
            self.product_collection = self.client.get_collection(name="products")
        except Exception as e:
            print(f"Creating new products collection: {str(e)}")
            self.product_collection = self.client.create_collection(
                name="products",
                metadata={"hnsw:space": "cosine"}
            )
            
        try:
            self.user_collection = self.client.get_collection(name="user_preferences")
        except Exception as e:
            print(f"Creating new user_preferences collection: {str(e)}")
            self.user_collection = self.client.create_collection(
                name="user_preferences",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Load products data
        self.load_products()
        self.load_feedback()
    
    def load_products_from_csv(self, csv_path):
        """Load wine products from the LCBO CSV file"""
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return 0
        
        products_added = 0
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                for row in csv_reader:
                    try:
                        # Use permanent_id as the product ID
                        product_id = str(row.get('permanent_id', uuid.uuid4()))
                        
                        # Format price properly
                        try:
                            price = float(row.get('price', 0.0))
                        except (ValueError, TypeError):
                            price = 0.0
                        
                        # Create product object mapping LCBO data to our structure
                        product = {
                            'id': product_id,
                            'name': row.get('title', ''),
                            'description': row.get('description', ''),
                            'price': price,
                            'category': row.get('subcategory', row.get('category', '')),
                            'tags': f"{row.get('country', '')},{row.get('brand', '')},{row.get('alcohol_content', '')}",
                            'image': row.get('image_url', ''),
                            'rating': row.get('rating', '0'),
                            'alcohol_content': row.get('alcohol_content', '0')
                        }
                        
                        # Skip products with missing essential data
                        if not product['name']:
                            continue
                        
                        # Add to products list
                        self.products.append(product)
                        
                        # Create product text for embedding
                        product_text = self._get_product_text(product)
                        product_embedding = self.model.encode(product_text)
                        
                        # Add metadata
                        metadata = {
                            'name': product['name'],
                            'category': product['category'],
                            'price': str(product['price']),
                            'country': row.get('country', ''),
                            'alcohol_content': row.get('alcohol_content', ''),
                            'rating': row.get('rating', ''),
                            'product_id': product_id
                        }
                        
                        # Add to collection
                        self.product_collection.add(
                            ids=[product_id],
                            documents=[product_text],
                            metadatas=[metadata]
                        )
                        
                        products_added += 1
                        
                    except Exception as e:
                        print(f"Error processing row: {str(e)}")
                        continue
                
                # Save products to JSON
                self.save_products()
                print(f"Added {products_added} wine products from CSV")
                return products_added
                
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            return 0

    def load_products(self):
        """Load products from JSON file"""
        try:
            with open(self.products_path, 'r') as f:
                self.products = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.products = []
            self.save_products()
    
    def save_products(self):
        """Save products to JSON file"""
        with open(self.products_path, 'w') as f:
            json.dump(self.products, f, indent=2)
    
    def load_feedback(self):
        """Load user feedback from JSON file"""
        try:
            with open(self.feedback_path, 'r') as f:
                self.feedback = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.feedback = {"users": {}}
            self.save_feedback()
    
    def save_feedback(self):
        """Save user feedback to JSON file"""
        with open(self.feedback_path, 'w') as f:
            json.dump(self.feedback, f, indent=2)
    
    def _get_product_text(self, product):
        """Create a textual representation of a wine product for embedding"""
        tags = product.get('tags', '').split(',') if product.get('tags') else []
        tag_text = " ".join(tags) if tags else ""
        
        # Include more wine-specific attributes in the text representation
        wine_text = f"Wine: {product['name']}. "
        wine_text += f"Category: {product.get('category', '')}. "
        wine_text += f"Description: {product.get('description', '')}. "
        wine_text += f"Country: {product.get('country', '')}. " if 'country' in product else ""
        wine_text += f"Brand: {product.get('brand', '')}. " if 'brand' in product else ""
        wine_text += f"Alcohol Content: {product.get('alcohol_content', '')}. " if 'alcohol_content' in product else ""
        wine_text += f"Rating: {product.get('rating', '')}. " if 'rating' in product else ""
        wine_text += f"Tags: {tag_text}"
        
        return wine_text

    def add_product(self, product):
        """Add a product to the system"""
        # Generate product ID if not provided
        if 'id' not in product:
            product['id'] = str(uuid.uuid4())
        
        # Add timestamp
        product['created_at'] = datetime.now().isoformat()
        
        # Add to products list
        self.products.append(product)
        self.save_products()
        
        # Add to vector store
        product_text = self._get_product_text(product)
        product_embedding = self.model.encode(product_text)
        
        self.product_collection.add(
            ids=[product['id']],
            documents=[product_text],
            metadatas=[{
                'name': product['name'],
                'category': product['category'],
                'price': str(product['price']),
                'product_id': product['id']
            }]
        )
        
        return product
    
    def delete_product(self, product_id):
        """Delete a product from the system"""
        # Remove from products list
        self.products = [p for p in self.products if p['id'] != product_id]
        self.save_products()
        
        # Remove from vector store
        try:
            self.product_collection.delete(ids=[product_id])
        except:
            pass  # Product may not be in vector store
        
        return True
    
    def add_feedback(self, user_id, product_id, feedback_type):
        """Add user feedback for a product"""
        # Initialize user if not exists
        if user_id not in self.feedback["users"]:
            self.feedback["users"][user_id] = {
                "likes": [],
                "dislikes": [],
                "timestamps": {}
            }
        
        # Remove product from opposite list if it exists
        if feedback_type == "up":
            if product_id in self.feedback["users"][user_id]["dislikes"]:
                self.feedback["users"][user_id]["dislikes"].remove(product_id)
            if product_id not in self.feedback["users"][user_id]["likes"]:
                self.feedback["users"][user_id]["likes"].append(product_id)
        else:  # feedback_type == "down"
            if product_id in self.feedback["users"][user_id]["likes"]:
                self.feedback["users"][user_id]["likes"].remove(product_id)
            if product_id not in self.feedback["users"][user_id]["dislikes"]:
                self.feedback["users"][user_id]["dislikes"].append(product_id)
        
        # Add timestamp
        self.feedback["users"][user_id]["timestamps"][product_id] = datetime.now().isoformat()
        
        # Save feedback
        self.save_feedback()
        
        # Update user preference embedding
        self.update_user_preference(user_id)
        
        return True
    
    def update_user_preference(self, user_id):
        """Update user preference embedding based on feedback"""
        if user_id not in self.feedback["users"]:
            return None
        
        user_data = self.feedback["users"][user_id]
        liked_products = [p for p in self.products if p['id'] in user_data["likes"]]
        disliked_products = [p for p in self.products if p['id'] in user_data["dislikes"]]
        
        # If no feedback, return None
        if not liked_products and not disliked_products:
            return None
        
        # Get embeddings for liked and disliked products
        liked_embeddings = []
        for product in liked_products:
            product_text = self._get_product_text(product)
            liked_embeddings.append(self.model.encode(product_text))
        
        disliked_embeddings = []
        for product in disliked_products:
            product_text = self._get_product_text(product)
            disliked_embeddings.append(self.model.encode(product_text))
        
        # Compute preference embedding
        if liked_embeddings:
            liked_avg = np.mean(liked_embeddings, axis=0)
        else:
            liked_avg = np.zeros(self.model.get_sentence_embedding_dimension())
            
        if disliked_embeddings:
            disliked_avg = np.mean(disliked_embeddings, axis=0)
        else:
            disliked_avg = np.zeros(self.model.get_sentence_embedding_dimension())
        
        # Compute preference embedding: move toward liked, away from disliked
        preference_embedding = liked_avg
        if disliked_embeddings:
            # Subtract disliked, but ensure we don't go too far
            scale_factor = 0.5
            preference_embedding = preference_embedding - (disliked_avg * scale_factor)
            
            # Normalize the embedding
            norm = np.linalg.norm(preference_embedding)
            if norm > 0:
                preference_embedding = preference_embedding / norm
        
        # Update or add to vector store
        try:
            self.user_collection.get(ids=[user_id])
            self.user_collection.update(
                ids=[user_id],
                embeddings=[preference_embedding.tolist()],
                metadatas=[{'user_id': user_id}]
            )
        except:
            self.user_collection.add(
                ids=[user_id],
                embeddings=[preference_embedding.tolist()],
                metadatas=[{'user_id': user_id}]
            )
        
        return preference_embedding
    
    def get_recommendations(self, user_id, n_results=3, excluded_ids=None):
        """Get product recommendations for a user"""
        if excluded_ids is None:
            excluded_ids = []
        
        # Try to get user's preference embedding
        try:
            results = self.user_collection.get(ids=[user_id])
            preference_embedding = results['embeddings'][0]
        except:
            # If no profile exists, try to create one
            preference_embedding = self.update_user_preference(user_id)
            
            # If still no embedding, return random products
            if preference_embedding is None:
                # Get random products excluding already seen ones
                available_products = [p for p in self.products if p['id'] not in excluded_ids]
                if len(available_products) <= n_results:
                    return available_products
                else:
                    import random
                    return random.sample(available_products, n_results)
        
        # Query for products using the preference embedding
        results = self.product_collection.query(
            query_embeddings=[preference_embedding.tolist()],
            n_results=n_results + len(excluded_ids)  # Query more to account for excluded IDs
        )
        
        # Filter out excluded IDs and get product details
        recommended_products = []
        for i, product_id in enumerate(results['ids'][0]):
            if product_id not in excluded_ids:
                # Find the full product details
                product = next((p for p in self.products if p['id'] == product_id), None)
                if product:
                    recommended_products.append(product)
                
                if len(recommended_products) >= n_results:
                    break
        
        # If we didn't get enough recommendations, add some random products
        if len(recommended_products) < n_results:
            seen_ids = set([p['id'] for p in recommended_products] + excluded_ids)
            available_products = [p for p in self.products if p['id'] not in seen_ids]
            
            import random
            additional_needed = min(n_results - len(recommended_products), len(available_products))
            if additional_needed > 0:
                additional_products = random.sample(available_products, additional_needed)
                recommended_products.extend(additional_products)
        
        return recommended_products