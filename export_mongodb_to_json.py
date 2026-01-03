import os
import django
import json
from datetime import datetime
from decimal import Decimal
from bson import ObjectId

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from config.mongodb import connect_mongodb
from apps.products.mongo_models import Product, ProductVariant, ProductReview, Brand, Material, Category, Color, Size, ContentSection
from apps.users.mongo_models import User, UserAddress, UserInteraction, OutfitHistory, PasswordResetAudit
from apps.orders.mongo_models import Order, OrderItem
from apps.promotions.mongo_models import Promotion

class MongoJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB types"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

def export_collection(model, filename):
    """Export a MongoDB collection to JSON file"""
    try:
        documents = model.objects.all()
        count = documents.count()
        
        if count == 0:
            print(f"⚠️  {filename}: No data found")
            return 0
        
        # Convert to list of dictionaries
        data = []
        for doc in documents:
            doc_dict = doc.to_mongo().to_dict()
            data.append(doc_dict)
        
        # Write to JSON file
        output_path = os.path.join('exports_json', filename)
        os.makedirs('exports_json', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=MongoJSONEncoder, indent=2, ensure_ascii=False)
        
        print(f"✅ {filename}: Exported {count} documents")
        return count
        
    except Exception as e:
        print(f"❌ {filename}: Error - {str(e)}")
        return 0

def export_all_collections():
    """Export all MongoDB collections to JSON files"""
    connect_mongodb()
    
    print("=" * 60)
    print("EXPORTING MONGODB DATA TO JSON FILES")
    print("=" * 60)
    print()
    
    collections = [
        # Products
        (Product, 'products.json'),
        (ProductVariant, 'product_variants.json'),
        (ProductReview, 'product_reviews.json'),
        (Brand, 'brands.json'),
        (Material, 'materials.json'),
        (Category, 'categories.json'),
        (Color, 'colors.json'),
        (Size, 'sizes.json'),
        (ContentSection, 'content_sections.json'),
        
        # Users
        (User, 'users.json'),
        (UserAddress, 'user_addresses.json'),
        (UserInteraction, 'user_interactions.json'),
        (OutfitHistory, 'outfit_history.json'),
        (PasswordResetAudit, 'password_reset_audit.json'),
        
        # Orders
        (Order, 'orders.json'),
        (OrderItem, 'order_items.json'),
        
        # Promotions
        (Promotion, 'promotions.json'),
    ]
    
    total_documents = 0
    successful_exports = 0
    
    for model, filename in collections:
        count = export_collection(model, filename)
        if count > 0:
            successful_exports += 1
            total_documents += count
    
    print()
    print("=" * 60)
    print(f"EXPORT COMPLETED")
    print("=" * 60)
    print(f"✅ Successfully exported {successful_exports}/{len(collections)} collections")
    print(f"📊 Total documents: {total_documents}")
    print(f"📁 Output directory: exports_json/")
    print()

if __name__ == "__main__":
    export_all_collections()
