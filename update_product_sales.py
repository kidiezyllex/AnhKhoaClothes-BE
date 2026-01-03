import os
import django
import random
from decimal import Decimal

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.products.mongo_models import Product
from config.mongodb import connect_mongodb

def update_product_sales():
    connect_mongodb()
    
    products = Product.objects.all()
    total = products.count()
    print(f"Updating sale values for {total} products...")
    
    for idx, product in enumerate(products, 1):
        # Generate random sale from 0.0 to 15.0
        sale = round(random.uniform(0.0, 15.0), 1)
        
        product.sale = Decimal(str(sale))
        product.save()
        
        if idx % 100 == 0:
            print(f"Processed {idx}/{total} products...")
    
    print(f"\n✅ Successfully updated all {total} products")
    print(f"   - Sale values now range from 0.0% to 15.0%")

if __name__ == "__main__":
    update_product_sales()
