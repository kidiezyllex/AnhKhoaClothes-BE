import os
import django
import random
from decimal import Decimal

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.products.mongo_models import Product, ProductVariant, Color, Size
from config.mongodb import connect_mongodb

def seed_product_data():
    connect_mongodb()
    
    # Get or create some colors
    colors = []
    color_data = [
        {"name": "Black", "hex_code": "#000000"},
        {"name": "White", "hex_code": "#FFFFFF"},
        {"name": "Red", "hex_code": "#FF0000"},
        {"name": "Blue", "hex_code": "#0000FF"},
        {"name": "Green", "hex_code": "#00FF00"},
        {"name": "Yellow", "hex_code": "#FFFF00"},
        {"name": "Pink", "hex_code": "#FFC0CB"},
        {"name": "Gray", "hex_code": "#808080"},
        {"name": "Navy", "hex_code": "#000080"},
        {"name": "Brown", "hex_code": "#A52A2A"},
    ]
    
    for color_info in color_data:
        existing_color = Color.objects(name=color_info["name"]).first()
        if existing_color:
            color = existing_color
        else:
            color = Color(
                name=color_info["name"],
                hex_code=color_info["hex_code"],
                status="ACTIVE"
            )
            color.save()
        colors.append(color)
    
    # Get or create some sizes
    sizes = []
    size_data = [
        {"name": "Extra Small", "code": "XS"},
        {"name": "Small", "code": "S"},
        {"name": "Medium", "code": "M"},
        {"name": "Large", "code": "L"},
        {"name": "Extra Large", "code": "XL"},
        {"name": "XXL", "code": "XXL"},
    ]
    
    for size_info in size_data:
        existing_size = Size.objects(code=size_info["code"]).first()
        if existing_size:
            size = existing_size
        else:
            size = Size(
                code=size_info["code"],
                name=size_info["name"],
                status="ACTIVE"
            )
            size.save()
        sizes.append(size)
    
    print(f"Created/found {len(colors)} colors and {len(sizes)} sizes")
    
    # Update all products
    products = Product.objects.all()
    total = products.count()
    print(f"Updating {total} products...")
    
    for idx, product in enumerate(products, 1):
        # Generate random rating (3.0 to 5.0)
        rating = round(random.uniform(3.0, 5.0), 1)
        
        # Generate random sale (0%, 10%, 20%, 30%, 40%, 50%)
        sale_options = [0, 10, 20, 30, 40, 50]
        sale = Decimal(random.choice(sale_options))
        
        # Update product
        product.rating = rating
        product.sale = sale
        product.num_reviews = random.randint(5, 150)
        product.save()
        
        # Delete existing variants for this product
        ProductVariant.objects(product_id=product.id).delete()
        
        # Create 3-6 random variants
        num_variants = random.randint(3, 6)
        variant_combinations = set()
        
        for _ in range(num_variants):
            # Random color and size
            color = random.choice(colors)
            size = random.choice(sizes)
            
            # Avoid duplicate combinations
            combo = (color.hex_code, size.code)
            if combo in variant_combinations:
                continue
            variant_combinations.add(combo)
            
            # Generate random price (100,000 to 2,000,000 VND)
            base_price = random.randint(100, 2000) * 1000
            price = Decimal(base_price)
            
            # Random stock (0 to 100)
            stock = random.randint(0, 100)
            
            # Create variant
            variant = ProductVariant(
                product_id=product.id,
                color=color.hex_code,
                size=size.code,
                price=price,
                stock=stock
            )
            variant.save()
        
        # Update product stock from variants
        product.update_stock_from_variants()
        
        if idx % 50 == 0:
            print(f"Processed {idx}/{total} products...")
    
    print(f"\n✅ Successfully updated all {total} products with:")
    print(f"   - Random ratings (3.0-5.0)")
    print(f"   - Random sales (0-50%)")
    print(f"   - Random variants (3-6 per product)")
    print(f"   - Random prices and stock levels")

if __name__ == "__main__":
    seed_product_data()
