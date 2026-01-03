from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from django.core.management.base import BaseCommand
from apps.users.mongo_models import User, UserInteraction
from apps.products.mongo_models import Product
from config.mongodb import connect_mongodb
from bson import ObjectId

class Command(BaseCommand):
    help = 'Import users, products, and interactions from CSV files in apps/exports'

    def handle(self, *args, **options):
        # Connect to MongoDB
        connect_mongodb()

        # Get the path to the CSV files
        # They are in apps/exports/
        current_file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        
        # Import Products
        products_file = os.path.join(package_dir, 'products.csv')
        if os.path.exists(products_file):
            self.stdout.write("Importing products...")
            with open(products_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    try:
                        product_id = int(row['id'])
                        images = json.loads(row['images']) if row['images'] else []
                        
                        product = Product.objects(id=product_id).first()
                        if not product:
                            product = Product(id=product_id)
                        
                        product.gender = row['gender']
                        product.masterCategory = row['masterCategory']
                        product.subCategory = row['subCategory']
                        product.articleType = row['articleType']
                        product.baseColour = row['baseColour']
                        product.season = row['season']
                        product.year = int(row['year']) if row['year'] and row['year'] != '' else None
                        product.usage = row['usage']
                        product.productDisplayName = row['productDisplayName']
                        product.images = images
                        product.save()
                        count += 1
                    except Exception as e:
                        self.stdout.write(self.style.WARNING(f"Error importing product {row.get('id')}: {e}"))
            self.stdout.write(self.style.SUCCESS(f"Successfully imported {count} products"))
        else:
            self.stdout.write(self.style.ERROR(f"Products file not found: {products_file}"))

        # Import Users
        users_file = os.path.join(package_dir, 'users.csv')
        if os.path.exists(users_file):
            self.stdout.write("Importing users...")
            with open(users_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    try:
                        user_id = row['id']
                        interaction_history = json.loads(row['interaction_history']) if row['interaction_history'] else []
                        
                        # Convert timestamps in interaction history
                        for item in interaction_history:
                            if 'timestamp' in item and isinstance(item['timestamp'], str):
                                try:
                                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                                except ValueError:
                                    pass

                        user = User.objects(id=user_id).first()
                        if not user:
                            # Use ObjectId if it looks like one, otherwise let mongoengine handle it
                            user = User(id=ObjectId(user_id) if len(user_id) == 24 else user_id)
                        
                        user.name = row['name']
                        user.email = row['email']
                        user.age = int(row['age']) if row['age'] and row['age'] != '' else None
                        user.gender = row['gender'] if row['gender'] and row['gender'] != '' else None
                        user.interaction_history = interaction_history
                        user.save()
                        count += 1
                    except Exception as e:
                        self.stdout.write(self.style.WARNING(f"Error importing user {row.get('id')}: {e}"))
            self.stdout.write(self.style.SUCCESS(f"Successfully imported {count} users"))
        else:
            self.stdout.write(self.style.ERROR(f"Users file not found: {users_file}"))

        # Import Interactions (into user_interactions collection)
        interactions_file = os.path.join(package_dir, 'interactions.csv')
        if os.path.exists(interactions_file):
            self.stdout.write("Importing interactions...")
            with open(interactions_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    try:
                        # Find existing interaction to avoid duplicates or just create new
                        # Based on timestamp, user and product
                        timestamp = datetime.fromisoformat(row['timestamp'])
                        user_id = ObjectId(row['user_id']) if len(row['user_id']) == 24 else row['user_id']
                        
                        interaction = UserInteraction.objects(
                            user_id=user_id,
                            product_id=int(row['product_id']),
                            timestamp=timestamp
                        ).first()
                        
                        if not interaction:
                            interaction = UserInteraction(
                                user_id=user_id,
                                product_id=int(row['product_id']),
                                interaction_type=row['interaction_type'],
                                timestamp=timestamp
                            )
                            interaction.save()
                            count += 1
                    except Exception as e:
                        self.stdout.write(self.style.WARNING(f"Error importing interaction: {e}"))
            self.stdout.write(self.style.SUCCESS(f"Successfully imported {count} interactions"))
        else:
            self.stdout.write(self.style.ERROR(f"Interactions file not found: {interactions_file}"))
