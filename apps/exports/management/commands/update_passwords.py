from __future__ import annotations

from django.core.management.base import BaseCommand
from apps.users.mongo_models import User
from config.mongodb import connect_mongodb


class Command(BaseCommand):
    help = 'Update passwords for all users'

    def handle(self, *args, **options):
        # Connect to MongoDB
        connect_mongodb()

        # Get all users
        users = User.objects.all()
        
        admin_count = 0
        customer_count = 0
        
        for user in users:
            if user.email == "adminnoware@gmail.com":
                # Set admin password
                user.set_password("Admin123!")
                user.save()
                admin_count += 1
                self.stdout.write(self.style.SUCCESS(f"Updated password for admin: {user.email}"))
            else:
                # Set customer password
                user.set_password("Customer123!")
                user.save()
                customer_count += 1
                self.stdout.write(f"Updated password for customer: {user.email}")
        
        self.stdout.write(self.style.SUCCESS(f"\nTotal updated:"))
        self.stdout.write(self.style.SUCCESS(f"  - Admin accounts: {admin_count}"))
        self.stdout.write(self.style.SUCCESS(f"  - Customer accounts: {customer_count}"))
        self.stdout.write(self.style.SUCCESS(f"  - Total: {admin_count + customer_count}"))
