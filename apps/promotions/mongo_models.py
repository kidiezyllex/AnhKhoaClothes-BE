from __future__ import annotations
from datetime import datetime
import mongoengine as me
from mongoengine import fields

class Promotion(me.Document):
    meta = {
        "collection": "promotions",
        "indexes": ["name", "start_date", "end_date"],
        "strict": False,
    }

    name = fields.StringField(required=True)
    description = fields.StringField()
    discount_value = fields.DecimalField(precision=2, db_field="discountValue")
    discount_type = fields.StringField(choices=["PERCENTAGE", "FIXED_AMOUNT"], default="PERCENTAGE", db_field="discountType")
    start_date = fields.DateTimeField(db_field="startDate")
    end_date = fields.DateTimeField(db_field="endDate")
    apply_to = fields.StringField(db_field="applyTo", default="ALL_PRODUCTS") # ALL_PRODUCTS, SPECIFIC_PRODUCTS, etc
    product_ids = fields.ListField(fields.ObjectIdField(), db_field="productIds")
    status = fields.StringField(default="ACTIVE")

    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
