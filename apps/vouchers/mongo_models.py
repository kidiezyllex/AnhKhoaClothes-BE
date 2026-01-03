from __future__ import annotations
from datetime import datetime
import mongoengine as me
from mongoengine import fields

class Voucher(me.Document):
    meta = {
        "collection": "vouchers",
        "indexes": ["code"],
        "strict": False,
    }

    code = fields.StringField(required=True, unique=True)
    name = fields.StringField()
    type = fields.StringField(choices=["PERCENTAGE", "FIXED_AMOUNT"], default="PERCENTAGE")
    value = fields.DecimalField(precision=2)
    quantity = fields.IntField(default=0)
    used_count = fields.IntField(default=0)
    start_date = fields.DateTimeField()
    end_date = fields.DateTimeField()
    min_order_value = fields.DecimalField(precision=2, default=0)
    max_discount = fields.DecimalField(precision=2, default=0)
    status = fields.StringField(default="ACTIVE") # ACTIVE, INACTIVE

    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
