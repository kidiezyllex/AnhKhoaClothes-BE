from __future__ import annotations
from datetime import datetime
import mongoengine as me
from mongoengine import fields

class ReturnRequest(me.Document):
    meta = {
        "collection": "return_requests",
        "indexes": ["user_id", "order_id", "status"],
        "strict": False,
    }

    user_id = fields.ObjectIdField(required=True, db_field="userId")
    order_id = fields.ObjectIdField(required=True, db_field="originalOrder")
    # items can be a list of objects describing what to return
    items = fields.ListField(fields.DictField(), default=list) 
    reason = fields.StringField(required=True)
    status = fields.StringField(default="PENDING", choices=["PENDING", "APPROVED", "REJECTED", "COMPLETED"])
    
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
