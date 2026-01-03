from __future__ import annotations
from datetime import datetime
import mongoengine as me
from mongoengine import fields

class Notification(me.Document):
    meta = {
        "collection": "notifications",
        "indexes": ["user_id", "is_read", "type"],
        "strict": False,
    }

    user_id = fields.ObjectIdField(db_field="userId", required=False) # Optional if generic system notification
    title = fields.StringField(required=True)
    message = fields.StringField(required=True, db_field="content") # Map content from payload to message
    type = fields.StringField(default="SYSTEM", choices=["SYSTEM", "ORDER", "PROMOTION", "VOUCHER"])
    is_read = fields.BooleanField(default=False, db_field="isRead")
    
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
