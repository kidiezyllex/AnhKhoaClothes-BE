from __future__ import annotations

import os

import mongoengine
from django.conf import settings

def connect_mongodb() -> None:
    mongo_uri = getattr(settings, "MONGO_URI", "mongodb://localhost:27017/config")
    db_name = getattr(settings, "MONGODB_DB_NAME", "config")

    try:
        connect_kwargs = {
            "db": db_name,
            "host": mongo_uri,
            "alias": "default",
        }
        
        try:
            import certifi
            connect_kwargs["tlsCAFile"] = certifi.where()
        except ImportError:
            pass

        mongoengine.connect(**connect_kwargs)
        print(f"Connected to MongoDB: {db_name}")
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        raise

