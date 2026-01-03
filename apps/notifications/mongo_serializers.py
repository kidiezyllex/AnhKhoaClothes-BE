from rest_framework import serializers
from .mongo_models import Notification
from bson import ObjectId

class NotificationSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    userId = serializers.CharField(source="user_id", required=False, allow_null=True)
    title = serializers.CharField()
    content = serializers.CharField(source="message")
    type = serializers.CharField(required=False, default="SYSTEM")
    isRead = serializers.BooleanField(source="is_read", read_only=True)
    createdAt = serializers.DateTimeField(source="created_at", read_only=True)
    
    def get_id(self, obj):
        return str(obj.id)

    def create(self, validated_data):
        if "user_id" in validated_data and validated_data["user_id"]:
            validated_data["user_id"] = ObjectId(validated_data["user_id"])
        return Notification(**validated_data).save()
