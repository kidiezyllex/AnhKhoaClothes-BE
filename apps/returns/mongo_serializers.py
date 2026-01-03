from rest_framework import serializers
from .mongo_models import ReturnRequest
from bson import ObjectId

class ReturnRequestSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    userId = serializers.CharField(source="user_id", read_only=True)
    originalOrder = serializers.CharField(source="order_id")
    items = serializers.ListField(child=serializers.DictField())
    reason = serializers.CharField()
    status = serializers.CharField(read_only=True)
    createdAt = serializers.DateTimeField(source="created_at", read_only=True)
    
    def get_id(self, obj):
        return str(obj.id)

    def create(self, validated_data):
        user = self.context['request'].user
        validated_data['user_id'] = user.id
        validated_data['order_id'] = ObjectId(validated_data['order_id'])
        return ReturnRequest(**validated_data).save()

class ReturnStatusUpdateSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=["APPROVED", "REJECTED", "COMPLETED"])
