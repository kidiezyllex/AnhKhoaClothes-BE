from rest_framework import serializers
from .mongo_models import Promotion
from bson import ObjectId

class PromotionSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    name = serializers.CharField()
    description = serializers.CharField(required=False, allow_blank=True)
    discountValue = serializers.DecimalField(source="discount_value", max_digits=10, decimal_places=2)
    discountType = serializers.CharField(source="discount_type")
    startDate = serializers.DateTimeField(source="start_date")
    endDate = serializers.DateTimeField(source="end_date")
    applyTo = serializers.CharField(source="apply_to")
    productIds = serializers.ListField(child=serializers.CharField(), source="product_ids", required=False)
    status = serializers.CharField(required=False, default="ACTIVE")
    
    def get_id(self, obj):
        return str(obj.id)

    def create(self, validated_data):
        if "product_ids" in validated_data and validated_data["product_ids"]:
            validated_data["product_ids"] = [ObjectId(pid) for pid in validated_data["product_ids"]]
        return Promotion(**validated_data).save()
        
    def update(self, instance, validated_data):
        if "product_ids" in validated_data and validated_data["product_ids"]:
            validated_data["product_ids"] = [ObjectId(pid) for pid in validated_data["product_ids"]]
            
        for key, value in validated_data.items():
            setattr(instance, key, value)
        instance.save()
        return instance
