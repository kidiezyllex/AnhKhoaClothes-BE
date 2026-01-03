from rest_framework import serializers
from .mongo_models import Voucher

class VoucherSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    code = serializers.CharField()
    name = serializers.CharField(required=False, allow_blank=True)
    type = serializers.CharField()
    value = serializers.DecimalField(max_digits=10, decimal_places=2)
    quantity = serializers.IntegerField()
    usedCount = serializers.IntegerField(source="used_count", required=False)
    startDate = serializers.DateTimeField(source="start_date")
    endDate = serializers.DateTimeField(source="end_date")
    minOrderValue = serializers.DecimalField(source="min_order_value", max_digits=10, decimal_places=2)
    maxDiscount = serializers.DecimalField(source="max_discount", max_digits=10, decimal_places=2)
    status = serializers.CharField()

    def get_id(self, obj):
        return str(obj.id)
    
    def create(self, validated_data):
        return Voucher(**validated_data).save()
        
    def update(self, instance, validated_data):
        for key, value in validated_data.items():
             setattr(instance, key, value)
        instance.save()
        return instance

class VoucherValidateSerializer(serializers.Serializer):
    code = serializers.CharField()
    orderValue = serializers.DecimalField(max_digits=10, decimal_places=2)
