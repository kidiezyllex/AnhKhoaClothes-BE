from datetime import datetime
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from apps.utils import api_error, api_success
from .mongo_models import Voucher
from .mongo_serializers import VoucherSerializer, VoucherValidateSerializer

class VoucherViewSet(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    @action(detail=False, methods=["post"], url_path="validate")
    def validate(self, request):
        serializer = VoucherValidateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        code = serializer.validated_data["code"]
        order_value = serializer.validated_data["orderValue"]
        
        try:
            voucher = Voucher.objects.get(code=code)
        except Voucher.DoesNotExist:
             return api_error("Voucher not found", status_code=status.HTTP_404_NOT_FOUND)
        
        # Check validity
        if voucher.status != "ACTIVE":
             return api_error("Voucher is not active", status_code=status.HTTP_400_BAD_REQUEST)
        
        now = datetime.utcnow()
        if voucher.start_date and voucher.start_date > now:
             return api_error("Voucher has not started yet", status_code=status.HTTP_400_BAD_REQUEST)
             
        if voucher.end_date and voucher.end_date < now:
             return api_error("Voucher has expired", status_code=status.HTTP_400_BAD_REQUEST)
             
        if voucher.quantity <= voucher.used_count:
             return api_error("Voucher out of stock", status_code=status.HTTP_400_BAD_REQUEST)
             
        if voucher.min_order_value > order_value:
             return api_error(f"Order value must be at least {voucher.min_order_value}", status_code=status.HTTP_400_BAD_REQUEST)
        
        # Calculate discount
        discount_amount = 0
        if voucher.type == "FIXED_AMOUNT":
            discount_amount = voucher.value
        elif voucher.type == "PERCENTAGE":
            discount_amount = (order_value * voucher.value) / 100
            if voucher.max_discount > 0:
                discount_amount = min(discount_amount, voucher.max_discount)
                
        return api_success(
            "Voucher is valid",
            {
                "voucher": {
                    "code": voucher.code,
                    "discountValue": float(voucher.value),
                    "discountType": voucher.type
                },
                "discountAmount": float(discount_amount)
            }
        )
