from datetime import datetime
from bson import ObjectId
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from apps.utils import api_error, api_success, get_pagination_params, paginate_queryset
from .mongo_models import Voucher
from .mongo_serializers import VoucherSerializer, VoucherValidateSerializer

class VoucherViewSet(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def list(self, request):
        queryset = Voucher.objects.all().order_by("-created_at")
        
        # Filter by status if provided
        status_filter = request.query_params.get("status")
        if status_filter:
            queryset = queryset.filter(status=status_filter)
            
        page, page_size = get_pagination_params(request)
        vouchers, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        
        serializer = VoucherSerializer(vouchers, many=True)
        return api_success(
            "Vouchers retrieved successfully",
            {
                "vouchers": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            }
        )

    def retrieve(self, request, pk=None):
        try:
            voucher = Voucher.objects.get(id=ObjectId(pk))
        except (Voucher.DoesNotExist, Exception):
            return api_error("Voucher not found", status_code=status.HTTP_404_NOT_FOUND)
        
        serializer = VoucherSerializer(voucher)
        return api_success("Voucher retrieved successfully", {"voucher": serializer.data})

    def create(self, request):
        serializer = VoucherSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        voucher = serializer.save()
        return api_success(
            "Voucher created successfully",
            {"voucher": VoucherSerializer(voucher).data},
            status_code=status.HTTP_201_CREATED
        )

    def update(self, request, pk=None):
        return self._update(request, pk, partial=False)

    def partial_update(self, request, pk=None):
        return self._update(request, pk, partial=True)

    def _update(self, request, pk, partial=False):
        try:
            voucher = Voucher.objects.get(id=ObjectId(pk))
        except (Voucher.DoesNotExist, Exception):
            return api_error("Voucher not found", status_code=status.HTTP_404_NOT_FOUND)
            
        serializer = VoucherSerializer(voucher, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        voucher = serializer.save()
        return api_success("Voucher updated successfully", {"voucher": VoucherSerializer(voucher).data})

    def destroy(self, request, pk=None):
        try:
            voucher = Voucher.objects.get(id=ObjectId(pk))
            voucher.delete()
            return api_success("Voucher deleted successfully")
        except (Voucher.DoesNotExist, Exception):
            return api_error("Voucher not found", status_code=status.HTTP_404_NOT_FOUND)

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
