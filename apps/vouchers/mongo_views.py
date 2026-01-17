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
            "Lấy danh sách mã giảm giá thành công",
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
            return api_error("Không tìm thấy mã giảm giá", status_code=status.HTTP_404_NOT_FOUND)
        
        serializer = VoucherSerializer(voucher)
        return api_success("Lấy thông tin mã giảm giá thành công", {"voucher": serializer.data})

    def create(self, request):
        serializer = VoucherSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        voucher = serializer.save()
        return api_success(
            "Tạo mã giảm giá thành công",
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
            return api_error("Không tìm thấy mã giảm giá", status_code=status.HTTP_404_NOT_FOUND)
            
        serializer = VoucherSerializer(voucher, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        voucher = serializer.save()
        return api_success("Cập nhật mã giảm giá thành công", {"voucher": VoucherSerializer(voucher).data})

    def destroy(self, request, pk=None):
        try:
            voucher = Voucher.objects.get(id=ObjectId(pk))
            voucher.delete()
            return api_success("Xóa mã giảm giá thành công")
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
             return api_error("Không tìm thấy mã giảm giá", status_code=status.HTTP_404_NOT_FOUND)
        
        # Check validity
        if voucher.status != "ACTIVE":
             return api_error("Mã giảm giá không hoạt động", status_code=status.HTTP_400_BAD_REQUEST)
        
        now = datetime.utcnow()
        if voucher.start_date and voucher.start_date > now:
             return api_error("Mã giảm giá chưa đến thời gian áp dụng", status_code=status.HTTP_400_BAD_REQUEST)
             
        if voucher.end_date and voucher.end_date < now:
             return api_error("Mã giảm giá đã hết hạn", status_code=status.HTTP_400_BAD_REQUEST)
             
        if voucher.quantity <= voucher.used_count:
             return api_error("Mã giảm giá đã hết số lượng", status_code=status.HTTP_400_BAD_REQUEST)
             
        if voucher.min_order_value > order_value:
             return api_error(f"Giá trị đơn hàng phải từ {voucher.min_order_value}", status_code=status.HTTP_400_BAD_REQUEST)
        
        # Calculate discount
        discount_amount = 0
        if voucher.type == "FIXED_AMOUNT":
            discount_amount = voucher.value
        elif voucher.type == "PERCENTAGE":
            discount_amount = (order_value * voucher.value) / 100
            if voucher.max_discount > 0:
                discount_amount = min(discount_amount, voucher.max_discount)
                
        return api_success(
            "Mã giảm giá hợp lệ",
            {
                "voucher": {
                    "code": voucher.code,
                    "discountValue": float(voucher.value),
                    "discountType": voucher.type
                },
                "discountAmount": float(discount_amount)
            }
        )

    @action(detail=False, methods=["get"], url_path="user/(?P<user_id>[^/.]+)")
    def user_vouchers(self, request, user_id=None):
        # Hiện tại trả về tất cả voucher đang active và còn hạn
        # Trong tương lai có thể thêm logic lọc voucher đã lưu/đã dùng của riêng user
        
        now = datetime.utcnow()
        queryset = Voucher.objects.filter(
            status="ACTIVE",
            start_date__lte=now,
            end_date__gte=now
        )
        
        # Filter logic manually for those that cannot be easily done with mongoengine filter if complex,
        # but here basic fields are fine.
        # Check quantity > used_count. MongoEngine Q objects usually needed for field comparison so doing partial python filter
        
        valid_vouchers = []
        for voucher in queryset:
            if voucher.quantity > voucher.used_count:
                valid_vouchers.append(voucher)
                
        serializer = VoucherSerializer(valid_vouchers, many=True)
        
        return api_success(
            "Lấy danh sách mã giảm giá cho người dùng thành công",
            {
                "vouchers": serializer.data,
                "count": len(valid_vouchers)
            }
        )
