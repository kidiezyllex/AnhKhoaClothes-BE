from __future__ import annotations

from bson import ObjectId
from datetime import datetime
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action

from apps.utils import api_error, api_success, get_pagination_params, paginate_queryset

from .mongo_models import Order
from .mongo_serializers import OrderSerializer, OrderStatusUpdateSerializer

class OrderViewSet(viewsets.ViewSet):

    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def list(self, request):
        queryset = Order.objects.all()

        if request.query_params.get("is_paid") == "true":
            queryset = queryset.filter(is_paid=True)
        elif request.query_params.get("is_paid") == "false":
            queryset = queryset.filter(is_paid=False)

        if request.query_params.get("is_processing") == "true":
            queryset = queryset.filter(is_processing=True)

        if request.query_params.get("is_outfit_purchase") == "true":
            queryset = queryset.filter(is_outfit_purchase=True)

        ordering = request.query_params.get("ordering", "-created_at")
        if ordering.startswith("-"):
            queryset = queryset.order_by(f"-{ordering[1:]}")
        else:
            queryset = queryset.order_by(ordering)

        page, page_size = get_pagination_params(request)
        orders, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = OrderSerializer(orders, many=True)
        return api_success(
            "Lấy danh sách đơn hàng thành công",
            {
                "orders": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )

    def retrieve(self, request, pk=None):
        try:
            order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
            return api_error(
                "Đơn hàng không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        serializer = OrderSerializer(order)
        return api_success(
            "Lấy thông tin đơn hàng thành công",
            {
                "order": serializer.data,
            },
        )

    def create(self, request):
        request_serializer = OrderSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)

        validated_data = request_serializer.validated_data.copy()
        if not validated_data.get("user_id"):
            if request.user and hasattr(request.user, 'id') and request.user.is_authenticated:
                validated_data["user_id"] = str(request.user.id)
            else:
                return api_error(
                    "Yêu cầu user_id khi chưa đăng nhập.",
                    data=None,
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

        order = request_serializer.create(validated_data)
        response_serializer = OrderSerializer(order)
        return api_success(
            "Đơn hàng đã được tạo thành công!",
            {
                "order": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )

    @action(detail=False, methods=["post"], url_path="pos")
    def create_pos(self, request):
        # Admin authentication required (assuming handled by permissions or gateway, 
        # but locally added check if needed. Keeping generic for now as per instructions).
        return self.create(request) # Logic is same as online order usually, but maybe different payment method handling


    def update(self, request, pk=None):
        try:
            order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
            return api_error(
                "Đơn hàng không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        request_serializer = OrderSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        order = request_serializer.update(order, request_serializer.validated_data)
        response_serializer = OrderSerializer(order)
        return api_success(
            "Cập nhật đơn hàng thành công",
            {
                "order": response_serializer.data,
            },
        )

    def destroy(self, request, pk=None):
        try:
            order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
            return api_error(
                "Đơn hàng không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        order.delete()
        return api_success(
            "Xóa đơn hàng thành công",
            data=None,
        )

    @action(detail=True, methods=["post", "put"])
    def mark_paid(self, request, pk=None):
        try:
            order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
            return api_error(
                "Đơn hàng không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        order.mark_paid()
        updated_serializer = OrderSerializer(order)
        return api_success(
            "Đơn hàng đã được đánh dấu là đã thanh toán.",
            {
                "order": updated_serializer.data,
            },
        )

    @action(detail=True, methods=["post", "put"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def mark_delivered(self, request, pk=None):
        try:
            order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
            return api_error(
                "Đơn hàng không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        order.is_delivered = True
        order.delivered_at = datetime.utcnow()
        order.save()
        updated_serializer = OrderSerializer(order)
        return api_success(
            "Đơn hàng đã được giao.",
            {
                "order": updated_serializer.data,
            },
        )

    @action(detail=True, methods=["post", "put"])
    def cancel(self, request, pk=None):
        try:
            order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
            return api_error(
                "Đơn hàng không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        order.is_cancelled = True
        order.save()
        updated_serializer = OrderSerializer(order)
        return api_success(
            "Đơn hàng đã bị hủy.",
            {
                "order": updated_serializer.data,
            },
        )

    @action(detail=True, methods=["put", "patch"], url_path="status")
    def update_status(self, request, pk=None):
        try:
             order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
             return api_error("Không tìm thấy đơn hàng", status_code=status.HTTP_404_NOT_FOUND)
        
        serializer = OrderStatusUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        if "status" in serializer.validated_data:
            new_status = serializer.validated_data["status"]
            order.status = new_status
            
            # Sync booleans
            if new_status == "DA_GIAO_HANG" or new_status == "HOAN_THANH":
                order.is_delivered = True
                if not order.delivered_at:
                    order.delivered_at = datetime.utcnow()
            if new_status == "DA_HUY":
                order.is_cancelled = True

        if "paymentStatus" in serializer.validated_data:
            payment_status = serializer.validated_data["paymentStatus"]
            if payment_status == "PAID":
                order.is_paid = True
                if not order.paid_at:
                    order.paid_at = datetime.utcnow()
            elif payment_status == "UNPAID":
                order.is_paid = False
                order.paid_at = None
        
        order.save()
        
        return api_success(
            "Cập nhật trạng thái đơn hàng thành công", 
            {"order": OrderSerializer(order).data}
        )

    @action(detail=False, methods=["get"], url_path="my/orders", permission_classes=[permissions.AllowAny], authentication_classes=[])
    def my_orders(self, request):
        user_id = request.query_params.get("user_id")
        if not user_id:
            if request.user and hasattr(request.user, 'id') and request.user.is_authenticated:
                user_id = str(request.user.id)
            else:
                return api_error(
                    "Yêu cầu user_id khi chưa đăng nhập.",
                    data=None,
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
        queryset = Order.objects.filter(user_id=user_id).order_by("-created_at")

        page, page_size = get_pagination_params(request)
        orders, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = OrderSerializer(orders, many=True)
        return api_success(
            "Lấy danh sách đơn hàng thành công",
            {
                "orders": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )
    @action(detail=False, methods=["get"], url_path="user/(?P<user_id>[^/.]+)", permission_classes=[permissions.AllowAny], authentication_classes=[])
    def user_orders(self, request, user_id=None):
        if not user_id:
            return api_error("Yêu cầu ID người dùng.", status_code=status.HTTP_400_BAD_REQUEST)
        
        try:
            from bson import ObjectId
            user_id_obj = ObjectId(user_id)
        except Exception:
             return api_error("Định dạng ID người dùng không hợp lệ.", status_code=status.HTTP_400_BAD_REQUEST)

        queryset = Order.objects.filter(user_id=user_id_obj).order_by("-created_at")

        page, page_size = get_pagination_params(request)
        orders, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = OrderSerializer(orders, many=True)
        return api_success(
            "Lấy danh sách đơn hàng của người dùng thành công",
            {
                "orders": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )
