from rest_framework import permissions, status, viewsets
from apps.utils import api_error, api_success, get_pagination_params, paginate_queryset
from .mongo_models import Promotion
from .mongo_serializers import PromotionSerializer
from bson import ObjectId

class PromotionViewSet(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def list(self, request):
        queryset = Promotion.objects.all().order_by("-created_at")
        page, page_size = get_pagination_params(request)
        paginated, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = PromotionSerializer(paginated, many=True)
        return api_success(
            "Lấy danh sách khuyến mãi thành công",
            {
                "promotions": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            }
        )

    def create(self, request):
        serializer = PromotionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        promotion = serializer.save()
        return api_success(
            "Tạo khuyến mãi thành công",
            {"promotion": PromotionSerializer(promotion).data},
            status_code=status.HTTP_201_CREATED
        )

    def retrieve(self, request, pk=None):
        try:
            promotion = Promotion.objects.get(id=ObjectId(pk))
        except Promotion.DoesNotExist:
            return api_error("Promotion not found", status_code=status.HTTP_404_NOT_FOUND)
        return api_success("Lấy thông tin khuyến mãi thành công", {"promotion": PromotionSerializer(promotion).data})

    def update(self, request, pk=None):
        try:
            promotion = Promotion.objects.get(id=ObjectId(pk))
        except Promotion.DoesNotExist:
             return api_error("Promotion not found", status_code=status.HTTP_404_NOT_FOUND)
        
        serializer = PromotionSerializer(promotion, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return api_success("Cập nhật khuyến mãi thành công", {"promotion": serializer.data})
        
    def destroy(self, request, pk=None):
        try:
             promotion = Promotion.objects.get(id=ObjectId(pk))
             promotion.delete()
             return api_success("Xóa khuyến mãi thành công")
        except Promotion.DoesNotExist:
             return api_error("Promotion not found", status_code=status.HTTP_404_NOT_FOUND)
