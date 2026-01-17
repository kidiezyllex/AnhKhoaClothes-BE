from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from apps.utils import api_success, api_error
from apps.products.mongo_models import Product, Color, Size
from apps.products.mongo_serializers import ColorSerializer, SizeSerializer
from apps.products.mongo_views import ensure_mongodb_connection

class AttributesViewSet(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    @action(detail=False, methods=["get"])
    def categories(self, request):
        """
        Trả về danh sách các articleType duy nhất từ Product.
        Endpoint: /api/v1/attributes/categories
        """
        try:
            ensure_mongodb_connection()
            
            # Sử dụng distinct để lấy các articleType không trùng lặp
            article_types = Product.objects.distinct("articleType")
            
            # Lọc bỏ các giá trị None hoặc rỗng và sắp xếp
            article_types = sorted([str(t).strip() for t in article_types if t and str(t).strip()])
            
            return api_success(
                "Lấy danh sách các loại sản phẩm thành công",
                {
                    "categories": article_types,
                    "count": len(article_types)
                }
            )
        except Exception as e:
            return api_error(f"Error retrieving article types: {str(e)}")

    @action(detail=False, methods=["get"])
    def colors(self, request):
        """
        Trả về danh sách các màu sắc.
        Endpoint: /api/v1/attributes/colors
        """
        try:
            ensure_mongodb_connection()
            colors = Color.objects.filter(status="ACTIVE").order_by("name")
            serializer = ColorSerializer(colors, many=True)
            return api_success(
                "Lấy danh sách màu sắc thành công",
                {
                    "colors": serializer.data,
                    "count": len(serializer.data)
                }
            )
        except Exception as e:
            return api_error(f"Error retrieving colors: {str(e)}")

    @action(detail=False, methods=["get"])
    def sizes(self, request):
        """
        Trả về danh sách các kích thước.
        Endpoint: /api/v1/attributes/sizes
        """
        try:
            ensure_mongodb_connection()
            sizes = Size.objects.filter(status="ACTIVE").order_by("name")
            serializer = SizeSerializer(sizes, many=True)
            return api_success(
                "Lấy danh sách kích thước thành công",
                {
                    "sizes": serializer.data,
                    "count": len(serializer.data)
                }
            )
        except Exception as e:
            return api_error(f"Error retrieving sizes: {str(e)}")
