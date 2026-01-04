from __future__ import annotations

from rest_framework import permissions, viewsets
from rest_framework.decorators import action

from apps.utils import api_success

from .mongo_models import Product


class ProductFilterViewSet(viewsets.ViewSet):
    """
    ViewSet để trả về data bộ lọc product
    """
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def filter_options(self, request):
        """
        API trả về tất cả các giá trị unique cho bộ lọc product
        Endpoint: /api/v1/products/filters/filter_options/
        
        Returns:
        - articleTypes: Danh sách tất cả articleType
        - genders: Danh sách tất cả gender
        - baseColours: Danh sách tất cả baseColour (màu sắc)
        - seasons: Danh sách tất cả season
        - usages: Danh sách tất cả usage
        """
        try:
            # Lấy tất cả products
            products = Product.objects.only(
                "articleType", 
                "gender", 
                "baseColour", 
                "season", 
                "usage"
            ).all()

            # Sử dụng set để lấy giá trị unique
            article_types_set = set()
            genders_set = set()
            base_colours_set = set()
            seasons_set = set()
            usages_set = set()

            # Duyệt qua tất cả products và thu thập giá trị unique
            for product in products:
                # ArticleType
                if hasattr(product, "articleType") and product.articleType:
                    article_type = str(product.articleType).strip()
                    if article_type:
                        article_types_set.add(article_type)

                # Gender
                if hasattr(product, "gender") and product.gender:
                    gender = str(product.gender).strip()
                    if gender:
                        genders_set.add(gender)

                # BaseColour
                if hasattr(product, "baseColour") and product.baseColour:
                    base_colour = str(product.baseColour).strip()
                    if base_colour:
                        base_colours_set.add(base_colour)

                # Season
                if hasattr(product, "season") and product.season:
                    season = str(product.season).strip()
                    if season:
                        seasons_set.add(season)

                # Usage
                if hasattr(product, "usage") and product.usage:
                    usage = str(product.usage).strip()
                    if usage:
                        usages_set.add(usage)

            # Chuyển set thành list và sắp xếp
            article_types = sorted(list(article_types_set))
            genders = sorted(list(genders_set))
            base_colours = sorted(list(base_colours_set))
            seasons = sorted(list(seasons_set))
            usages = sorted(list(usages_set))

            return api_success(
                "Filter options retrieved successfully",
                {
                    "articleTypes": article_types,
                    "genders": genders,
                    "baseColours": base_colours,
                    "seasons": seasons,
                    "usages": usages,
                },
            )

        except Exception as e:
            from apps.utils import api_error
            from rest_framework import status
            return api_error(
                f"Error retrieving filter options: {str(e)}",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
