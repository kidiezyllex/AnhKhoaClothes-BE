from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from .mongo_models import Product
from apps.utils import api_success, api_error

class ProductFilterViewSet(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    @action(detail=False, methods=["get"])
    def filter_options(self, request):
        try:
            # Import connection helper
            from .mongo_views import ensure_mongodb_connection
            ensure_mongodb_connection()

            # Query only necessary fields to optimize performance
            products = Product.objects.only(
                "articleType", "gender", "baseColour", "season", "usage"
            ).all()

            article_types = set()
            genders = set()
            base_colours = set()
            seasons = set()
            usages = set()

            for product in products:
                # articleType
                val = getattr(product, "articleType", None)
                if val and str(val).strip():
                    article_types.add(str(val).strip())

                # gender
                val = getattr(product, "gender", None)
                if val and str(val).strip():
                    genders.add(str(val).strip())

                # baseColour
                val = getattr(product, "baseColour", None)
                if val and str(val).strip():
                    base_colours.add(str(val).strip())

                # season
                val = getattr(product, "season", None)
                if val and str(val).strip():
                    seasons.add(str(val).strip())

                # usage
                val = getattr(product, "usage", None)
                if val and str(val).strip():
                    usages.add(str(val).strip())

            data = {
                "articleTypes": sorted(list(article_types)),
                "genders": sorted(list(genders)),
                "baseColours": sorted(list(base_colours)),
                "seasons": sorted(list(seasons)),
                "usages": sorted(list(usages)),
            }

            return api_success("Filter options retrieved successfully", data)

        except Exception as e:
            return api_error(f"Error retrieving filter options: {str(e)}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
