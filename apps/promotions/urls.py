from rest_framework import routers
from .mongo_views import PromotionViewSet

router = routers.DefaultRouter(trailing_slash=False)
router.register(r"promotions", PromotionViewSet, basename="promotion")
