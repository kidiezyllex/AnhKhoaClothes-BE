from rest_framework import routers
from .mongo_views import ReturnRequestViewSet

router = routers.DefaultRouter(trailing_slash=False)
router.register(r"returns", ReturnRequestViewSet, basename="return")
