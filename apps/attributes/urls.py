from rest_framework import routers
from .mongo_views import AttributesViewSet

router = routers.DefaultRouter(trailing_slash=False)
router.register(r"attributes", AttributesViewSet, basename="attribute")
