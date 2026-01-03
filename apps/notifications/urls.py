from rest_framework import routers
from .mongo_views import NotificationViewSet

router = routers.DefaultRouter(trailing_slash=False)
router.register(r"notifications", NotificationViewSet, basename="notification")
