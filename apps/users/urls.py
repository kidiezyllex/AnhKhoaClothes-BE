from rest_framework import routers

from .mongo_views import OutfitHistoryViewSet, UserInteractionViewSet, UserViewSet, UserAddressViewSet
from .admin_stats_views import AdminStatsViewSet

router = routers.DefaultRouter(trailing_slash=False)
router.register(r"accounts/address", UserAddressViewSet, basename="user-address")
router.register(r"accounts", UserViewSet, basename="user")
router.register(r"users", UserViewSet, basename="user-legacy") # Keep legacy route if needed or remove
router.register(r"user-interactions", UserInteractionViewSet, basename="user-interaction")
router.register(r"outfit-history", OutfitHistoryViewSet, basename="outfit-history")
router.register(r"admin-stats", AdminStatsViewSet, basename="admin-stats")

