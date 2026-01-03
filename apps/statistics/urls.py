from rest_framework import routers
from .mongo_views import StatisticsViewSet

router = routers.DefaultRouter(trailing_slash=False)
router.register(r"statistics", StatisticsViewSet, basename="statistics")
