from rest_framework import routers
from .mongo_views import VoucherViewSet

router = routers.DefaultRouter(trailing_slash=False)
router.register(r"vouchers", VoucherViewSet, basename="voucher")
