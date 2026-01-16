from __future__ import annotations

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from rest_framework import routers

from apps.products.urls import router as products_router
from apps.orders.urls import router as orders_router
from apps.users.urls import router as users_router
from apps.recommendations.urls import router as recommendations_router
from apps.vouchers.urls import router as vouchers_router
from apps.promotions.urls import router as promotions_router
from apps.returns.urls import router as returns_router
from apps.notifications.urls import router as notifications_router
from apps.upload.urls import router as upload_router
from apps.statistics.urls import router as statistics_router
from apps.attributes.urls import router as attributes_router

class OptionalDefaultRouter(routers.DefaultRouter):

    def extend(self, router: routers.DefaultRouter) -> None:
        for prefix, viewset, basename in router.registry:
            self.register(prefix, viewset, basename=basename)

api_router = OptionalDefaultRouter(trailing_slash=False)
api_router.extend(users_router)
api_router.extend(products_router)
api_router.extend(orders_router)
api_router.extend(recommendations_router)
api_router.extend(vouchers_router)
api_router.extend(promotions_router)
api_router.extend(returns_router)
api_router.extend(notifications_router)
api_router.extend(upload_router)
api_router.extend(statistics_router)
api_router.extend(attributes_router)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include(api_router.urls)),
    path("api/v1/auth/", include("apps.users.auth_urls")),
    path("api/v1/hybrid/", include("apps.recommendations.hybrid.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

