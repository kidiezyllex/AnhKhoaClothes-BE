from rest_framework import routers

from .mongo_views import (
    CategoryViewSet,
    ColorViewSet,
    ContentSectionViewSet,
    ProductViewSet,
    SizeViewSet,
    BrandViewSet,
    MaterialViewSet,
)
from .filter_views import ProductFilterViewSet

router = routers.DefaultRouter(trailing_slash=False)
router.register(r"products", ProductViewSet, basename="product")
router.register(r"filters", ProductFilterViewSet, basename="filter")
router.register(r"categories", CategoryViewSet, basename="category")
router.register(r"colors", ColorViewSet, basename="color")
router.register(r"sizes", SizeViewSet, basename="size")
router.register(r"brands", BrandViewSet, basename="brand")
router.register(r"materials", MaterialViewSet, basename="material")
router.register(r"content-sections", ContentSectionViewSet, basename="content-section")
