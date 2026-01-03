from rest_framework import routers
from .mongo_views import UploadViewSet

router = routers.DefaultRouter(trailing_slash=False)
router.register(r"upload/image", UploadViewSet, basename="upload-image")
