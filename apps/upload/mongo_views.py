from rest_framework import permissions, status, viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from apps.utils import api_error, api_success
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

class UploadViewSet(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return api_error("Không có tệp tin nào được cung cấp", status_code=status.HTTP_400_BAD_REQUEST)
        
        # Save file locally
        # Should ideally use distinct names or Cloudinary
        import uuid
        ext = file_obj.name.split('.')[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        
        # Ensure media directory exists
        media_root = settings.MEDIA_ROOT
        if not os.path.exists(media_root):
             os.makedirs(media_root)

        # Simplified local storage logic
        path = default_storage.save(filename, ContentFile(file_obj.read()))
        
        # Construct URL
        # Assuming MEDIA_URL is /media/
        url = f"{request.scheme}://{request.get_host()}{settings.MEDIA_URL}{path}"
        
        return api_success(
            "Tải tệp tin lên thành công",
            {
                "url": url,
                "publicId": filename # Mock publicId
            }
        )
