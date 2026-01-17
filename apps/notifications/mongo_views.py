from rest_framework import permissions, status, viewsets
from apps.utils import api_error, api_success, get_pagination_params, paginate_queryset
from .mongo_models import Notification
from .mongo_serializers import NotificationSerializer

class NotificationViewSet(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def list(self, request):
        queryset = Notification.objects.all().order_by("-created_at")
        if request.user and request.user.is_authenticated:
             # Show system notifications OR user specific
             queryset = queryset.filter(__raw__={"$or": [{"userId": request.user.id}, {"userId": None}]})

        page, page_size = get_pagination_params(request)
        paginated, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = NotificationSerializer(paginated, many=True)
        return api_success(
            "Lấy danh sách thông báo thành công",
            {
                "notifications": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "count": total_count
            }
        )

    def create(self, request):
        serializer = NotificationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        notification = serializer.save()
        return api_success(
            "Tạo thông báo thành công",
            {"notification": NotificationSerializer(notification).data},
            status_code=status.HTTP_201_CREATED
        )
