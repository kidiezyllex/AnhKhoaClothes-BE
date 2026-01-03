from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from apps.utils import api_error, api_success, get_pagination_params, paginate_queryset
from .mongo_models import ReturnRequest
from .mongo_serializers import ReturnRequestSerializer, ReturnStatusUpdateSerializer
from bson import ObjectId

class ReturnRequestViewSet(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def list(self, request):
        # Filter by user if not admin, or just list all for admin? 
        # Requirement doesn't specify. Assuming admin sees all, user sees own.
        # Implemented simply for now.
        queryset = ReturnRequest.objects.all().order_by("-created_at")
        
        # If user is authenticated and not admin (need check), filter.
        # For now, simplistic implementation as per patterns in this codebase.
        if request.user and request.user.is_authenticated and not getattr(request.user, 'is_staff', False):
             queryset = queryset.filter(user_id=request.user.id)

        page, page_size = get_pagination_params(request)
        paginated, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = ReturnRequestSerializer(paginated, many=True)
        return api_success(
            "Return requests retrieved",
            {
                "returns": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "count": total_count
            }
        )

    @action(detail=False, methods=["post"], url_path="request")
    def create_request(self, request):
        if not request.user or not request.user.is_authenticated:
             return api_error("Login required", status_code=status.HTTP_401_UNAUTHORIZED)
             
        serializer = ReturnRequestSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        return_req = serializer.save()
        return api_success(
            "Return request created",
            {"returnRequest": ReturnRequestSerializer(return_req).data},
            status_code=status.HTTP_201_CREATED
        )

    @action(detail=True, methods=["put"], url_path="status")
    def update_status(self, request, pk=None):
        try:
            return_req = ReturnRequest.objects.get(id=ObjectId(pk))
        except ReturnRequest.DoesNotExist:
             return api_error("Return request not found", status_code=status.HTTP_404_NOT_FOUND)
        
        serializer = ReturnStatusUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        return_req.status = serializer.validated_data["status"]
        return_req.save()
        
        return api_success("Return status updated", {"returnRequest": ReturnRequestSerializer(return_req).data})
