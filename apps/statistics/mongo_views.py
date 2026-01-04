from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from apps.utils import api_error, api_success
from apps.orders.mongo_models import Order
from apps.products.mongo_models import Product

class StatisticsViewSet(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def list(self, request):
        stats_type = request.query_params.get("type", "GENERAL")
        
        if stats_type == "MONTHLY":
            pipeline = [
                {"$match": {"is_paid": True}},
                {
                    "$project": {
                        "month": {"$month": "$created_at"},
                        "year": {"$year": "$created_at"},
                        "total_price": "$total_price"
                    }
                },
                {
                    "$group": {
                        "_id": {"month": "$month", "year": "$year"},
                        "revenue": {"$sum": "$total_price"},
                        "orderCount": {"$sum": 1}
                    }
                },
                {"$sort": {"_id.year": -1, "_id.month": -1}}
            ]
            
            try:
                monthly_stats = list(Order.objects.aggregate(*pipeline))
                # Format for frontend
                formatted_stats = [
                    {
                        "month": item["_id"]["month"],
                        "year": item["_id"]["year"],
                        "revenue": float(item["revenue"]),
                        "orderCount": item["orderCount"]
                    }
                    for item in monthly_stats
                ]
            except Exception as e:
                return api_error(f"Error calculating monthly stats: {str(e)}")

            return api_success("Monthly statistics", {"data": formatted_stats})

        # Default summary
        total_revenue = sum(float(o.total_price) for o in Order.objects(is_paid=True))
        total_orders = Order.objects.count()
        total_products = Product.objects.count()

        return api_success(
            "General statistics",
            {
                "totalRevenue": total_revenue,
                "totalOrders": total_orders,
                "totalProducts": total_products
            }
        )

    @action(detail=False, methods=["get"], url_path="revenue")
    def revenue_report(self, request):
        start_date_str = request.query_params.get("startDate")
        end_date_str = request.query_params.get("endDate")
        
        # Mock logic or simple aggregation
        # Real implementation would require parsing dates and aggregating Order.total_price
        
        total_revenue = 0
        order_count = 0
        
        # Simplified: Sum all paid orders for now if no date provided, or implement filter
        queryset = Order.objects(is_paid=True)
        if start_date_str and end_date_str:
             # Add date filter logic here
             pass
             
        for order in queryset:
            total_revenue += float(order.total_price)
            order_count += 1
            
        return api_success(
            "Revenue report",
            {
                "totalRevenue": total_revenue,
                "orderCount": order_count,
                 # "chartData": ...
            }
        )
    
    @action(detail=False, methods=["get"], url_path="top-products")
    def top_products(self, request):
        # Already implemented in ProductViewSet.top but alias here or reimplement
        # Let's redirect logic or just fetch top products
        
        pipeline = [
            {"$unwind": "$items"},
            {"$group": {"_id": "$items.product_id", "totalSold": {"$sum": "$items.qty"}, "name": {"$first": "$items.name"}}},
            {"$sort": {"totalSold": -1}},
            {"$limit": 10}
        ]
        
        try:
            top_sold = list(Order.objects.aggregate(*pipeline))
        except:
             top_sold = []
             
        # Format response
        result = [
            {
                "id": item["_id"],
                "name": item.get("name", "Unknown"),
                "sold": item["totalSold"]
            }
            for item in top_sold
        ]
        
        return api_success("Top products", {"products": result})
