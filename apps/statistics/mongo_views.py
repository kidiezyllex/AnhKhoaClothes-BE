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
                return api_error(f"Lỗi khi tính toán thống kê hàng tháng: {str(e)}")

            return api_success("Thống kê hàng tháng", {"data": formatted_stats})

        # Default summary
        total_revenue = sum(float(o.total_price) for o in Order.objects(is_paid=True))
        total_orders = Order.objects.count()
        total_products = Product.objects.count()

        return api_success(
            "Thống kê chung",
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
        
        pipeline = [
            {"$match": {"is_paid": True}}
        ]
        
        if start_date_str and end_date_str:
            try:
                from datetime import datetime, timedelta
                # Parse inputs (YYYY-MM-DD expected)
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                
                # Make end_date inclusive of the full day
                end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)
                
                pipeline[0]["$match"]["created_at"] = {
                    "$gte": start_date,
                    "$lte": end_date
                }
            except ValueError:
                 return api_error("Định dạng ngày không hợp lệ. Sử dụng YYYY-MM-DD.", status_code=status.HTTP_400_BAD_REQUEST)

        # Aggregation to sum total_price and count orders
        pipeline.append({
            "$group": {
                "_id": None,
                "totalRevenue": {"$sum": "$total_price"},
                "orderCount": {"$sum": 1}
            }
        })
        
        try:
             result = list(Order.objects.aggregate(*pipeline))
             if result:
                 data = result[0]
                 total_revenue = float(data.get("totalRevenue", 0))
                 order_count = data.get("orderCount", 0)
             else:
                 total_revenue = 0
                 order_count = 0
                 
             return api_success(
                "Báo cáo doanh thu",
                {
                    "totalRevenue": total_revenue,
                    "orderCount": order_count,
                }
            )
        except Exception as e:
            return api_error(f"Lỗi khi tạo báo cáo doanh thu: {str(e)}")
    
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
        
        return api_success("Sản phẩm bán chạy nhất", {"products": result})
