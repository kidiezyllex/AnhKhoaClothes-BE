from __future__ import annotations

from bson import ObjectId
from rest_framework import serializers

from .mongo_models import Order, OrderItem, ShippingAddress
from apps.products.mongo_models import Product, Color, Size
from bson.errors import InvalidId

class OrderItemSerializer(serializers.Serializer):
    product_id = serializers.IntegerField()
    name = serializers.CharField(required=False, allow_blank=True)
    qty = serializers.IntegerField(required=False)
    size_selected = serializers.CharField(required=False, allow_blank=True)
    color_selected = serializers.CharField(required=False, allow_blank=True)
    images = serializers.ListField(child=serializers.CharField(), required=False, default=list)
    price_sale = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)

    def to_representation(self, instance):
        return {
            "product_id": instance.product_id,
            "name": instance.name,
            "qty": instance.qty,
            "size_selected": instance.size_selected,
            "color_selected": instance.color_selected,
            "images": instance.images,
            "price_sale": float(instance.price_sale),
        }

    def to_internal_value(self, data):
        # Convert camelCase to snake_case
        converted_data = {}
        field_mapping = {
            "product": "product_id",
            "productId": "product_id",
            "quantity": "qty",
            "price": "price_sale",
            "priceSale": "price_sale",
            "sizeSelected": "size_selected",
            "colorSelected": "color_selected",
        }
        
        for key, value in data.items():
            if key in field_mapping:
                converted_data[field_mapping[key]] = value
            else:
                converted_data[key] = value

        # Handle variant object if present
        if "variant" in data and isinstance(data["variant"], dict):
            variant = data["variant"]
            if "colorId" in variant and variant["colorId"]:
                try:
                    cid = variant["colorId"]
                    # Only try to fetch if it looks like a valid ID (int for SQL, but here we expect ObjectId or maybe legacy int)
                    # MongoEngine handles string-to-ObjectId conversion for 'id' field
                    color = Color.objects.get(id=cid)
                    converted_data["color_selected"] = color.name
                except Exception:
                    # If it's something like "1", it will fail if the collection uses ObjectIds
                    pass
            if "sizeId" in variant and variant["sizeId"]:
                try:
                    sid = variant["sizeId"]
                    size = Size.objects.get(id=sid)
                    converted_data["size_selected"] = size.code
                except Exception:
                    pass

        # Fetch missing product info
        if "product_id" in converted_data:
            try:
                product_id = int(converted_data["product_id"])
                product = Product.objects.get(id=product_id)
                if not converted_data.get("name"):
                    converted_data["name"] = product.productDisplayName or f"Product {product_id}"
                if not converted_data.get("images") or len(converted_data.get("images", [])) == 0:
                    converted_data["images"] = product.images if isinstance(product.images, list) else []
                if not converted_data.get("price_sale"):
                    # Check if there is a variant price, otherwise use product (wait, product doesn't have base price?)
                    # For now just default if missing
                    converted_data["price_sale"] = 0
            except (Product.DoesNotExist, ValueError):
                pass
        
        # Ensure default values for required fields in model if still missing
        if not converted_data.get("name"):
            converted_data["name"] = "Unknown Product"
        if not converted_data.get("size_selected"):
            converted_data["size_selected"] = "N/A"
        if not converted_data.get("color_selected"):
            converted_data["color_selected"] = "N/A"
        if "qty" not in converted_data:
            converted_data["qty"] = 1
        if "price_sale" not in converted_data:
            converted_data["price_sale"] = 0

        validated = super().to_internal_value(converted_data)
        if "product_id" in validated:
            validated["product_id"] = int(validated["product_id"])
        return validated

class ShippingAddressSerializer(serializers.Serializer):
    address = serializers.CharField(required=False, allow_blank=True)
    city = serializers.CharField(required=False, allow_blank=True)
    postal_code = serializers.CharField(required=False, allow_blank=True)
    country = serializers.CharField(required=False, allow_blank=True)
    recipient_phone_number = serializers.CharField(required=False, allow_blank=True)

    def to_representation(self, instance):
        if instance is None:
            return None
        return {
            "address": instance.address,
            "city": instance.city,
            "postal_code": instance.postal_code,
            "country": instance.country,
            "recipient_phone_number": instance.recipient_phone_number,
        }

    def to_internal_value(self, data):
        # Convert camelCase to snake_case
        converted_data = {}
        field_mapping = {
            "specificAddress": "address",
            "postalCode": "postal_code",
            "phoneNumber": "recipient_phone_number",
            "recipientPhoneNumber": "recipient_phone_number",
        }
        
        for key, value in data.items():
            if key in field_mapping:
                converted_data[field_mapping[key]] = value
            else:
                converted_data[key] = value
        
        # Fill in missing fields
        if not converted_data.get("city"):
            # Try to extract city from address if it contains common Vietnamese city patterns
            address = converted_data.get("address", "")
            if "Hà Nội" in address: converted_data["city"] = "Hà Nội"
            elif "Hồ Chí Minh" in address: converted_data["city"] = "Hồ Chí Minh"
            elif "Đà Nẵng" in address: converted_data["city"] = "Đà Nẵng"
            elif "Hà Giang" in address: converted_data["city"] = "Hà Giang"
            else: converted_data["city"] = "Other"
            
        if not converted_data.get("postal_code"):
            converted_data["postal_code"] = "100000"
            
        if not converted_data.get("country"):
            converted_data["country"] = "Vietnam"

        if not converted_data.get("address"):
            converted_data["address"] = "Unknown Address"

        if not converted_data.get("recipient_phone_number"):
            converted_data["recipient_phone_number"] = "0000000000"
        
        return super().to_internal_value(converted_data)

class OrderSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    user_id = serializers.CharField(write_only=True, required=False)
    payment_method = serializers.CharField()
    payment_result = serializers.DictField(default=dict)
    tax_price = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    shipping_price = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    total_price = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    is_paid = serializers.BooleanField(read_only=True)
    paid_at = serializers.DateTimeField(read_only=True, allow_null=True)
    is_delivered = serializers.BooleanField(read_only=True)
    delivered_at = serializers.DateTimeField(read_only=True, allow_null=True)
    is_cancelled = serializers.BooleanField(read_only=True)
    is_processing = serializers.BooleanField(read_only=True)
    is_outfit_purchase = serializers.BooleanField(required=False, default=False)
    status = serializers.CharField(read_only=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    items = OrderItemSerializer(many=True)
    shipping_address = ShippingAddressSerializer()

    def to_internal_value(self, data):
        # Convert camelCase to snake_case
        converted_data = {}
        field_mapping = {
            "orderItems": "items",
            "customerId": "user_id",
            "paymentMethod": "payment_method",
            "paymentResult": "payment_result",
            "itemsPrice": "items_price",
            "taxPrice": "tax_price",
            "shippingPrice": "shipping_price",
            "totalPrice": "total_price",
            "isOutfitPurchase": "is_outfit_purchase",
            "shippingAddress": "shipping_address",
            "total": "total_price",
        }
        
        for key, value in data.items():
            if key in field_mapping:
                converted_data[field_mapping[key]] = value
            else:
                converted_data[key] = value
        
        # Defaults for price fields if missing
        if "tax_price" not in converted_data:
            converted_data["tax_price"] = 0
        if "shipping_price" not in converted_data:
            converted_data["shipping_price"] = 0
        if "total_price" not in converted_data:
            # If total is not provided but subTotal is, maybe use subTotal
            if "subTotal" in data:
                converted_data["total_price"] = data["subTotal"]
            else:
                converted_data["total_price"] = 0
        
        # Remove items_price if present (not in model)
        converted_data.pop("items_price", None)
        
        return super().to_internal_value(converted_data)

    def get_id(self, obj):
        return str(obj.id)

    def to_representation(self, instance):
        data = {
            "id": str(instance.id),
            "user_id": str(instance.user_id),
            "payment_method": instance.payment_method,
            "payment_result": instance.payment_result,
            "tax_price": float(instance.tax_price),
            "shipping_price": float(instance.shipping_price),
            "total_price": float(instance.total_price),
            "is_paid": instance.is_paid,
            "paid_at": instance.paid_at,
            "is_delivered": instance.is_delivered,
            "delivered_at": instance.delivered_at,
            "is_cancelled": instance.is_cancelled,
            "is_processing": instance.is_processing,
            "is_outfit_purchase": instance.is_outfit_purchase,
            "status": getattr(instance, "status", "CHO_XAC_NHAN"),
            "created_at": instance.created_at,
            "updated_at": instance.updated_at,
            "items": [OrderItemSerializer(item).to_representation(item) for item in instance.items],
            "shipping_address": ShippingAddressSerializer(instance.shipping_address).to_representation(instance.shipping_address) if instance.shipping_address else None,
        }
        return data

    def create(self, validated_data):
        items_data = validated_data.pop("items")
        shipping_data = validated_data.pop("shipping_address")

        if "user_id" in validated_data:
            validated_data["user_id"] = ObjectId(validated_data["user_id"])

        order_items = []
        for item_data in items_data:
            # product_id is already converted to integer in OrderItemSerializer.to_internal_value
            order_items.append(OrderItem(**item_data))

        shipping_address = ShippingAddress(**shipping_data) if shipping_data else None

        order = Order(
            items=order_items,
            shipping_address=shipping_address,
            **validated_data
        )
        order.save()
        return order

    def update(self, instance, validated_data):
        items_data = validated_data.pop("items", None)
        shipping_data = validated_data.pop("shipping_address", None)

        for key, value in validated_data.items():
            setattr(instance, key, value)

        if shipping_data:
            instance.shipping_address = ShippingAddress(**shipping_data)

        if items_data is not None:
            order_items = []
            for item_data in items_data:
                # product_id is already converted to integer in OrderItemSerializer.to_internal_value
                order_items.append(OrderItem(**item_data))
            instance.items = order_items

        instance.save()
        return instance


class OrderStatusUpdateSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=[
        "CHO_XAC_NHAN", 
        "CHO_GIAO_HANG", 
        "DANG_VAN_CHUYEN", 
        "DA_GIAO_HANG", 
        "HOAN_THANH", 
        "DA_HUY"
    ])
