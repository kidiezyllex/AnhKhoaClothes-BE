# API Endpoints Documentation (Used in Hooks)

This document lists the API endpoints that are currently utilized by the application's hooks, including their request payloads and response formats.

---

## 🔐 Authentication (Auth)

### 1. Register Account

- **Path**: `/auth/register`
- **Method**: `POST`
- **Hook**: `useRegister`
- **Payload JSON**:

```json
{
  "fullName": "Le Van A",
  "email": "vana@example.com",
  "password": "hashed_password",
  "phoneNumber": "0123456789",
  "role": "CUSTOMER",
  "gender": "male",
  "birthday": "2000-01-01",
  "citizenId": "123456789"
}
```

- **Response JSON**:

```json
{
  "success": true,
  "message": "Registration successful",
  "data": {
    "id": "user_id_123",
    "fullName": "Le Van A",
    "email": "vana@example.com",
    "role": "CUSTOMER",
    "token": "jwt_access_token"
  }
}
```

### 2. Login

- **Path**: `/auth/login`
- **Method**: `POST`
- **Hook**: `useLogin`
- **Payload JSON**:

```json
{
  "email": "vana@example.com",
  "password": "your_password"
}
```

- **Response JSON**: Same as Register Account.

---

## 👤 Account Management

### 1. Get All Accounts (Admin)

- **Path**: `/accounts`
- **Method**: `GET`
- **Hook**: `useAccounts`
- **Params**: `role`, `status`, `search`, `page`, `limit`
- **Response JSON**:

```json
{
  "success": true,
  "data": {
    "accounts": [
      {
        "id": "acc_id_123",
        "fullName": "Le Van A",
        "email": "vana@example.com",
        "phoneNumber": "0123456789",
        "role": "CUSTOMER",
        "status": "ACTIVE"
      }
    ],
    "pagination": { "totalItems": 100, "totalPages": 10, "currentPage": 1 }
  }
}
```

### 2. Profile & Security

- **Update Profile**: `useUpdateUserProfile` (PUT `/accounts/profile`)
- **Change Password**: `useChangePassword` (PUT `/accounts/profile/password`)
  - Payload: `{ "currentPassword": "...", "newPassword": "...", "confirmPassword": "..." }`

### 3. Address Management

- **Add Address**: `useAddAddress` (POST `/accounts/address`)
  - Payload: `{ "name": "...", "phoneNumber": "...", "provinceId": "...", "districtId": "...", "wardId": "...", "specificAddress": "..." }`
- **Update Address**: `useUpdateAddress` (PUT `/accounts/address/:addressId`)
- **Delete Address**: `useDeleteAddress` (DELETE `/accounts/address/:addressId`)

---

## 🏷️ Product Attributes (Admin)

### 1. Brands, Categories, Materials, Colors, Sizes

- **General Hooks**: `use[Attribute]s`, `use[Attribute]Detail`, `useCreate[Attribute]`, `useUpdate[Attribute]`, `useDelete[Attribute]`
- **Example Create (Brand)**: `{ "name": "Adidas", "status": "ACTIVE" }`
- **Example Create (Color)**: `{ "name": "Red", "code": "#FF0000", "status": "ACTIVE" }`
- **Example Create (Size)**: `{ "value": 42, "status": "ACTIVE" }`

---

## 📦 Products

### 1. Get All Products

- **Path**: `/products`
- **Method**: `GET`
- **Hook**: `useProducts`
- **Response JSON**:

```json
{
  "success": true,
  "data": {
    "products": [
      {
        "id": "prod_1",
        "name": "Nike Air Max",
        "price": 2500000,
        "stock": 50,
        "status": "ACTIVE",
        "variants": [
          { "color": "c1", "size": "s1", "price": 2500000, "stock": 50 }
        ]
      }
    ],
    "pagination": { "totalItems": 50, "totalPages": 5, "currentPage": 1 }
  }
}
```

### 2. Create Product (Admin)

- **Path**: `/products`
- **Method**: `POST`
- **Hook**: `useCreateProduct`
- **Payload JSON**:

```json
{
  "name": "Nike Air Max 2024",
  "brand": "brand_id",
  "category": "cat_id",
  "material": "mat_id",
  "description": "Premium running shoes",
  "weight": 550,
  "variants": [
    {
      "colorId": "color_id_1",
      "sizeId": "size_id_1",
      "price": 2500000,
      "stock": 100
    }
  ]
}
```

---

## 🛒 Orders

### 1. Create Online Order

- **Path**: `/orders`
- **Method**: `POST`
- **Hook**: `useCreateOrder`
- **Payload JSON**:

```json
{
  "customer": "cust_id",
  "items": [
    {
      "product": "p1",
      "variant": { "colorId": "c1", "sizeId": "s1" },
      "quantity": 1,
      "price": 500000
    }
  ],
  "subTotal": 500000,
  "discount": 50000,
  "total": 450000,
  "shippingAddress": {
    "name": "Le Van A",
    "phoneNumber": "0123456789",
    "provinceId": "p1",
    "districtId": "d1",
    "wardId": "w1",
    "specificAddress": "123 Street"
  },
  "paymentMethod": "COD",
  "orderId": "ORD_CUSTOM_123"
}
```

### 2. Create POS Order (Admin/Staff)

- **Path**: `/orders/pos`
- **Method**: `POST`
- **Hook**: `useCreatePOSOrder`
- **Payload**: Similar to Create Online Order, but often with `paymentMethod: "CASH"` or `BANK_TRANSFER`.

### 3. Order Status Management

- **Update Status**: `useUpdateOrderStatus` (PUT `/orders/:orderId/status`)
  - Payload: `{ "status": "DELIVERED" }` (Statuses: `CHO_XAC_NHAN`, `CHO_GIAO_HANG`, `DANG_VAN_CHUYEN`, `DA_GIAO_HANG`, `HOAN_THANH`, `DA_HUY`)

---

## 🎫 Vouchers

### 1. Validate Voucher

- **Path**: `/vouchers/validate`
- **Method**: `POST`
- **Hook**: `useValidateVoucher`
- **Payload**: `{ "code": "KEM20", "orderValue": 1000000 }`
- **Response JSON**:

```json
{
  "success": true,
  "data": {
    "voucher": {
      "code": "KEM20",
      "discountValue": 20,
      "discountType": "PERCENTAGE"
    },
    "discountAmount": 200000
  }
}
```

---

## 📈 Promotions (Admin)

### 1. Create Promotion

- **Path**: `/promotions`
- **Method**: `POST`
- **Hook**: `useCreatePromotion`
- **Payload JSON**:

```json
{
  "name": "Summer Sale",
  "discountValue": 15,
  "discountType": "PERCENTAGE",
  "startDate": "2024-07-01",
  "endDate": "2024-07-15",
  "applyTo": "ALL_PRODUCTS"
}
```

---

## 📊 Statistics (Admin)

### 1. Revenue & Reports

- **Revenue Report**: `useRevenueReport` (GET `/statistics/revenue?startDate=...&endDate=...&type=day`)
- **Top Products**: `useTopProducts` (GET `/statistics/top-products`)

---

## 🔄 Returns

### 1. Return Requests

- **Create Request**: `useCreateReturnRequest` (POST `/returns/request`)
  - Payload: `{ "originalOrder": "...", "items": [...], "reason": "Sản phẩm lỗi" }`
- **Update Status**: `useUpdateReturnStatus` (PUT `/returns/:returnId/status`)
  - Payload: `{ "status": "APPROVED" }`

---

## 🔔 Notifications

### 1. System Notifications

- **Create**: `useCreateNotification` (POST `/notifications`)
  - Payload: `{ "title": "Title", "content": "Content", "type": "SYSTEM" }`

---

## 📷 Upload

### 1. Upload Image

- **Path**: `/upload/image`
- **Method**: `POST`
- **Hook**: `useUploadImage`
- **Payload**: `Multipart/FormData` with key `file`.
- **Response**: `{ "success": true, "data": { "url": "...", "publicId": "..." } }`
