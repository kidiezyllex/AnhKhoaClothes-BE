"""
Script để xuất tất cả dữ liệu từ MongoDB về dưới dạng JSON
"""
import os
import json
from datetime import datetime
from pathlib import Path
import mongoengine
from pymongo import MongoClient
from bson import ObjectId, json_util
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection settings
MONGO_URI = os.getenv('MONGO_URI')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'allwear')

# Tạo thư mục để lưu exports
EXPORT_DIR = Path('exports/mongodb_export')
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def custom_json_encoder(obj):
    """Custom JSON encoder để xử lý các kiểu dữ liệu MongoDB"""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def export_collection_to_json(db, collection_name, export_dir):
    """
    Xuất một collection ra file JSON
    
    Args:
        db: MongoDB database instance
        collection_name: Tên collection cần xuất
        export_dir: Thư mục để lưu file
    """
    try:
        collection = db[collection_name]
        documents = list(collection.find())
        
        if not documents:
            print(f"⚠️  Collection '{collection_name}' không có dữ liệu")
            return
        
        # Tạo file JSON
        file_path = export_dir / f"{collection_name}.json"
        
        # Sử dụng json_util từ bson để serialize đúng cách
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2, default=json_util.default)
        
        print(f"✅ Đã xuất {len(documents)} documents từ '{collection_name}' -> {file_path}")
        return len(documents)
        
    except Exception as e:
        print(f"❌ Lỗi khi xuất collection '{collection_name}': {str(e)}")
        return 0

def export_all_data():
    """Xuất tất cả dữ liệu từ MongoDB"""
    try:
        print("=" * 70)
        print("🚀 BẮT ĐẦU XUẤT DỮ LIỆU TỪ MONGODB")
        print("=" * 70)
        print(f"📊 Database: {MONGODB_DB_NAME}")
        print(f"📁 Thư mục xuất: {EXPORT_DIR.absolute()}")
        print("=" * 70)
        
        # Kết nối đến MongoDB
        client = MongoClient(MONGO_URI)
        db = client[MONGODB_DB_NAME]
        
        # Lấy danh sách tất cả collections
        collection_names = db.list_collection_names()
        
        if not collection_names:
            print("⚠️  Không tìm thấy collection nào trong database!")
            return
        
        print(f"\n📋 Tìm thấy {len(collection_names)} collections:")
        for name in collection_names:
            print(f"   - {name}")
        print()
        
        # Xuất từng collection
        total_documents = 0
        successful_exports = 0
        
        for collection_name in collection_names:
            count = export_collection_to_json(db, collection_name, EXPORT_DIR)
            if count and count > 0:
                total_documents += count
                successful_exports += 1
        
        # Tạo file metadata
        metadata = {
            "export_date": datetime.now().isoformat(),
            "database_name": MONGODB_DB_NAME,
            "total_collections": len(collection_names),
            "successful_exports": successful_exports,
            "total_documents": total_documents,
            "collections": collection_names
        }
        
        metadata_path = EXPORT_DIR / "_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 70)
        print("✨ HOÀN THÀNH XUẤT DỮ LIỆU")
        print("=" * 70)
        print(f"📊 Tổng số collections: {len(collection_names)}")
        print(f"✅ Xuất thành công: {successful_exports}")
        print(f"📄 Tổng số documents: {total_documents}")
        print(f"📁 Dữ liệu đã được lưu tại: {EXPORT_DIR.absolute()}")
        print(f"ℹ️  File metadata: {metadata_path}")
        print("=" * 70)
        
        # Đóng kết nối
        client.close()
        
    except Exception as e:
        print(f"\n❌ LỖI: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_all_data()
