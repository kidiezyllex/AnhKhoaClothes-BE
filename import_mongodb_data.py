"""
Script để import dữ liệu từ các file JSON vào MongoDB
Sử dụng để chuyển dữ liệu sang database mới
"""
import os
import json
from datetime import datetime
from pathlib import Path
import mongoengine
from pymongo import MongoClient
from bson import json_util
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cấu hình MongoDB đích (target database)
# Bạn có thể thay đổi các giá trị này hoặc tạo file .env.target
TARGET_MONGO_URI = os.getenv('TARGET_MONGO_URI', os.getenv('MONGO_URI'))
TARGET_DB_NAME = os.getenv('TARGET_MONGODB_DB_NAME', os.getenv('MONGODB_DB_NAME'))

# Thư mục chứa các file JSON cần import
IMPORT_DIR = Path('exports/mongodb_export')

def import_collection_from_json(db, json_file_path):
    """
    Import dữ liệu từ file JSON vào collection
    
    Args:
        db: MongoDB database instance
        json_file_path: Đường dẫn đến file JSON
    """
    try:
        collection_name = json_file_path.stem  # Lấy tên file không có extension
        
        # Bỏ qua file metadata
        if collection_name.startswith('_'):
            return None
        
        print(f"\n📥 Đang import collection: {collection_name}")
        
        # Đọc dữ liệu từ file JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f, object_hook=json_util.object_hook)
        
        if not documents:
            print(f"   ⚠️  File rỗng, bỏ qua")
            return 0
        
        collection = db[collection_name]
        
        # Kiểm tra xem collection đã tồn tại chưa
        existing_count = collection.count_documents({})
        if existing_count > 0:
            print(f"   ⚠️  Collection đã có {existing_count} documents")
            response = input(f"   ❓ Bạn muốn: [1] Xóa và import lại, [2] Thêm vào, [3] Bỏ qua? (1/2/3): ")
            
            if response == '1':
                collection.delete_many({})
                print(f"   🗑️  Đã xóa {existing_count} documents cũ")
            elif response == '3':
                print(f"   ⏭️  Bỏ qua collection này")
                return 0
            # response == '2' thì tiếp tục thêm vào
        
        # Insert documents
        if len(documents) == 1:
            result = collection.insert_one(documents[0])
            inserted_count = 1
        else:
            result = collection.insert_many(documents, ordered=False)
            inserted_count = len(result.inserted_ids)
        
        print(f"   ✅ Đã import {inserted_count} documents")
        return inserted_count
        
    except Exception as e:
        print(f"   ❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

def import_all_data(skip_confirmation=False):
    """Import tất cả dữ liệu từ các file JSON"""
    try:
        print("=" * 70)
        print("🚀 BẮT ĐẦU IMPORT DỮ LIỆU VÀO MONGODB")
        print("=" * 70)
        print(f"📊 Target Database: {TARGET_DB_NAME}")
        print(f"🔗 Target URI: {TARGET_MONGO_URI[:50]}...")
        print(f"📁 Import từ: {IMPORT_DIR.absolute()}")
        print("=" * 70)
        
        # Kiểm tra thư mục tồn tại
        if not IMPORT_DIR.exists():
            print(f"❌ Không tìm thấy thư mục: {IMPORT_DIR}")
            return
        
        # Lấy danh sách file JSON
        json_files = list(IMPORT_DIR.glob('*.json'))
        json_files = [f for f in json_files if not f.stem.startswith('_')]
        
        if not json_files:
            print("⚠️  Không tìm thấy file JSON nào để import!")
            return
        
        print(f"\n📋 Tìm thấy {len(json_files)} file JSON:")
        for f in json_files:
            file_size = f.stat().st_size / 1024  # KB
            print(f"   - {f.name} ({file_size:.1f} KB)")
        
        # Xác nhận trước khi import
        if not skip_confirmation:
            print("\n" + "=" * 70)
            print("⚠️  CẢNH BÁO: Script sẽ import dữ liệu vào database:")
            print(f"   Database: {TARGET_DB_NAME}")
            print(f"   URI: {TARGET_MONGO_URI[:50]}...")
            response = input("\n❓ Bạn có chắc chắn muốn tiếp tục? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("❌ Đã hủy import")
                return
        
        # Kết nối đến MongoDB
        print("\n🔌 Đang kết nối đến MongoDB...")
        client = MongoClient(TARGET_MONGO_URI)
        db = client[TARGET_DB_NAME]
        
        # Test connection
        client.server_info()
        print("✅ Kết nối thành công!")
        
        # Import từng collection
        total_imported = 0
        successful_imports = 0
        
        for json_file in json_files:
            count = import_collection_from_json(db, json_file)
            if count and count > 0:
                total_imported += count
                successful_imports += 1
        
        print("\n" + "=" * 70)
        print("✨ HOÀN THÀNH IMPORT DỮ LIỆU")
        print("=" * 70)
        print(f"📊 Tổng số file: {len(json_files)}")
        print(f"✅ Import thành công: {successful_imports}")
        print(f"📄 Tổng số documents: {total_imported}")
        print(f"💾 Database: {TARGET_DB_NAME}")
        print("=" * 70)
        
        # Đóng kết nối
        client.close()
        
    except Exception as e:
        print(f"\n❌ LỖI: {str(e)}")
        import traceback
        traceback.print_exc()

def import_specific_collections(collection_names, skip_confirmation=False):
    """
    Import chỉ các collection cụ thể
    
    Args:
        collection_names: List tên các collection cần import
        skip_confirmation: Bỏ qua xác nhận
    """
    try:
        print("=" * 70)
        print("🚀 IMPORT CÁC COLLECTION CỤ THỂ")
        print("=" * 70)
        print(f"📊 Target Database: {TARGET_DB_NAME}")
        print(f"📋 Collections: {', '.join(collection_names)}")
        print("=" * 70)
        
        # Kết nối đến MongoDB
        client = MongoClient(TARGET_MONGO_URI)
        db = client[TARGET_DB_NAME]
        
        total_imported = 0
        successful_imports = 0
        
        for collection_name in collection_names:
            json_file = IMPORT_DIR / f"{collection_name}.json"
            if not json_file.exists():
                print(f"\n⚠️  Không tìm thấy file: {json_file.name}")
                continue
            
            count = import_collection_from_json(db, json_file)
            if count and count > 0:
                total_imported += count
                successful_imports += 1
        
        print("\n" + "=" * 70)
        print("✨ HOÀN THÀNH")
        print("=" * 70)
        print(f"✅ Import thành công: {successful_imports}/{len(collection_names)}")
        print(f"📄 Tổng số documents: {total_imported}")
        print("=" * 70)
        
        client.close()
        
    except Exception as e:
        print(f"\n❌ LỖI: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 70)
    print("📦 MONGODB DATA IMPORT TOOL")
    print("=" * 70)
    print("\nChọn chế độ:")
    print("1. Import tất cả collections")
    print("2. Import các collection cụ thể")
    print("3. Thoát")
    
    choice = input("\nNhập lựa chọn (1/2/3): ")
    
    if choice == '1':
        import_all_data()
    elif choice == '2':
        collections_input = input("\nNhập tên các collection (cách nhau bởi dấu phẩy): ")
        collection_names = [c.strip() for c in collections_input.split(',')]
        import_specific_collections(collection_names)
    else:
        print("👋 Thoát chương trình")
