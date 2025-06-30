import sqlite3  # Thư viện để làm việc với SQLite database
import pickle   # Thư viện dùng để serialize (nén) dữ liệu phức tạp (như mảng numpy)

DB_NAME = "database.sqlite"  # Tên file cơ sở dữ liệu SQLite sẽ lưu khuôn mặt

# Hàm khởi tạo database nếu chưa có
def init_db():
    conn = sqlite3.connect(DB_NAME)  # Kết nối tới database (sẽ tạo file nếu chưa tồn tại)
    c = conn.cursor()
    # Tạo bảng faces nếu chưa có, lưu tên và encoding (vector khuôn mặt)
    c.execute('''CREATE TABLE IF NOT EXISTS faces (
                    name TEXT PRIMARY KEY,
                    encoding BLOB
                )''')
    conn.commit()  # Ghi thay đổi vào file
    conn.close()   # Đóng kết nối database

# Hàm lưu khuôn mặt mới vào database
def save_face(name, encoding):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Lưu tên và encoding. Dùng pickle để nén vector encoding thành nhị phân (BLOB).
    # REPLACE: Nếu tên đã tồn tại thì ghi đè
    c.execute("REPLACE INTO faces (name, encoding) VALUES (?, ?)", (name, pickle.dumps(encoding)))
    conn.commit()
    conn.close()

# Hàm load tất cả khuôn mặt đã lưu từ database
def load_known_faces():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM faces")  # Lấy tất cả tên và encoding từ bảng
    rows = c.fetchall()
    conn.close()

    names = []      # Danh sách tên
    encodings = []  # Danh sách encoding (vector khuôn mặt)
    for name, encoding in rows:
        names.append(name)
        encodings.append(pickle.loads(encoding))  # Giải nén từ BLOB thành numpy array
    return names, encodings  # Trả về 2 danh sách: tên và vector encoding
def delete_all_faces():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM faces")  # Xóa toàn bộ dữ liệu trong bảng faces
    conn.commit()
    conn.close()
