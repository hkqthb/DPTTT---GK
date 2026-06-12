"""
╔══════════════════════════════════════════════════════════════╗
║   Multi-Head Self-Attention - Interactive UI Launcher       ║
║                                                              ║
║   Chỉ cần chạy:  python run.py                              ║
║   Hoặc double-click file này trên Windows                    ║
║                                                              ║
║   Script sẽ tự động:                                         ║
║     1. Tạo virtual environment (nếu chưa có)                 ║
║     2. Cài đặt thư viện cần thiết                            ║
║     3. Khởi động web server                                  ║
║     4. Mở trình duyệt tại http://localhost:8000              ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import subprocess
import platform
import time
import threading
import webbrowser

# === CẤU HÌNH ===
PORT = 8000
HOST = "127.0.0.1"
VENV_DIR = ".venv"

def get_python_executable():
    """Lấy đường dẫn python trong venv."""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")

def get_pip_executable():
    """Lấy đường dẫn pip trong venv."""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "pip")

def print_banner():
    print()
    print("=" * 60)
    print("  🧠 Multi-Head Self-Attention - Interactive Visualizer")
    print("  📚 Đồ án Phân tích Độ phức tạp Thuật toán")
    print("=" * 60)
    print()

def step(msg):
    print(f"  ▸ {msg}")

def success(msg):
    print(f"  ✅ {msg}")

def error(msg):
    print(f"  ❌ {msg}")

def setup_venv():
    """Tạo virtual environment nếu chưa có."""
    python_exe = get_python_executable()
    
    if os.path.exists(python_exe):
        success("Virtual environment đã tồn tại")
        return True
    
    step("Đang tạo virtual environment...")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", VENV_DIR],
            check=True,
            capture_output=True
        )
        success("Tạo virtual environment thành công")
        return True
    except subprocess.CalledProcessError as e:
        error(f"Không thể tạo virtual environment: {e}")
        return False

def install_dependencies():
    """Cài đặt thư viện từ requirements.txt."""
    pip_exe = get_pip_executable()
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    
    if not os.path.exists(req_file):
        error("Không tìm thấy requirements.txt!")
        return False
    
    step("Đang kiểm tra và cài đặt thư viện...")
    try:
        result = subprocess.run(
            [pip_exe, "install", "-r", req_file, "-q"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            success("Tất cả thư viện đã sẵn sàng")
            return True
        else:
            # Thử lại không dùng -q để xem lỗi
            step("Đang cài đặt thư viện (có thể mất 1-2 phút lần đầu)...")
            result = subprocess.run(
                [pip_exe, "install", "-r", req_file],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                success("Cài đặt thư viện thành công")
                return True
            else:
                error(f"Lỗi cài đặt: {result.stderr}")
                return False
    except Exception as e:
        error(f"Lỗi: {e}")
        return False

def open_browser_delayed():
    """Mở trình duyệt sau 2 giây (chờ server khởi động)."""
    time.sleep(2.5)
    url = f"http://{HOST}:{PORT}"
    print()
    print(f"  🌐 Đang mở trình duyệt: {url}")
    print()
    webbrowser.open(url)

def start_server():
    """Khởi động uvicorn server."""
    python_exe = get_python_executable()
    
    print()
    print("─" * 60)
    print(f"  🚀 Server đang chạy tại: http://{HOST}:{PORT}")
    print(f"  💡 Mở trình duyệt và truy cập link trên để xem UI")
    print(f"  🛑 Nhấn Ctrl+C để tắt server")
    print("─" * 60)
    print()
    
    # Mở browser trong thread riêng
    browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
    browser_thread.start()
    
    # Chạy uvicorn (blocking)
    try:
        # Set encoding cho Windows
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        subprocess.run(
            [python_exe, "-m", "uvicorn", "app:app", 
             "--host", HOST, 
             "--port", str(PORT),
             "--reload"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env
        )
    except KeyboardInterrupt:
        print()
        print("  👋 Server đã tắt. Tạm biệt!")
        print()

def main():
    # Chuyển working directory về thư mục chứa script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print_banner()
    
    # Bước 1: Tạo venv
    step("Bước 1/3: Kiểm tra môi trường Python...")
    if not setup_venv():
        error("Không thể thiết lập môi trường. Hãy đảm bảo Python 3.8+ đã được cài đặt.")
        input("\nNhấn Enter để thoát...")
        sys.exit(1)
    
    # Bước 2: Cài dependencies
    step("Bước 2/3: Kiểm tra thư viện...")
    if not install_dependencies():
        error("Không thể cài đặt thư viện. Kiểm tra kết nối mạng và thử lại.")
        input("\nNhấn Enter để thoát...")
        sys.exit(1)
    
    # Bước 3: Chạy server
    step("Bước 3/3: Khởi động web server...")
    start_server()

if __name__ == "__main__":
    main()
