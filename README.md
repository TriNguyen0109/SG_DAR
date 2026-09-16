# Stable Diffusion 3.5 VisDial Baseline (Docker Setup & Guide)

Thư mục này chứa script baseline `sd3_5_visdial_baseline.py` sinh ảnh từ tập câu hỏi/hội thoại VisDial v1.0 bằng mô hình **Stable Diffusion 3.5**, kèm theo cấu hình **Docker tối ưu** và các phương pháp chạy ngắn gọn nhất.

---

## ⚡ Các cách chạy Siêu Ngắn Gọn (Không cần gõ lệnh Docker dài)

### Cách 1: Sử dụng Helper Script `./run.sh` (Nhanh nhất & Linh hoạt nhất ⭐⭐⭐⭐⭐)

Chúng tôi đã tạo sẵn script `run.sh` tự động bọc toàn bộ các tham số GPU, Mount Volume, Shared Memory dài dòng.

* **Chạy Dry-Run:**
  ```bash
  ./run.sh python3 sd3_5_visdial_baseline.py --dry_run
  ```
* **Chạy thật trên GPU RTX 3060 (Tối ưu VRAM & Dung lượng đĩa):**
  ```bash
  python3 sd3_5_visdial_baseline.py --skip_t5 --cpu_offload --skip_existing
  ```
* **Thay đổi GPU trực tiếp (Mặc định GPU 0 - RTX 3060):**
  ```bash
  GPU_IDS=0 ./run.sh python3 sd3_5_visdial_baseline.py --cpu_offload
  ```
* **Vào Terminal Bash của Container:**
  ```bash
  ./run.sh
  ```

---

### Cách 2: Sử dụng Docker Compose

Đã cấu hình sẵn file `compose.yaml`:

* **Chạy script Python bất kỳ:**
  ```bash
  docker compose run --rm app python3 sd3_5_visdial_baseline.py --dry_run
  ```
* **Vào môi trường Bash:**
  ```bash
  docker compose run --rm app bash
  ```

---

### Cách 3: Khởi chạy vào Shell 1 lần rồi gõ lệnh trực tiếp

Nếu bạn muốn gõ nhiều lệnh Python liên tục mà không muốn gõ lại cờ docker:

1. Mở môi trường Docker:
   ```bash
   ./run.sh
   ```
2. Sau khi đã ở trong container (`root@container:/workspace#`), bạn chỉ cần gõ bình thường:
   ```bash
   python3 sd3_5_visdial_baseline.py --cpu_offload
   python3 sd3_5_visdial_baseline.py --dry_run --max_dialogs 5
   ```
3. Gõ `exit` khi muốn thoát ra ngoài máy chủ.

---

## 📁 Cấu trúc Thư mục

```
diffusion/
├── Dockerfile                   # Cấu hình Docker image tối ưu
├── compose.yaml                 # Docker Compose cấu hình dịch vụ
├── run.sh                       # Wrapper script chạy lệnh Docker siêu ngắn
├── .dockerignore                # Bỏ qua các file rác khi build
├── requirements.txt             # Danh sách thư viện Python
├── sd3_5_visdial_baseline.py    # Script chính sinh ảnh VisDial
├── VisDial_v1_0_queries_val.json # Dataset VisDial v1.0
└── README.md                    # Hướng dẫn chi tiết
```
