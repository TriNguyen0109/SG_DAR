# Chi Tiết Về Hoạt Động Của Các File Python (`.py`)

Tài liệu này giải thích chi tiết mục đích, luồng hoạt động, cấu trúc dữ liệu và các tham số dòng lệnh của tất cả các file Python trong dự án sinh ảnh VisDial với Stable Diffusion 3.5.

---

## 📋 Danh Sách File Python & Phân Định Vai Trò

| Tệp Python | Mục đích & Vai trò |
| :--- | :--- |
| **[prepare_sketch_dataset.py](file:///d:/hocAI/research/diffusion2/diffusion/prepare_sketch_dataset.py)** | Định dạng lại tệp JSON VisDial để bổ sung trường đường dẫn tệp sketch (`"sketch"`) cho từng lượt hội thoại. |
| **[sd3_5_visdial_baseline.py](file:///d:/hocAI/research/diffusion2/diffusion/sd3_5_visdial_baseline.py)** | **Text-Only Baseline:** Chỉ chạy từ câu thoại văn bản (`text`). Tự động bỏ qua trường sketch kể cả khi tệp JSON có thông tin sketch. Hiển thị tiến trình sinh từng tấm ảnh thực tế. |
| **[sd3_5_visdial_sketch.py](file:///d:/hocAI/research/diffusion2/diffusion/sd3_5_visdial_sketch.py)** | **Dual-Input (Text + Sketch):** Sinh ảnh kết hợp cả mô tả văn bản và ảnh phác thảo (sketch image condition). Hiển thị tiến trình và thời gian xử lý thực tế. |

---

## 1. `prepare_sketch_dataset.py`

### 🎯 Chức năng
Chỉ tập trung vào việc **chuyển đổi và định dạng tệp JSON VisDial v1.0**. Script trích xuất thông tin ảnh gốc và thêm đường dẫn tệp phác thảo tương ứng (`sketches/{stem}_{turn_idx}.png`) cho từng turn hội thoại mà không cần thao tác quản lý/sao chép tệp ảnh thực tế.

### 🔄 Luồng dữ liệu (JSON Formatting)
* **Đầu vào (Gốc):**
  ```json
  [
      {
          "img": "unlabeled2017/000000185565.jpg",
          "dialog": [
              "a bedroom is filled with lots of posters and a busy computer desk",
              "a vibrant bedroom filled with colorful posters..."
          ]
      }
  ]
  ```
* **Đầu ra (Đã bổ sung đường dẫn sketch):**
  ```json
  [
      {
          "img": "unlabeled2017/000000185565.jpg",
          "dialog": [
              {
                  "text": "a bedroom is filled with lots of posters and a busy computer desk",
                  "sketch": "sketches/000000185565_0.png"
              },
              {
                  "text": "a vibrant bedroom filled with colorful posters...",
                  "sketch": "sketches/000000185565_1.png"
              }
          ]
      }
  ]
  ```

---

## 2. `sd3_5_visdial_baseline.py` (Text-Only Baseline)

### 🎯 Chức năng & Hiển thị tiến trình thực tế
Script thực thi sinh ảnh **Text-to-Image thuần (Baseline)** từ hội thoại VisDial bằng Stable Diffusion 3.5.

#### 📊 Định dạng Log Tiến Trình Thời Gian Thực:
Mỗi tấm ảnh được sinh ra sẽ hiển thị chi tiết:
- **Thứ tự ảnh hiện tại / Tổng số ảnh cần sinh** & Phần trăm `%`.
- **Thời gian xử lý của từng tấm ảnh** (tính bằng giây/phút).
- **Tốc độ trung bình** (`Avg: X.XXs/img`) và **Thời gian dự kiến hoàn thành còn lại (ETA)**.

```text
[5/22 (22.7%)] Generating '0_4.jpg'...
✓ Saved '0_4.jpg' in 3.45s | Avg: 3.20s/img | ETA: 54s
```

---

## 3. `sd3_5_visdial_sketch.py` (Dual-Input Text + Sketch)

### 🎯 Chức năng
Script thực thi sinh ảnh **Dual-Input (Văn bản + Phác thảo)**. Mỗi lượt hội thoại sẽ kết hợp mô tả câu thoại và hình ảnh sketch tương ứng làm điều kiện đầu vào cho Stable Diffusion. Cũng hỗ trợ hiển thị tiến trình thời gian thực, thời gian sinh mỗi bức ảnh và ETA tự động.

---

## 💡 Ví Dụ Chạy Thử Super Fast (Dry-Run Mode)

```bash
# Test luồng Baseline Text-Only
python3 sd3_5_visdial_baseline.py --dry_run --max_dialogs 2

# Test luồng Dual-Input Text + Sketch
python3 sd3_5_visdial_sketch.py --dry_run --max_dialogs 2
```
