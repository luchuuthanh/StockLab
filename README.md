# StockLab

Ứng dụng phân tích và sàng lọc cổ phiếu Việt Nam.

## Chạy trên Streamlit
- Main file: `stock_app.py`
- Dependencies: `requirements.txt`
- Workflow kiểm tra và cập nhật dữ liệu: `.github/workflows/stocklab.yml`

## Bộ lọc
- Bộ lọc nền: khối lượng trung bình 20 phiên và giá so với EMA100.
- Tab MOM10 và ROC14 độc lập.
- Biểu đồ kỹ thuật có MACD 12–26–9.
