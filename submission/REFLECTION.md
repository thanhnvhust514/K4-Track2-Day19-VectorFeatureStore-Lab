# Reflection — Lab 19

**Tên:** Nguyễn Việt Thanh  
**Cohort:** A20-K4  
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

Trên golden set, BM25 thường mạnh với query `exact` vì query chứa đúng keyword kỹ thuật xuất hiện trong corpus. Semantic search hữu ích hơn khi query là paraphrase, dù model `bge-small-en` trong lite path chưa tối ưu cho tiếng Việt nên kết quả có thể chưa vượt trội. Hybrid RRF tốt nhất ở nhóm `mixed` và thường thắng trung bình vì kết hợp tín hiệu keyword chính xác với tín hiệu semantic. Tôi sẽ không dùng hybrid khi latency/cost rất chặt, corpus nhỏ, query gần như toàn exact keyword, hoặc khi retriever còn lại thêm nhiều noise hơn giá trị.

---

## Điều ngạc nhiên nhất khi làm lab này

Embedding model ảnh hưởng rất rõ tới chất lượng search tiếng Việt; đổi model có thể quan trọng không kém đổi thuật toán retrieval.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _N/A_
