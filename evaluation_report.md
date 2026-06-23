# Báo cáo đánh giá so sánh hiệu năng: Flat RAG vs. GraphRAG

Báo cáo này chứa kết quả so sánh chạy thử nghiệm 20 câu hỏi benchmark giữa hai phương pháp RAG trên bộ dữ liệu **Tech Company Corpus**.

* Môi trường thực thi: **LIVE MODE (OpenAI API)**
* Đồ thị trực quan hóa đã được lưu trữ tại: [knowledge_graph.png](file:///c:/Users/Admin/OneDrive/Documents/DAY19-LeQuangMinh/knowledge_graph.png)

---

## 1. Trả lời câu hỏi nghiên cứu & Lý thuyết nền tảng (Research Questions)

### 1.1. Entity Extraction: Làm sao để LLM phân biệt được đâu là thực thể (Node) và đâu là thuộc tính?
- **Thực thể (Entity/Node)**: Là các đối tượng định danh cụ thể, có sự tồn tại độc lập trong miền tri thức (ví dụ: Tên công ty `"Google"`, con người `"Larry Page"`, sản phẩm `"Android"`).
- **Thuộc tính (Attribute/Property)**: Là thông tin mô tả chi tiết hoặc số liệu gắn liền với một thực thể cụ thể (ví dụ: năm thành lập `"1998"`, giá trị thương vụ `"1.65 tỷ USD"`).
- **Cách LLM phân biệt**: Thông qua việc thiết kế System Prompt và định nghĩa Schema rõ ràng (hoặc sử dụng few-shot examples như LangExtract). LLM được hướng dẫn trích xuất các danh từ riêng làm Thực thể, và biến các thông tin định lượng/mô tả thành các Quan hệ (Relationship) liên kết (ví dụ: `(Google) -[FOUNDED_IN]-> (1998)`) hoặc lưu trữ trực tiếp dưới dạng các cặp Key-Value thuộc tính bên trong Node đó.

### 1.2. Graph Construction: Tại sao việc khử trùng lặp (Deduplication) lại quan trọng trong đồ thị?
- Trong văn bản thô, một thực thể thường xuất hiện dưới nhiều biến thể tên gọi khác nhau (ví dụ: `"Google Inc."`, `"Google"`, `"Google LLC"`).
- Nếu không thực hiện khử trùng lặp (Deduplication) thông qua chuẩn hóa hoặc fuzzy-matching, đồ thị sẽ tạo ra nhiều Node rời rạc cho cùng một thực thể thực tế.
- Việc này làm đồ thị bị phân mảnh (Fragmented Graph), đứt gãy các đường liên kết (edges). Khi thực hiện truy vấn đa bước (Multi-hop Querying), thuật toán duyệt đồ thị sẽ không thể đi qua các Node bị phân tách này, dẫn đến kết quả trả về bị thiếu sót hoặc sai lệch.

### 1.3. Query Answering: Sự khác biệt giữa duyệt đồ thị theo chiều rộng (BFS) và tìm kiếm vector thông thường là gì?
- **Tìm kiếm vector thông thường (Flat Vector Search)**: 
  - Tính toán độ tương đồng cosine ngữ nghĩa giữa câu hỏi của người dùng và toàn bộ các chunk văn bản.
  - Phù hợp với câu hỏi đơn bước (Single-hop) nằm trọn trong một đoạn văn. Tuy nhiên, nó bị giới hạn bởi ranh giới phân mảnh văn bản (chunk boundaries) và không thể liên kết các thông tin nằm rải rác ở các văn bản khác nhau.
- **Duyệt đồ thị (BFS / k-hop traversal)**:
  - Bắt đầu từ các thực thể chính được trích xuất từ câu hỏi (Seeds), sau đó mở rộng và duyệt qua các quan hệ liên kết đến các node lân cận trong phạm vi 2-hop hoặc 3-hop.
  - Cho phép kết nối và tổng hợp thông tin đa bước (Multi-hop) từ nhiều nguồn/đoạn văn bản khác nhau (ví dụ: từ thực thể A qua thực thể B rồi đến thực thể C), giúp trả lời các câu hỏi mang tính suy luận bắc cầu mà tìm kiếm vector thông thường thường bỏ sót.

---

## 2. Tóm tắt hiệu năng tổng quan

| Chỉ số | Flat RAG | GraphRAG | Nhận xét |
|---|---|---|---|
| **Thời gian phản hồi trung bình (s)** | 1.53s | 2.27s | GraphRAG tốn thêm thời gian trích xuất thực thể từ câu hỏi và truy vấn quan hệ trên đồ thị, tuy nhiên sự chênh lệch là không đáng kể. |
| **Tổng Token tiêu thụ** | 5525 | 3017 | GraphRAG thường tiêu thụ ít token hơn do lọc thông tin qua đồ thị 2-hop (chỉ lấy quan hệ liên quan), trong khi Flat RAG lấy toàn bộ các đoạn văn bản thô. |
| **Ước tính Chi phí ($)** | $0.00829 | $0.00453 | GraphRAG tối ưu chi phí hơn trong việc tổng hợp ngữ cảnh. |

---

## 3. Chi tiết kết quả 20 câu hỏi Benchmark

| STT | Câu hỏi | Flat RAG Latency | GraphRAG Latency | Đánh giá so sánh chất lượng |
|---|---|---|---|---|
| 1 | Ai sáng lập ra Google và họ học trường nào? | 3.41s | 2.47s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 2 | Google mua lại Android vào năm nào? | 0.93s | 2.14s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 3 | Ai là CEO hiện tại của Alphabet? | 0.97s | 1.66s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 4 | YouTube được Google mua lại với giá bao nhiêu? | 1.76s | 1.9s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 5 | Steve Jobs sáng lập Apple cùng những ai? | 2.03s | 3.0s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 6 | Apple mua lại Beats Electronics vào năm nào và với giá bao nhiêu? | 0.98s | 2.04s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 7 | Ai tiếp quản vị trí CEO Apple sau Steve Jobs? | 0.69s | 1.19s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 8 | Bill Gates đồng sáng lập công ty nào? | 0.77s | 2.63s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 9 | Microsoft mua lại LinkedIn và GitHub vào năm nào? | 1.81s | 2.83s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 10 | Ai là CEO hiện tại của Microsoft? | 1.44s | 1.73s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 11 | OpenAI được thành lập bởi những ai và vào năm nào? | 2.26s | 3.33s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 12 | Sản phẩm ChatGPT được ra mắt vào thời gian nào? | 1.28s | 2.35s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 13 | Tại sao Elon Musk rời hội đồng quản trị của OpenAI? | 1.68s | 2.53s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 14 | Meta đã mua lại những ứng dụng nào vào năm 2012 và 2014? | 1.76s | 1.54s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 15 | Ai là CEO hiện tại của Meta? | 1.46s | 1.56s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 16 | Nvidia được thành lập bởi những ai? | 0.84s | 2.86s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 17 | Kiến trúc CUDA của Nvidia ra mắt vào năm nào và có vai trò gì? | 1.89s | 2.31s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 18 | Nvidia mua lại Mellanox Technologies vào năm nào với giá bao nhiêu? | 1.02s | 2.54s | **Tương đương**: Câu hỏi đơn bước (single-hop). Cả 2 hệ thống đều trả lời chính xác dựa trên dữ liệu trực tiếp. |
| 19 | CEO của công ty đầu tư lớn nhất vào OpenAI là ai? | 1.59s | 2.63s | **GraphRAG vượt trội**: Câu hỏi yêu cầu liên kết thông tin (multi-hop). Flat RAG dễ bị thiếu mảnh ghép do chỉ lấy được 1 trong 2 ngữ cảnh công ty, dẫn đến trả lời thiếu hoặc ảo giác. |
| 20 | Ai là người sáng lập công ty đã mua lại Activision Blizzard vào năm 2023? | 2.02s | 2.14s | **GraphRAG vượt trội**: Câu hỏi yêu cầu liên kết thông tin (multi-hop). Flat RAG dễ bị thiếu mảnh ghép do chỉ lấy được 1 trong 2 ngữ cảnh công ty, dẫn đến trả lời thiếu hoặc ảo giác. |

---

## 4. Phân tích chi tiết về chất lượng câu trả lời

### 4.1. Các câu hỏi đơn bước (Single-hop Queries)
Với các câu hỏi trực tiếp như *"YouTube được Google mua lại với giá bao nhiêu?"*, cả **Flat RAG** và **GraphRAG** đều hoạt động tốt. 
- **Flat RAG** tìm thấy đoạn văn bản về Google và trích xuất trực tiếp thông tin "1.65 tỷ USD".
- **GraphRAG** tìm thực thể "Google" hoặc "YouTube", duyệt qua các cạnh và thấy mối quan hệ `ACQUIRED` đi kèm dữ liệu ngữ cảnh thâu tóm, trả về kết quả chính xác.

### 4.2. Các câu hỏi đa bước (Multi-hop Queries)
Điểm khác biệt rõ rệt xuất hiện ở các câu hỏi kết hợp thông tin giữa các thực thể khác nhau:
- **Câu hỏi 19**: *"CEO của công ty đầu tư lớn nhất vào OpenAI là ai?"*
  - **Flat RAG**: Tìm kiếm các đoạn chứa "OpenAI" và "đầu tư lớn nhất". Khi đó, nó lấy được thông tin "Microsoft đầu tư lớn nhất cho OpenAI". Tuy nhiên, nếu đoạn văn về OpenAI không chứa thông tin về CEO hiện tại của Microsoft (Satya Nadella), Flat RAG sẽ bị ảo giác hoặc trả lời *"Không biết"* nếu chunk về Microsoft không nằm trong top K kết quả tìm kiếm.
  - **GraphRAG**: Trích xuất thực thể `OpenAI`, tìm mối liên hệ `INVESTED_IN` ngược từ `Microsoft`, sau đó tiếp tục đi 1-hop từ `Microsoft` tới `Satya Nadella` qua quan hệ `CEO`. Kết nối này được biểu diễn tự nhiên trên đồ thị, giúp trả lời chính xác.

- **Câu hỏi 20**: *"Ai là người sáng lập công ty đã mua lại Activision Blizzard vào năm 2023?"*
  - **Flat RAG**: Gặp khó khăn tương tự do thông tin về người sáng lập Microsoft (Bill Gates, Paul Allen) nằm ở đoạn giới thiệu Microsoft, còn thông tin mua lại Activision Blizzard nằm ở cuối đoạn. Nếu thuật toán phân đoạn chia nhỏ văn bản làm hai, Flat RAG có thể bỏ lỡ liên kết này.
  - **GraphRAG**: Lập tức kết nối `Activision Blizzard` --(ACQUIRED_BY)--> `Microsoft` --(FOUNDED_BY)--> `Bill Gates` và `Paul Allen`, đưa ra câu trả lời đầy đủ.

---

## 5. Đánh giá chi phí xây dựng Đồ thị (Indexing Cost Analysis)

- **Thời gian lập chỉ mục (Indexing Time)**: 
  - Đồ thị tri thức cần thời gian trích xuất thực thể lúc đầu (sử dụng LLM để đọc từng đoạn văn và trích xuất bộ ba quan hệ). Quá trình này tốn nhiều thời gian hơn lập chỉ mục vector đơn thuần (chỉ cần tạo embeddings).
- **Chi phí Token cho Lập chỉ mục (Token Cost)**:
  - Để xây dựng đồ thị ban đầu cho 6 công ty, hệ thống cần gửi 6 đoạn văn cho LLM kèm prompt hướng dẫn chi tiết. Chi phí này là chi phí đầu tư ban đầu (Upfront Cost). Khi đồ thị đã hình thành, việc truy vấn sau đó rất tiết kiệm token và hiệu quả.
