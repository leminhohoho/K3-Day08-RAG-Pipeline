---
id: "day8-lab-rag-pipeline"
title: "Lab 08 — RAG Pipeline v2: Retrieval Hybrid, Vectorless Fallback & Generation có Citation"
duration: 180
author: "VinUni Codelab"
updated: "2026-08-02"
category: "RAG & Retrieval"
collection: "codelabs"
published: true
format: "steps"
day: "8"
preparationTipIds: ["huong-dan-cai-vs-code-va-git", "huong-dan-cai-python-va-cau-hinh-python-trong-vs-code", "thiet-lap-venv-voi-pip-va-uv", "huong-dan-tai-bai-lab"]
level: "intermediate"
prerequisites: ["Đã hoàn thành Lab 07 (chunking, vector store, RAG agent cơ bản)", "Biết dùng requests/HTTP client và đọc JSON", "Đã có tài khoản OpenRouter (hoặc OpenAI) và PageIndex"]
outcomes: ["Xây dựng pipeline thu thập → convert → chunk → index → retrieve → generate hoàn chỉnh", "Kết hợp semantic search (dense) và BM25 (sparse) bằng Reciprocal Rank Fusion", "Nhận biết khi nào retrieval fusion score KHÔNG phản ánh độ liên quan thật, và sửa đúng chỗ", "Triển khai fallback sang vectorless RAG (PageIndex) khi hybrid search yếu", "Sinh câu trả lời có citation, chống lost-in-the-middle bằng document reordering"]
supportedOs: ["Windows", "macOS", "Linux"]
requiredTools: ["Python 3.10+ (khuyến nghị 3.11)", "pip", "Git + tài khoản GitHub", "Tài khoản OpenRouter (API key)", "Tài khoản PageIndex (API key, tùy chọn)"]
commonErrors: ["Lỗi MissingDependencyException khi convert PDF do quên cài markitdown[pdf]", "Lỗi Executable doesn't exist do cài crawl4ai nhưng chưa gõ playwright install chromium", "Logic Fallback không bao giờ chạy do so nhầm điểm RRF (~0.016) thay vì điểm Cosine gốc (<0.48)", "Lỗi UnicodeEncodeError khi print tiếng Việt trên Windows console do thiếu PYTHONIOENCODING=utf-8", "Dữ liệu cũ và mới lẫn lộn do không xoá thư mục chroma_db/ cũ trước khi index lại", "Chạm Rate Limit (429) khi chạy RAGAS do gọi LLM quá nhiều lần liên tục"]
requiresSubmission: true
description: "Học viên xây dựng RAG pipeline 10 bước end-to-end: thu thập dữ liệu domain thật, chunking + indexing vào ChromaDB, hybrid retrieval (semantic + BM25 + RRF), vectorless fallback (PageIndex), và generation có citation chống lost-in-the-middle."
---

# 🎓 Lab 08 — RAG Pipeline v2 (Hướng Dẫn Chi Tiết Nguyên Lý & Thực Hành)

> 💡 **Dành cho học viên**: Bài lab này hướng dẫn bạn xây dựng hệ thống **RAG Pipeline chuyên nghiệp (v2)**. Bạn sẽ hiểu rõ lý do kỹ thuật đằng sau từng bước: từ khâu xử lý dữ liệu, kết hợp tìm kiếm ngữ nghĩa & từ khoá, đến kỹ thuật chống giảm chú ý (*lost-in-the-middle*) của mô hình ngôn ngữ lớn (LLM).

---

## 📖 0. Bảng Giải Thích Thuật Ngữ Kỹ Thuật Trực Quan

| Thuật ngữ gốc | Bản chất khái niệm | Minh hoạ trực quan |
| :--- | :--- | :--- |
| **RAG (Retrieval-Augmented Generation)** | **Mô hình Thi Mở Sách** | Mô hình LLM ban đầu chỉ trả lời bằng tri thức đóng gói sẵn (dễ bị ảo giác/chém gió). RAG bắt buộc AI tra cứu đúng tài liệu thực tế trước rồi mới tổng hợp ra câu trả lời. |
| **Chunking** | **Cắt tài liệu thành phân đoạn** | Văn bản dài hàng trăm trang vượt quá giới hạn cửa sổ ngữ cảnh (context window) của LLM. Chunking cắt nhỏ thành các đoạn 800 ký tự để tìm kiếm nhanh và chính xác. |
| **Embedding & Vector Store (ChromaDB)** | **Mã hoá toạ độ ngữ nghĩa** | Chuyển đổi văn bản thành dãy số toạ độ (vector 1024 chiều). Các đoạn văn có ý nghĩa tương đồng sẽ nằm gần nhau trong không gian toạ độ. |
| **Semantic Search (Dense Retrieval)** | **Tìm kiếm theo ý nghĩa** | Tìm kiếm dựa trên độ tương đồng Cosine giữa vector câu hỏi và vector tài liệu. Giúp tìm ra đáp án ngay cả khi người dùng dùng từ đồng nghĩa (*"Học phí"* vs *"Chi phí đào tạo"*). |
| **Lexical Search (BM25 Sparse)** | **Tìm kiếm theo từ khoá chính xác** | Tìm kiếm dựa trên tần suất xuất hiện từ khoá (TF-IDF / BM25). Cực kỳ hiệu quả với số hiệu văn bản, tên riêng, mã số (*"Quyết định 1234/QĐ-ĐHQGHN"*). |
| **HyDE (Hypothetical Document Embeddings)** | **Sinh câu trả lời giả định** | Cho LLM sinh một đoạn trả lời giả lập trước, sau đó dùng đoạn đó để truy vấn trong cơ sở dữ liệu. |
| **Query Expansion** | **Diễn đạt lại câu hỏi theo nhiều cách** | Dùng LLM sinh 2-3 biến thể/đồng nghĩa của câu hỏi gốc, search riêng từng biến thể rồi gộp kết quả (bằng RRF) — tăng recall khi người dùng dùng từ ngữ khác với tài liệu. |
| **RRF Reranking (Reciprocal Rank Fusion)** | **Thuật toán gộp thứ hạng** | Gộp thứ hạng từ Semantic Search và BM25 theo công thức $1/(60 + rank)$. Giải quyết triệt để vấn đề lệch thang điểm giữa Cosine `[0,1]` và BM25 `[0, ∞)`. |
| **PageIndex (Vectorless Fallback)** | **Truy vấn theo cấu trúc Mục Lục** | Khi câu hỏi mang tính tổng hợp cả chương/mục hoặc tìm kiếm từng chunk thất bại, hệ thống chuyển sang đọc cấu trúc cây của tài liệu mà không cần chunking. |
| **Lost-in-the-Middle** | **Hiện tượng Giảm chú ý ở giữa** | Hiện tượng các mô hình LLM ghi nhớ rất tốt thông tin ở **đầu** và **cuối** prompt, nhưng lại bỏ sót thông tin nằm ở **giữa**. |
| **Citation** | **Trích dẫn nguồn** | Yêu cầu LLM ghi rõ nguồn tài liệu tham khảo cho từng khẳng định trong câu trả lời. |
| **RAGAS Evaluation** | **Đánh giá tự động** | Sử dụng LLM độc lập để đo lường chất lượng câu trả lời và độ chính xác của tài liệu thu thập. |

---

## 👥 1. Quy Mô & Sơ Đồ Phân Vai Nhóm (4–6 Thành Viên)

Tùy theo số lượng thành viên thực tế của từng nhóm (4, 5 hoặc 6 người), nhóm lựa chọn sơ đồ phân công phù hợp bên dưới:

### 🔹 Phương Án A: Nhóm 4 Thành Viên (Cấu Trúc Chuẩn)
* **Role 1 (Team Leader & RAG Architect)**: Điều phối tiến độ, ghép code tổng hợp (`supervisor.py` & Task 9).
* **Role 2 (Data & Retrieval Specialist)**: Phụ trách thu thập, chuẩn hoá dữ liệu (Task 1–3) và xây dựng ChromaDB (Task 4–5).
* **Role 3 (Frontend & Chatbot Developer)**: Xây dựng giao diện Streamlit `app.py` và nối LLM Generation (Task 10).
* **Role 4 (Evaluation & QA Engineer)**: Tạo `golden_dataset.json` (15 câu hỏi), thực thi RAGAS `eval_pipeline.py` và viết `results.md`.

---

### 🔹 Phương Án B: Nhóm 5 Thành Viên (Chuyên Sâu Retrieval)
Tách phần tìm kiếm (Retrieval) thành 2 vị trí chuyên biệt:
* **Role 1 (Team Leader & RAG Architect)**: Quản lý chung, ghép code pipeline chính (`supervisor.py` & Task 9).
* **Role 2 (Data & Dense Search Dev)**: Task 1–3 (Data) + Task 4 (ChromaDB) + Task 5 (Semantic Search & HyDE).
* **Role 3 (Sparse Search & Advanced Reranking Dev)**: Task 6 (BM25/TF-IDF) + Task 7 (RRF Reranking) + Task 8 (PageIndex Fallback).
* **Role 4 (Frontend & Chatbot Developer)**: Xây dựng Streamlit Chatbot `app.py` + Task 10 (Generation có Citation).
* **Role 5 (Evaluation & QA Engineer)**: Bộ câu hỏi `golden_dataset.json` + Đánh giá RAGAS & báo cáo so sánh A/B `results.md`.

---

### 🔹 Phương Án C: Nhóm 6 Thành Viên (Mở Rộng Dữ Liệu & Benchmark)
Chia nhỏ các công đoạn dữ liệu và kiểm thử chuyên sâu:
* **Role 1 (Team Leader & RAG Architect)**: Quản lý nhóm, kiến trúc Supervisor và điều phối thuyết trình demo.
* **Role 2 (Data Engineering & Scraping Dev)**: Phụ trách Task 1 (tải PDF chính sách) + Task 2 (crawl bài viết tin tức) + Task 3 (convert Markdown).
* **Role 3 (Vector Database & Dense Search Dev)**: Task 4 (Chunking & ChromaDB Indexing) + Task 5 (Semantic Search & HyDE).
* **Role 4 (Sparse Retrieval & Fallback Dev)**: Task 6 (BM25 / TF-IDF) + Task 7 (RRF Reranking) + Task 8 (PageIndex Fallback).
* **Role 5 (Frontend UI & App Integration Dev)**: Thiết kế Streamlit Chatbot `app.py` + Task 10 (Citation Generation).
* **Role 6 (Evaluation & Benchmark QA Dev)**: Xây dựng `golden_dataset.json` mở rộng (20 câu hỏi) + Chạy RAGAS benchmark & viết báo cáo `results.md`.

---

## 🎯 2. Phân Công Vai Trò & Công Việc Theo Từng Checkpoint

### 🔹 Checkpoint 0: Setup Môi Trường & Khởi Tạo Project (0:00 – 0:10)
* 👑 **Role 1 (Team Leader & RAG Architect)**: Kiểm tra cả nhóm clone thành công repo Starter, khởi tạo repository chung cho nhóm và chia sẻ file `.env` với các API keys cần thiết (`OPENROUTER_API_KEY`).
* ⚙️ **Role 2 (Data & Pipeline Specialist / Data Dev)**: Tạo môi trường ảo (`python -m venv .venv`), cài đặt gói phụ thuộc từ `requirements.txt`, kiểm tra import `chromadb` và `sentence_transformers`.
* 🎨 **Role 3 (Frontend & Chatbot Dev)**: Kiểm tra cài đặt Streamlit bằng lệnh `streamlit run app.py`.
* 📊 **Role 4 / Role 5 / Role 6 (Evaluation & QA Engineer)**: Kiểm tra sự tồn tại và cài đặt của thư viện đánh giá `ragas` và `datasets`.
* ✅ **Tiêu chí hoàn thành (Pass Criteria)**: Tất cả các thành viên khởi tạo xong môi trường làm việc không có lỗi import (`CP0 Passed`).

---

### 🔹 Checkpoint 1: Thu Thập & Chuẩn Hoá Dữ Liệu — Task 1..3 (0:10 – 0:35)
* 👑 **Role 1 (Team Leader & RAG Architect)**: Kiểm tra phân công nguồn dữ liệu để tránh trùng lặp tài liệu giữa các thành viên.
* ⚙️ **Role 2 (Data & Pipeline Specialist / Data Dev)**: Thực hiện **Task 1** — Tải $\ge 3$ tài liệu quy định/chính sách gốc (PDF/DOCX) lưu vào `data/landing/legal/`.
* 🎨 **Role 3 (Frontend & Chatbot Dev)**: Thực hiện **Task 2** — Chạy script crawl $\ge 5$ bài viết/thông báo hướng dẫn lưu vào `data/landing/news/`.
* 📊 **Role 4 / Role 5 / Role 6 (Evaluation & QA Engineer)**: Thực hiện **Task 3** — Thực thi `python -m src.task3_convert_markdown` chuyển đổi toàn bộ tài liệu sang dạng Markdown trong `data/standardized/`.
* ✅ **Tiêu chí hoàn thành (Pass Criteria)**: Đủ $\ge 3$ file trong `legal/`, $\ge 5$ file trong `news/`, và đã có các file `.md` tương ứng trong `standardized/` (`CP1 Passed`).

---

### 🔹 Checkpoint 2: Chunking, Indexing & Search Cơ Bản — Task 4..6 (0:35 – 1:00)
* 👑 **Role 1 (Team Leader & RAG Architect)**: Kiểm tra tham số chunking (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`) và xác nhận việc sử dụng embedding model `BAAI/bge-m3`.
* ⚙️ **Role 2 (Data & Dense Search Dev)**: Thực hiện **Task 4** — Cắt đoạn văn bản, gọi model embedding và tạo cơ sở dữ liệu vector ChromaDB (`chroma_db/`).
* 🎨 **Role 3 (Sparse Search Dev / UI Dev)**: Thực hiện **Task 5** — Hoàn thiện hàm `semantic_search()` trong `src/task5_semantic_search.py` (Dense Retrieval dựa trên Cosine Similarity & HyDE).
* 📊 **Role 4 / Role 5 / Role 6 (Evaluation & QA Engineer)**: Thực hiện **Task 6** — Hoàn thiện hàm `lexical_search()` trong `src/task6_lexical_search.py` (Sparse Retrieval sử dụng BM25 & TF-IDF).
* ✅ **Tiêu chí hoàn thành (Pass Criteria)**: Khởi tạo xong `chroma_db/`; chạy `pytest tests/test_individual.py` vượt qua các kiểm thử của Task 4, 5, 6 (`CP2 Passed`).

---

### 🔹 Checkpoint 3: Reranking & Vectorless Fallback — Task 7..8 (1:00 – 1:20)
* 👑 **Role 1 (Team Leader & RAG Architect)**: Kiểm tra công thức gộp thứ hạng RRF ($k=60$) đảm bảo cân bằng giữa kết quả Semantic và BM25.
* ⚙️ **Role 2 (Pipeline Specialist / Sparse Dev)**: Thực hiện **Task 7** — Hoàn thiện hàm `rerank_rrf()` trong `src/task7_reranking.py`.
* 🎨 **Role 3 (Frontend & Chatbot Dev)**: Thực hiện **Task 8** — Tích hợp SDK PageIndex trong `src/task8_pageindex_vectorless.py` để xử lý truy vấn trên văn bản dạng cấu trúc.
* 📊 **Role 4 / Role 5 / Role 6 (Evaluation & QA Engineer)**: Thử nghiệm các câu hỏi ngoài domain để kiểm tra khả năng kích hoạt fallback của hệ thống.
* ✅ **Tiêu chí hoàn thành (Pass Criteria)**: Thuật toán RRF rerank gộp thành công kết quả từ 2 ranker; PageIndex trả về kết quả truy vấn phù hợp (`CP3 Passed`).

---

### 🔹 Checkpoint 4: Pipeline Hoàn Chỉnh & Generation — Task 9..10 (1:20 – 1:45)
* 👑 **Role 1 (Team Leader & RAG Architect)**: Kiểm tra toàn bộ mã nguồn bài cá nhân, chạy `pytest tests/test_individual.py` để xác nhận thành viên đạt đủ điểm bài cá nhân.
* ⚙️ **Role 2 (Data & Pipeline Specialist)**: Hoàn thiện **Task 9** (`src/task9_retrieval_pipeline.py`) — Nối chuỗi Semantic + BM25 + RRF + PageIndex Fallback khi điểm Cosine $< 0.48$.
* 🎨 **Role 3 (Frontend & Chatbot Dev)**: Hoàn thiện **Task 10** (`src/task10_generation.py`) — Áp dụng kỹ thuật Reordering (`front + back[::-1]`) và gọi LLM sinh câu trả lời có trích dẫn nguồn.
* 📊 **Role 4 / Role 5 / Role 6 (Evaluation & QA Engineer)**: Rà soát định dạng trích dẫn nguồn (citation format) trong câu trả lời từ LLM.
* ✅ **Tiêu chí hoàn thành (Pass Criteria)**: Chạy `pytest tests/test_individual.py` đạt **35/35 test passed** (Hoàn thành 50 điểm cá nhân) (`CP4 Passed`).

---

### 🔹 Checkpoint 5: Bài Tập Nhóm — Chatbot UI & Đánh Giá RAGAS (1:45 – 2:15)
* 👑 **Role 1 (Team Leader & RAG Architect)**: Phân công tổng hợp đoạn mã nguồn tối ưu nhất của nhóm vào `app.py` và theo dõi tiến độ hoàn thiện báo cáo.
* ⚙️ **Role 2 (Data & Pipeline Specialist)**: Kết nối hàm `generate_with_citation()` từ Task 10 vào luồng xử lý câu hỏi của `app.py`.
* 🎨 **Role 3 (Frontend & Chatbot Dev)**: Hoàn thiện ứng dụng Chatbot Streamlit (`app.py`), thiết kế giao diện chat, thanh cài đặt tham số `top_k`, vùng hiển thị danh sách tài liệu tham khảo và các câu hỏi gợi ý.
* 📊 **Role 4 / Role 5 / Role 6 (Evaluation & QA Engineer)**: Xây dựng `group_project/evaluation/golden_dataset.json` (15–20 câu hỏi), thực thi `python -m group_project.evaluation.eval_pipeline` để thu thập 4 chỉ số RAGAS và hoàn thiện báo cáo `group_project/evaluation/results.md`.
* ✅ **Tiêu chí hoàn thành (Pass Criteria)**: Chatbot UI phản hồi chính xác kèm danh sách nguồn; báo cáo `results.md` hiển thị đầy đủ bảng điểm đánh giá A/B testing (`CP5 Passed`).

---

### 🔹 Checkpoint 6: Thuyết Trình Demo & Nộp Bài (2:15 – 3:00)
* 👑 **Role 1 (Team Leader & RAG Architect)**: Thuyết trình tổng quan về ứng dụng Chatbot và kiến trúc RAG Pipeline trước lớp.
* ⚙️ **Role 2 (Data & Pipeline Specialist)**: Trả lời các câu hỏi kỹ thuật liên quan đến thuật toán Hybrid Search, RRF và Fallback logic từ Giảng viên / Coach.
* 🎨 **Role 3 (Frontend & Chatbot Dev)**: Trực tiếp thao tác và trình diễn ứng dụng Streamlit live demo trên màn hình chiếu.
* 📊 **Role 4 / Role 5 / Role 6 (Evaluation & QA Engineer)**: Báo cáo kết quả đánh giá RAGAS và đưa ra phân tích về hiệu quả giữa Hybrid Search vs Dense-Only.
* ✅ **Tiêu chí hoàn thành (Pass Criteria)**: Hoàn tất buổi thuyết trình demo và cập nhật toàn bộ mã nguồn lên repository GitHub của nhóm (`CP6 Passed`).

---

## 🧭 3. Lộ Trình & Các Checkpoint Quan Trọng (3.0 Giờ)

Đúng theo 7 checkpoint trong `checkpoint_timer.html` (tổng 180 phút = 3 giờ). Mỗi checkpoint đã
bao gồm sẵn vài phút review/demo ngẫu nhiên cuối chặng (xem "Pass Criteria" trong dashboard),
không cần thêm slot review riêng.

| Checkpoint | Thời gian | Mục tiêu phải đạt | File nộp / Kiểm tra |
| :--- | :---: | :--- | :--- |
| **CP0** 🟦 | 0:00–0:10 (10m) | Cài xong môi trường venv, có file `.env` chứa API Key | `pip install -r requirements.txt` |
| **CP1** 🟦 | 0:10–0:35 (25m) | Có $\ge 3$ PDF trong `legal/`, $\ge 5$ JSON trong `news/` và convert sang `.md` | `python -m src.task3_convert_markdown` |
| **CP2** 🟩 | 0:35–1:00 (25m) | Cắt đoạn văn bản, lưu ChromaDB, chạy thử Semantic & BM25 | `python -m src.task4_chunking_indexing` |
| **CP3** 🟩 | 1:00–1:20 (20m) | Viết thuật toán RRF Rerank gộp thứ hạng & tích hợp PageIndex | `python -m src.task7_reranking` |
| **CP4** 🟩 | 1:20–1:45 (25m) | **Mốc cá nhân 50đ**: Chạy Pytest đạt 35/35 PASSED | `python -m pytest tests/test_individual.py -v` |
| **CP5** 🟧 | 1:45–2:15 (30m) | **Mốc bài nhóm 50đ**: Chạy Chatbot Streamlit + Đánh giá RAGAS | `streamlit run app.py` |
| **CP6** 🟦 | 2:15–3:00 (45m) | **Thuyết trình Live Demo các nhóm (45 phút)** & Push code GitHub | `git push origin main` |

---

## 🛠️ 4. Hướng Dẫn Chi Tiết Từng Task (Task 1 $\rightarrow$ Task 10)

### 🔹 Task 1 & 2: Thu Thập Dữ Liệu Ban Đầu
- 💡 **Ý tưởng cốt lõi**: Thu thập tài liệu quy định thực tế (PDF) và các bài viết tin tức/thông báo (JSON).
- ❓ **Tại sao phải làm bước này?**: Các mô hình LLM chung (như GPT-4 hay Ling-Flash) không hề biết quy định nội bộ của từng trường đại học hay doanh nghiệp. Thu thập dữ liệu thực tế là bước bắt buộc để tạo "kho tri thức chuẩn" (Ground Truth) cho AI tra cứu.
- 🛠️ **Cách làm**:
  - **Task 1**: Tải $\ge 3$ file PDF chính sách (Học phí, Học bổng, Ký túc xá...) lưu vào `data/landing/legal/`.
  - **Task 2**: Chạy `python src/task2_crawl_news.py` để crawl $\ge 5$ bài tin tức lưu vào `data/landing/news/`.
- ⚡ **Mẹo tiết kiệm thời gian**: Nếu trang web bị chặn bot (Lỗi HTTP 403), bạn có thể dùng ngay bộ dữ liệu mẫu 11 file có sẵn trong repo!

---

### 🔹 Task 3: Chuyển Đổi Dữ Liệu Sang Markdown Standardized
- 💡 **Ý tưởng cốt lõi**: Chuyển đổi toàn bộ PDF/DOCX/JSON sang định dạng chuẩn thuần văn bản Markdown (`.md`).
- ❓ **Tại sao phải làm bước này?**: File PDF gốc chứa rất nhiều mã định dạng hiển thị phức tạp, hình ảnh, header/footer rác khiến LLM và các bộ phân đoạn (Text Splitter) bị nhiễu. Định dạng Markdown giúp giữ nguyên cấu trúc tiêu đề (`#`, `##`) và danh sách một cách sạch sẽ nhất.
- 🛠️ **Cách làm**: Chạy lệnh `python src/task3_convert_markdown.py`.
- ✅ **Kiểm tra**: Xem thư mục `data/standardized/legal/` và `news/` đã xuất hiện các file `.md` tương ứng chưa.

---

### 🔹 Task 4: Phân Đoạn & Lưu Trữ Vector (Chunking & Indexing)
- 💡 **Ý tưởng cốt lõi**: Cắt file Markdown thành các đoạn nhỏ (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`), chuyển thành vector 1024 chiều (model `BAAI/bge-m3`) và lưu vào ChromaDB.
- ❓ **Tại sao phải làm bước này?**:
  1. **Tại sao cắt 800 ký tự?**: Nếu đưa cả file 50 trang vào LLM sẽ rất tốn chi phí và làm LLM bị loãng thông tin.
  2. **Tại sao Overlap 100 ký tự?**: Tránh việc câu văn quan trọng bị cắt đôi ngay ở ranh giới giữa 2 đoạn.
  3. **Tại sao dùng ChromaDB?**: ChromaDB hỗ trợ truy vấn vector theo độ tương đồng Cosine chỉ trong vài miligiây thay vì phải quét thủ công từng file text.
- 🛠️ **Cách làm**: Chạy lệnh `python src/task4_chunking_indexing.py`.

---

### 🔹 Task 5 & 6: Tìm Kiếm Ngữ Nghĩa (Semantic) & Từ Khoá (BM25)
- 💡 **Ý tưởng cốt lõi**:
  - **Task 5 (Semantic Dense Search + HyDE)**: Tìm kiếm theo ý nghĩa ngữ nghĩa (Cosine Similarity).
  - **Task 6 (Lexical Sparse Search - BM25)**: Tìm kiếm theo từ khoá chính xác.
- ❓ **Tại sao phải làm CẢ HAI bước này?**:
  - Semantic Search giỏi tìm câu từ đồng nghĩa nhưng hay bỏ sót các từ khoá đặc biệt như số hiệu văn bản (*"Quyết định 456"*).
  - BM25 giỏi tìm từ khoá chính xác nhưng dốt khi người dùng dùng từ ngữ khác với tài liệu.
  - Kết hợp cả hai giúp hệ thống bù trừ khuyết điểm cho nhau (Hybrid Retrieval).
- 🛠️ **Cách làm**: Mở file `src/task5_semantic_search.py` và `src/task6_lexical_search.py` hoàn thiện hàm theo hướng dẫn.

---

### 🔹 Task 7: Gộp Thứ Hạng Reranking (RRF)
- 💡 **Ý tưởng cốt lõi**: Áp dụng thuật toán Reciprocal Rank Fusion $RRF(d) = \sum \frac{1}{60 + r(d)}$ để gộp thứ hạng từ Semantic Search và BM25.
- ❓ **Tại sao phải làm bước này?**: Điểm Cosine Similarity nằm trong khoảng `[0, 1]`, trong khi điểm BM25 là điểm thô không giới hạn (có thể từ `0` đến `20+`). Bạn **KHÔNG THỂ cộng trực tiếp hai loại điểm này**. Thuật toán RRF giải quyết vấn đề bằng cách chỉ dựa vào **thứ hạng (rank)** của đoạn văn trong từng danh sách để gộp điểm một cách công bằng.
- 🛠️ **Cách làm**: Chạy `python src/task7_reranking.py`.

---

### 🔹 Task 8: Vectorless RAG với PageIndex (Fallback)
- 💡 **Ý tưởng cốt lõi**: Tích hợp PageIndex SDK để truy vấn tài liệu theo cấu trúc Mục Lục (Tree Hierarchy) mà không qua chia nhỏ (chunking).
- ❓ **Tại sao phải làm bước này?**: Khi người dùng hỏi các câu hỏi mang tính tổng quan như *"Tóm tắt toàn bộ quy trình xin học bổng?"*, việc chia nhỏ thành các đoạn 800 ký tự sẽ làm mất đi bức tranh toàn cảnh của tài liệu. PageIndex giúp đọc hiểu cấu trúc chương/mục lớn để trả lời các câu hỏi tổng hợp.
- 🛠️ **Cách làm**: Điền `PAGEINDEX_API_KEY` vào `.env` và hoàn thiện `src/task8_pageindex_vectorless.py`.

---

### 🔹 Task 9: Nối Chuỗi Retrieval Pipeline Hoàn Chỉnh
- 💡 **Ý tưởng cốt lõi**: Nối chuỗi Semantic + BM25 $\rightarrow$ RRF Rerank. Nếu điểm Cosine gốc tốt nhất $< 0.48 \rightarrow$ Tự động chuyển sang PageIndex Fallback.
- ❓ **Tại sao phải chọn ngưỡng 0.48?**: Khi điểm Cosine Similarity tốt nhất $< 0.48$, điều đó chứng tỏ trong cơ sở dữ liệu ChromaDB **không có đoạn văn nào thực sự liên quan đến câu hỏi**. Nếu cố tình đưa đoạn rác cho LLM, LLM sẽ trả lời sai hoặc chém gió. Lúc này tự động chuyển sang PageIndex Fallback là giải pháp an toàn nhất.
- 🛠️ **Cách làm**: Hoàn thiện hàm `retrieve()` trong `src/task9_retrieval_pipeline.py`.
- 🚨 **BẪY QUAN TRỌNG**: So sánh điểm Cosine gốc `dense_results[0]['score'] < 0.48`, **KHÔNG** so sánh với điểm RRF đã gộp (điểm RRF luôn rất nhỏ ~0.016)!

---

### 🔹 Task 10: Document Reordering & Sinh Câu Trả Lời Có Citation
- 💡 **Ý tưởng cốt lõi**:
  1. Xếp lại các đoạn văn theo thứ tự xen kẽ `front + back[::-1]`.
  2. Yêu cầu LLM sinh câu trả lời kèm nhãn trích dẫn `[Nguồn tài liệu]`.
- ❓ **Tại sao phải làm bước này?**:
  - **Tại sao phải Reorder?**: Theo nghiên cứu *Lost in the Middle (Liu et al. 2023)*, LLM chú ý rất mạnh vào thông tin nằm ở **đầu** và **cuối** prompt, nhưng ngó lơ thông tin nằm ở **giữa**. Xếp đoạn quan trọng nhất vào đầu và cuối giúp LLM không bỏ sót dữ liệu.
  - **Tại sao phải ép Citation?**: Trích dẫn nguồn giúp kiểm chứng câu trả lời, minh bạch thông tin và ngăn chặn AI bịa đặt.
- 🛠️ **Cách làm**: Chạy `python -m src.task10_generation` (từ thư mục gốc repo, không cd vào src/).
- 🎯 **KIỂM TRA HOÀN THÀNH BÀI CÁ NHÂN**: Chạy lệnh `pytest tests/test_individual.py -v`. Khi màn hình báo **`35 passed`** là bạn đã đạt **50/50 điểm cá nhân**!

---

## 👥 3. Bài Tập Nhóm (Chatbot UI & Đánh Giá RAGAS)

1. **Giao diện Chatbot Streamlit (`app.py`)**:
   - Chạy lệnh `streamlit run app.py`.
   - Kết nối hàm `generate_with_citation()` từ Task 10 để tạo ứng dụng hỏi đáp hội thoại trực quan.
2. **Đánh giá tự động RAGAS (`group_project/evaluation/`)**:
   - Tạo bộ 15 câu hỏi kiểm thử trong `golden_dataset.json`.
   - Chạy `python -m group_project.evaluation.eval_pipeline` để thu thập 4 chỉ số (Faithfulness, Relevancy, Recall, Precision).
   - Xuất bảng so sánh kết quả vào file `results.md`.

---

## 🚨 4. Bảng Giải Mã & Khắc Phục Lỗi Thường Gặp (Troubleshooting)

Bảng dưới đây liệt kê các lỗi thực tế bạn sẽ thấy trên màn hình Terminal và cách gõ lệnh sửa ngay lập tức:

| 🔴 Bạn thấy lỗi gì trên màn hình? | ❓ Tại sao lại bị lỗi này? | ✅ Cách khắc phục nhanh (10 giây) |
| :--- | :--- | :--- |
| **`MissingDependencyException: markitdown requires pdf extra`** | Do bạn gõ `pip install markitdown` mà quên cài mô-đun đọc file PDF. | Gõ lệnh: `pip install "markitdown[pdf]"` |
| **`BrowserType.launch: Executable doesn't exist`** | Do thư viện `crawl4ai` chưa được tải trình duyệt Chromium về máy. | Gõ lệnh: `playwright install chromium` |
| **`UnicodeEncodeError: 'charmap' codec can't encode...`** | Do cửa sổ Console của Windows đang dùng bảng mã cũ (cp1252/cp1258). | Gõ lệnh: `$env:PYTHONIOENCODING="utf-8"` hoặc dùng `python -X utf8`. |
| **Hệ thống KHÔNG BAO GIỜ tự chuyển sang PageIndex Fallback** | Do bạn đem ngưỡng `0.48` so sánh với điểm RRF (luôn ~0.016) thay vì điểm Cosine gốc. | Sửa trong `task9`: Lấy `dense_results[0]['score'] < 0.48` làm điều kiện. |
| **Kết quả tìm kiếm trả về các đoạn văn rác từ bài cũ** | Do bạn thay đổi file dữ liệu nhưng chưa làm sạch cơ sở dữ liệu cũ trong máy. | Xoá thư mục `chroma_db/` bằng tay hoặc gõ: `Remove-Item -Recurse -Force chroma_db` rồi chạy lại Task 4. |
| **Báo lỗi Rate Limit `429 Too Many Requests` khi chạy RAGAS** | Do thư viện RAGAS gọi LLM quá nhiều lần liên tục chạm hạn mức OpenRouter free (50 req/ngày). | Tạm thời giảm số lượng câu hỏi trong `golden_dataset.json` xuống 5 câu khi chạy thử. |
