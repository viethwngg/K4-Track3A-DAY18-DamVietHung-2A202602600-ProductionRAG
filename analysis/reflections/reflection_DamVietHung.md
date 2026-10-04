# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Đàm Việt Hưng  
**MSSV:** 2A202602600 — **Khóa:** K4, Track 3A  
**Ngày hoàn thành:** 04/10/2026

## Phần 1: Mapping bài giảng

| Lecture concept | Module | Hàm cụ thể | Observation và phân tích |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()`, `compare_strategies()` | Encode từng câu bằng all-MiniLM-L6-v2, chuẩn hóa vector rồi so cosine của hai câu liên tiếp. Threshold cao tạo nhiều điểm tách hơn; cần đo số chunks và khả năng giữ bằng chứng thay vì chỉ chọn threshold theo cảm giác. Hierarchical là lựa chọn chính cho pipeline vì retrieve child nhỏ nhưng trả parent đủ ngữ cảnh. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | Cả corpus và query được chuẩn hóa Unicode, lowercase và bỏ `_` từ underthesea. RRF cộng `1/(60+rank+1)` nên không cần so trực tiếp thang điểm BM25 và cosine. Một hit bị lặp trong cùng danh sách không được cộng hai lần. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Model chấm trực tiếp cặp query–candidate, giữ retrieval score riêng để đối chiếu. Pipeline rerank candidates rồi deduplicate parent trước khi chọn top-3, giúp câu hỏi multi-hop có chỗ cho nhiều nguồn bằng chứng. Latency đo thật được ghi trong `analysis/latency_breakdown.md`. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Faithfulness kiểm tra độ bám context; answer relevancy kiểm tra có trả lời đúng ý; context precision và recall giúp phân biệt nhiễu với thiếu bằng chứng. Báo cáo giữ từng câu và context để chẩn đoán. Smoke test API thật cho một câu đơn giản chạy thành công; điểm toàn bộ test set phải đọc từ báo cáo, không suy ra từ smoke test. |
| Contextual embeddings | M5 | `contextual_prepend()`, `_enrich_single_call()` | Combined mode dùng một call cho summary, questions, context và metadata. Index dùng cả context, summary và câu hỏi giả định; câu trả lời nhận parent gốc để tránh coi nội dung enrichment là bằng chứng độc lập. Metadata nguồn, phiên bản và parent ID từ ingestion không bị LLM ghi đè. |

Đo trên 26 tài liệu có text: basic tạo 57 chunks (trung bình 366 ký tự), semantic với threshold 0.85 tạo 208 chunks (trung bình 99 ký tự), structure-aware tạo 107 chunks; hierarchical tạo 26 parents và 101 children, child lớn nhất 256 ký tự. Semantic có chunk nhỏ nhất chỉ 6 ký tự, cho thấy threshold này làm nội dung bị phân mảnh; cần thử threshold thấp hơn và model phù hợp tiếng Việt. Số đo được lưu tại `reports/chunking_comparison.json`. Production lọc hai bản cũ, còn 24 tài liệu và 93 children.

Các thay đổi cùng được áp dụng trong production nên phép so sánh baseline–production là so sánh hai pipeline. Muốn xác định mức đóng góp của từng kỹ thuật cần ablation với cùng model, test set và prompt.

## Phần 2: Khó khăn và cách giải quyết

### Môi trường Python và model

Thông báo gặp trong lúc môi trường đang cài dependencies:

```text
ModuleNotFoundError: No module named 'torch'
```

Nguyên nhân là kiểm tra import trước khi pip cài xong toàn bộ môi trường. Cách debug là kiểm tra đúng executable `.venv/Scripts/python.exe`, đợi cài đặt hoàn tất, xác nhận import sentence-transformers/RAGAS và chạy `pip check`. Lưu `requirements-lock.txt` theo môi trường đã kiểm tra để có thể tái lập.

Model BGE có dung lượng lớn, tải lần đầu trở thành bottleneck setup. Script tải snapshot ghim theo commit, tải theo HTTP byte ranges và kiểm tra kích thước trước khi ghép file. Cache model không đưa lên Git; thời gian tải không được coi là latency truy vấn. Cache enrichment giúp tránh gọi lại API cho cùng nội dung, nguồn và model.

### PDF scan và giới hạn ingestion

Thông báo thực tế:

```text
Skipped BCTC.pdf: no text layer; OCR required.
Skipped Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf: no text layer; OCR required.
```

Kiểm tra `extract_text()` cho thấy không có lớp văn bản. Đây là thiếu dữ liệu đầu vào; sửa search không thể phục hồi nội dung chưa được index. Pipeline cảnh báo và bỏ qua thay vì tạo chunk rỗng. Hướng bổ sung là OCR, kiểm tra chất lượng text/bảng và lưu page number để trích dẫn.

### Phiên bản và mất ngữ cảnh

Corpus chứa cả nghỉ phép v2023/v2024 và mật khẩu v1/v2. Nếu retrieve bản cũ hoặc chỉ một child chứa con số mà thiếu điều kiện, LLM có thể trả lời sai dù câu trả lời có vẻ hợp lý. Cách xử lý trong lab là chọn newest version theo family, dùng ID parent có namespace theo tài liệu, khôi phục parent và loại kết quả trùng. Trong project thật cần catalog trạng thái hiệu lực thay vì phụ thuộc tên file.

Kiến thức cần bổ sung là kiểm thử chất lượng retrieval độc lập với generation, đánh giá câu hỏi multi-hop và thiết kế schema metadata theo miền dữ liệu. Tôi sẽ dùng evidence và test cases để quyết định thay đổi, không chỉ nhìn aggregate score.

## Phần 3: Action plan cho project cá nhân

### Project: RAG tìm dataset ADAS phục vụ huấn luyện từ scenes video và LiDAR

Theo mô tả project hiện tại, đã có RAG hỗ trợ tìm dataset để train ADAS. Chưa có thông tin chi tiết về loại index, model embedding và schema đang dùng; các bước dưới đây là kế hoạch nâng cấp, không giả định những thành phần đó đã triển khai.

Mục tiêu là giúp người dùng tìm dataset phù hợp yêu cầu huấn luyện, có đường dẫn nguồn và bằng chứng về cảm biến, scene, nhãn và điều kiện sử dụng. RAG tra cứu mô tả/catalog của dữ liệu; video và point cloud cần pipeline xử lý riêng, không được biến thành text chỉ bằng chunking.

### Kế hoạch áp dụng

1. **Chunking:** dùng structure-aware cho dataset card, paper và tài liệu schema; hierarchical cho tài liệu dài. Parent tương ứng dataset hoặc section, child mô tả scene/sensor/annotation. Giữ nguyên bảng thống kê, split và sensor specifications; luôn lưu URL, section và dataset version.
2. **Search:** dùng BM25 + dense + RRF. BM25 giữ chính xác tên dataset, cảm biến và loại nhãn; dense hỗ trợ cách mô tả scene bằng ngôn ngữ tự nhiên. Bổ sung filter cho `sensor_modalities`, `scene_tags`, `weather`, `time_of_day`, `region`, `annotation_tasks`, `license`, `split`, `calibration_available` và `synchronization_available`. Field chưa xác minh phải được đánh dấu unknown.
3. **Reranking:** áp dụng CrossEncoder cho top-20 metadata passages rồi chọn evidence từ các dataset khác nhau. Đo latency trên phần cứng triển khai; so sánh chất lượng với reranker nhẹ trước khi quyết định model. Không chọn một dataset chỉ vì tương đồng ngữ nghĩa khi nó thiếu modality hoặc annotation bắt buộc.
4. **Evaluation:** tạo bộ 30–50 câu hỏi theo nhu cầu thật: tìm scene ban đêm/mưa, cần video + LiDAR, nhãn 3D detection và đồng bộ cảm biến; so sánh các dataset; phát hiện thông tin không có. Gắn dataset phù hợp và đoạn evidence làm ground truth. Đo Recall@10/MRR cho retrieval, độ khớp điều kiện metadata, RAGAS và tính đúng của citation. Chia tập phát triển và kiểm tra để tránh tối ưu theo đúng câu đã nhìn thấy.
5. **Enrichment:** combined summary + HyQA cho dataset card để nối vocabulary gap giữa yêu cầu huấn luyện và mô tả dataset. Tách metadata do nguồn công bố với metadata suy luận; kiểm chứng giấy phép, split, calibration và synchronization bằng evidence trước khi đưa vào filter. Cache theo nội dung và phiên bản để giảm chi phí cập nhật.

### Timeline và tiêu chí hoàn thành

| Thời gian | Công việc | Kết quả cần bàn giao |
|---|---|---|
| Tuần 1 | Khảo sát RAG hiện tại; chuẩn hóa catalog/schema; lưu version và nguồn; thu thập test queries | Catalog có evidence cho từng field; bộ câu hỏi có nhãn; số liệu baseline retrieval và latency |
| Tuần 2 | Implement structure/hierarchical chunking, hybrid search và metadata filtering | So sánh Recall@10/MRR với baseline; truy vấn nhiều điều kiện trả kết quả có evidence |
| Tuần 3 | Reranking, combined enrichment và citations; chạy ablation | Báo cáo quality/cost/latency; phân tích bottom-5 trên tập kiểm tra; chọn cấu hình dựa trên số đo |
| Tuần 4 | Cập nhật catalog định kỳ, kiểm tra link và version, cảnh báo metadata unknown | Quy trình refresh và regression tests; log nguồn, filter và evidence của từng truy vấn |

Ưu tiên trước mắt là tính đúng của metadata và bằng chứng cảm biến/nhãn. Việc đề xuất dataset cho train sẽ chỉ có ý nghĩa khi người dùng xác nhận được dataset đáp ứng các điều kiện của mô hình ADAS cần huấn luyện.
