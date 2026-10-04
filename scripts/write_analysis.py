"""Build failure and latency analysis directly from reproducible report data."""

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def main():
    production = json.loads(
        (ROOT / "reports/ragas_report.json").read_text(encoding="utf-8")
    )
    baseline = json.loads(
        (ROOT / "reports/naive_baseline_report.json").read_text(encoding="utf-8")
    )
    lines = [
        "# Failure Analysis — Lab 18: Production RAG",
        "",
        "**Học viên:** Đàm Việt Hưng — **MSSV:** 2A202602600",
        "",
        "## Kết quả RAGAS",
        "",
        "Dữ liệu lấy trực tiếp từ hai báo cáo JSON; ground truth chỉ dùng trong đánh giá, không đưa vào retrieval hoặc sinh câu trả lời.",
        "",
        "| Metric | Naive Baseline | Production | Δ |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        old, new = baseline["aggregate"][metric], production["aggregate"][metric]
        values = (
            f"{old:.4f} | {new:.4f} | {new - old:+.4f}"
            if old is not None and new is not None
            else "N/A | N/A | N/A"
        )
        lines.append(f"| {metric} | {values} |")
    lines += [
        "",
        f"Trạng thái production: `{production['evaluation_status']}`; số câu hỏi: {production['num_questions']}.",
        "",
        "Baseline dùng basic chunking và dense-only trên toàn bộ corpus. Production dùng tài liệu hiện hành, hierarchical child retrieval, enrichment, hybrid RRF, reranking rồi trả về parent. Vì nhiều bước cùng thay đổi, chênh lệch trên không chứng minh tác động riêng của từng module.",
        "",
        "## Bottom-5 Failures",
        "",
    ]
    if production["evaluation_status"] != "completed":
        lines += [
            "RAGAS chưa hoàn tất nên chưa có bottom-5 được đo. Không diễn giải các điểm thiếu thành chất lượng bằng 0.",
            f"Lý do: {production.get('evaluation_error')}",
            "",
        ]
    for i, failure in enumerate(production["failures"], 1):
        metric = failure["worst_metric"]
        branch = (
            "Context thiếu → kiểm tra candidate/parent và truy vấn con → sửa retrieval."
            if metric == "context_recall"
            else (
                "Context chứa nhiễu → kiểm tra rerank và nguồn/phiên bản → sửa retrieval."
                if metric == "context_precision"
                else "Context có bằng chứng → kiểm tra câu trả lời, phủ định và phép tính → sửa generation."
            )
        )
        lines += [
            f"### #{i}",
            "",
            f"- **Question:** {failure['question']}",
            f"- **Expected:** {failure['ground_truth']}",
            f"- **Got:** {failure['answer']}",
            f"- **Worst metric:** `{metric}` = {failure['score']:.4f}; trung bình = {failure['average_score']:.4f}.",
            f"- **Error Tree:** Output chưa đạt → {branch}",
            f"- **Root cause (giả thuyết từ metric, cần đối chiếu evidence):** {failure['diagnosis']}.",
            f"- **Suggested fix:** {failure['suggested_fix']}.",
            "",
            "**Evidence đã retrieve:**",
            "",
        ]
        for j, context in enumerate(failure["contexts"], 1):
            lines += [
                f"<details><summary>Context {j}</summary>",
                "",
                context,
                "",
                "</details>",
                "",
            ]
    lines += [
        "## Case study: kiểm tra phiên bản và multi-hop",
        "",
        "Câu hỏi Senior có 9 năm thâm niên cần cả chính sách phép năm hiện hành và bảng lương. Error Tree: output có đủ số ngày phép và khoảng lương? → context có cả hai nguồn? → child của một tài liệu có chiếm hết top-3? → parent đã khôi phục đủ điều kiện? → LLM đã tính 15 + 9/3 = 18?",
        "",
        "Pipeline khắc phục việc thiếu ngữ cảnh bằng parent expansion và deduplicate parent trước khi chọn top-3. Loại bản cũ theo family `_vN` giúp tránh trả 12 ngày/5 năm từ chính sách cũ; cách này chỉ là quy tắc phù hợp corpus lab. Khi triển khai thật cần catalog trạng thái superseded và khoảng hiệu lực, đồng thời hỗ trợ truy vấn lịch sử.",
        "",
        "## Nếu có thêm một giờ",
        "",
        "1. Chạy ablation lần lượt: hybrid → rerank → parent expansion → version filter; giữ chung model và test set.",
        "2. Tách các câu hỏi multi-hop thành truy vấn con và gom bằng chứng trước khi trả lời.",
        "3. Thêm truy vấn số học/phủ định và kiểm tra thủ công câu trả lời có trích dẫn đúng nguồn.",
        "",
        "## Giới hạn dữ liệu",
        "",
        "PDF có text layer được nạp; PDF scan không có text được cảnh báo và bỏ qua, cần OCR trước indexing. Corpus lab chưa kiểm thử trực tiếp video/LiDAR. Các điểm RAGAS được dùng để chẩn đoán; metric thấp chưa đủ xác nhận duy nhất một nguyên nhân.",
        "Câu tạm ứng 15 triệu sau 20 ngày có ground truth khoảng 50.000 VNĐ cho 5 ngày quá hạn. Tài liệu chỉ nêu 2%/tháng, chưa quy định pro-rata hoặc số ngày quy đổi trong tháng. Vì vậy 50.000 VNĐ cần giả định tháng 30 ngày; nên bổ sung quy tắc vào corpus hoặc đánh dấu giả định trong ground truth. Không chỉnh prompt để buộc model khẳng định dữ kiện chưa có bằng chứng.",
        "",
    ]
    (ROOT / "analysis/failure_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    latency = [
        "# Latency breakdown",
        "",
        "Các số liệu đo bằng `time.perf_counter()`; bao gồm cold model load khi build, không coi đó là latency của truy vấn nóng.",
        "",
        "| Build stage | Production (ms) |",
        "|---|---:|",
    ]
    for stage, value in production["latency"]["build"].items():
        latency.append(f"| {stage} | {value:.2f} |")
    latency += [
        "",
        "| Query stage | Baseline mean (ms) | Production mean (ms) |",
        "|---|---:|---:|",
    ]
    for stage in ("retrieval_ms", "reranking_ms", "generation_ms"):
        averages = []
        for report in (baseline, production):
            values = [
                row[stage] for row in report["latency"]["queries"] if stage in row
            ]
            averages.append(f"{mean(values):.2f}" if values else "N/A")
        latency.append(f"| {stage} | {' | '.join(averages)} |")
    latency += [
        "",
        "Runtime production:",
        "",
        "```json",
        json.dumps(production["runtime"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (ROOT / "analysis/latency_breakdown.md").write_text(
        "\n".join(latency), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
