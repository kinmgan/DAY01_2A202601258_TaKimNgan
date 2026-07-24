# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Qua bốn phản hồi, tôi nhận thấy nhiệt độ (temperature) càng cao thì câu trả lời càng đa dạng và sáng tạo, tính ngẫu nhiên càng lớn (ở mức 0.0 rất khuôn mẫu, đến 0.7 thì văn phong tự nhiên hơn). Phản hồi bắt đầu kém mạch lạc và có hiện tượng "ảo giác" (như tự đặt tên lạ "cà phê trứng Nguyễn") ở mức temperature 1.8.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> - **Trợ lý soạn hợp đồng pháp lý**: Nên đặt temperature = 0.0, vì văn bản pháp lý đòi hỏi tính chính xác tuyệt đối, chuẩn mực và không được phép tự ý bịa đặt hay sáng tạo nội dung.
> - **Trợ lý viết slogan quảng cáo**: Nên đặt temperature = 0.7 đến 1.0 (hoặc cao hơn), vì slogan cần sự phá cách, độc đáo, sáng tạo và ngôn từ bất ngờ để thu hút người xem.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> - **Ước tính chi phí (20.000 user * 2 lần * 500 token = 20.000.000 token output/ngày)**: Model lớn (gpt-4o) tốn $200/ngày (20.000 * $0.010), đắt gấp hơn 16 lần so với model nhỏ (gpt-4o-mini) chỉ tốn $12/ngày (20.000 * $0.0006).
> - **Model lớn xứng đáng khi**: Cần giải quyết các bài toán phức tạp, suy luận logic sâu, viết code hoặc phân tích dữ liệu chuyên sâu.
> - **Model nhỏ là lựa chọn đúng khi**: Thực hiện các tác vụ đơn giản, lặp đi lặp lại với khối lượng lớn như tóm tắt bài viết, phân loại văn bản, hay chatbot hỏi đáp cơ bản.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Hai phản hồi khác biệt hoàn toàn: nhà thơ dùng từ ngữ giàu hình ảnh, ví von ("khu vườn kỳ diệu", "hạt giống"), không có thuật ngữ kỹ thuật và đậm chất văn chương nghệ thuật; trong khi đó kỹ sư phần mềm dùng giọng văn trực diện, chính xác với đầy đủ thuật ngữ ("thuật toán", "dữ liệu", "mô hình") và đi kèm luôn một ví dụ code Python thực tế để minh họa. Từ đó rút ra, system prompt không chỉ định hình được "nhân vật" (giọng điệu, phong cách hành văn) mà còn quyết định được độ chuyên sâu (mức kỹ thuật) và cấu trúc trình bày của văn bản đầu ra.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Thực tế thử nghiệm cho thấy số token ước lượng (181.33) cao hơn khoảng 8.46% so với số token thực tế đo được bằng tiktoken (166). Nếu dùng ước lượng thô `số từ / 0.75` cho ứng dụng tiếng Việt trong trường hợp này, ta sẽ **dự toán thừa**. Nguyên nhân là do tokenizer `o200k_base` mới của OpenAI (dành cho GPT-4o) tối ưu hóa và nén rất tốt các ngôn ngữ phi tiếng Anh như Tiếng Việt, giúp lượng token tiêu tốn thực tế ít hơn đáng kể so với việc dùng hệ số ước lượng vốn chỉ đúng với các tokenizer cũ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Ứng dụng hưởng lợi nhiều nhất là (a) chatbot văn bản và (b) trợ lý giọng nói, vì người dùng tương tác theo thời gian thực luôn mong đợi phản hồi nhanh; việc nhận dữ liệu streaming giúp giảm thiểu độ trễ ban đầu (Time To First Token), mang lại trải nghiệm tự nhiên và mượt mà hơn. Ngược lại, (c) pipeline dịch tài liệu chạy ngầm ban đêm không cần streaming, do quá trình này xử lý hàng loạt ở chế độ nền, không có người dùng chờ đợi để tương tác, hệ thống chỉ cần kết quả cuối cùng hoàn chỉnh để lưu lại hoặc thực hiện bước tiếp theo thay vì phải xử lý từng mảnh dữ liệu nhỏ một cách không cần thiết.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Khi hệ thống API quá tải, nếu dùng delay cố định, hàng nghìn client sẽ cùng lúc gọi lại API ở những khoảng thời gian giống hệt nhau, tạo ra hiện tượng "thundering herd" (cơn sốt bầy đàn) khiến máy chủ tiếp tục sập. Exponential backoff (đợi theo cấp số nhân) giúp các client giãn cách thời gian retry ngày càng dài ra, làm giảm nhanh chóng áp lực tổng thể lên server và tạo cơ hội cho máy chủ phục hồi. Tuy nhiên, nếu nhiều client cùng bắt đầu bị lỗi ở một thời điểm, các bước lùi cấp số nhân của chúng vẫn có thể trùng khớp nhau; việc thêm "jitter" (độ trễ ngẫu nhiên vào thời gian chờ) sẽ xé lẻ và phân tán hoàn toàn các thời điểm retry này, giúp dàn đều lưu lượng mạng và ngăn chặn các đỉnh tải phụ phát sinh.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> **System prompt**: "Bạn là một chuyên gia dinh dưỡng. Bạn luôn trả lời ngắn gọn dưới 3 câu. Nếu người dùng hỏi các chủ đề không liên quan đến ẩm thực hay sức khỏe, bạn phải từ chối trả lời."
> 
> **Hai chỗ quan trọng**:
> 1. *"Bạn luôn trả lời ngắn gọn dưới 3 câu"*: Nếu xóa câu này, hành vi trợ lý sẽ thay đổi rõ rệt ở độ dài câu trả lời. Trợ lý có thể sẽ giải thích dài dòng, lan man thay vì đi thẳng vào vấn đề cốt lõi.
> 2. *"Nếu người dùng hỏi các chủ đề không liên quan... thì từ chối"*: Nếu xóa câu này, trợ lý sẽ mất đi tính chuyên môn nghiệp vụ. Khi người dùng hỏi về toán học hay code, AI vẫn sẽ trả lời bình thường thay vì từ chối như thiết kế ban đầu.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> **Tình huống hội thoại cụ thể**:
> - Lượt 1: Người dùng nói "Tên tôi là Hoàng và tôi bị dị ứng đậu phộng."
> - Lượt 2, 3, 4, 5: Người dùng hỏi về các công thức nấu bún bò, phở, đồ ăn sáng...
> - Lượt 6: Người dùng hỏi "Tôi có thể ăn bánh quy bơ đậu phộng không?"
> - **Kết quả**: Vì trợ lý chỉ lưu giữ 4 lượt chat gần nhất (từ lượt 2 đến 5), nó đã quên mất thông tin "dị ứng đậu phộng" ở lượt 1, dẫn đến việc khuyên người dùng có thể ăn bình thường, điều này rất nguy hiểm và sai ngữ cảnh.
> 
> **Đề xuất cách khắc phục**: 
> Sử dụng cơ chế tóm tắt (Summarization). Thay vì cắt bỏ hoàn toàn các lượt chat cũ, trước khi xóa, hệ thống gọi một API ngầm để tóm tắt các thông tin quan trọng nhất từ các lượt đó (ví dụ: "Người dùng tên Hoàng, bị dị ứng đậu phộng"). Đoạn tóm tắt này sau đó được giữ lại và ghép vào đầu lịch sử hội thoại tiếp theo, giúp AI duy trì được bối cảnh đường dài.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
