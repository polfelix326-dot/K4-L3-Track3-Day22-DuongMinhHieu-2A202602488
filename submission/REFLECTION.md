# Bài phản tư — Lab 22: căn chỉnh mô hình bằng DPO/ORPO

**Họ tên:** Dương Minh Hiếu
**Mã sinh viên:** 2A202602488
**Khóa:** K4
**Tier:** T4 (Google Colab)
**Ngày thực hiện:** 2026-10-08

> Báo cáo này được điền từ notebook Colab và các file kết quả đã tải xuống. Những số liệu không có trong output được ghi là chưa đo/chưa chạy, không ước lượng.

## 1. Cấu hình và dữ liệu

| Mục | Kết quả |
|---|---|
| GPU | Tesla T4; log Unsloth báo tổng bộ nhớ GPU 14.56 GiB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`; 1.000 mẫu, 1 epoch; loss cuối 1.3603 |
| Dữ liệu preference | `sailor2/sea-ultrafeedback-onpolicy`, tiếng Việt; 800 train / 100 held-out |
| Độ dài preference | `chosen` dài hơn trong 65.875% cặp; median 94 token so với 86 token của `rejected` |
| DPO | β = 0.1; learning rate = 5×10⁻⁶; 1 epoch; sigmoid |
| Reference | `models/sft-merged` |
| Giám khảo NB4 | Chưa hoàn tất: nạp reward model gặp CUDA out-of-memory; không có `judge_summary.json` |
| Thời gian huấn luyện / chi phí | Không được ghi lại trong output đã tải xuống |

NB0 đã chạy các phần kiểm tra loss tham chiếu: tại policy = reference, loss bằng 0.6931, xấp xỉ log(2). Tuy nhiên, output cho thấy hàm `my_dpo_loss` vẫn trả về `None` và in “Chưa cài”; vì vậy phần tự cài loss NB0 chưa hoàn thành.

## 2. Kết quả huấn luyện DPO

| Chỉ số | Kết quả |
|---|---:|
| Loss DPO đầu tiên được log | 0.69319 |
| Loss train cuối | 0.67410 |
| Reward gap train cuối | 0.09829 |
| Reward chosen / rejected trên held-out | 0.42696 / 0.34095 |
| Margin held-out | 0.08601 |
| Reward accuracy held-out | 0.67 |
| Chẩn đoán | `INTENDED` |
| Độ dài trả lời trung bình SFT → DPO | 590 → 613 ký tự trên 58 prompt được sinh |
| Đỉnh VRAM sử dụng | Chưa được ghi nhận; 14.56 GiB là tổng bộ nhớ GPU theo log, không phải mức sử dụng đỉnh |

## 3. Đọc đường reward

Ở bước đầu loss được log là 0.69319, gần log(2) = 0.6931. Điều này phù hợp với thiết lập khởi tạo trong đó policy bắt đầu từ mô hình SFT và adapter DPO mới chưa tạo ra thay đổi đáng kể. Ở cuối train, reward của `chosen` là +0.4183 và `rejected` là +0.3200, tạo reward gap +0.0983. Trên held-out, `chosen` đạt +0.4270 và `rejected` +0.3410, margin +0.0860, reward accuracy 0.67. Như vậy, ở thời điểm cuối, cả hai reward đều dương so với reference, nhưng chosen cao hơn rejected; đây không phải dấu hiệu likelihood displacement theo tiêu chí chẩn đoán của lab. Margin held-out cũng cùng dấu và khá gần margin train, nên số liệu cuối không cho thấy khoảng cách chỉ tăng trên train. Tuy nhiên, accuracy held-out 0.67 còn cách xa mức hoàn hảo, và một epoch trên một split chưa đủ để kết luận tổng quát hóa tốt. Kết luận này mô tả reward ngầm của DPO, không chứng minh độc lập rằng người dùng sẽ thích câu trả lời hơn. Biểu đồ train/held-out đã được lưu, nhưng cần đọc cùng với đánh giá NB4 sau khi phần chấm được chạy thành công.

## 4. So sánh SFT với SFT+DPO

NB4 đã sinh 8 câu hỏi cố định và 50 prompt held-out (58 prompt tổng cộng). Trung bình câu trả lời là 590 ký tự với SFT và 613 ký tự với DPO. Tuy nhiên, bước chấm reward model bị dừng do CUDA out-of-memory: lỗi báo muốn cấp phát thêm 7.43 GiB khi chỉ còn 7.40 GiB khả dụng. Vì thế chưa có phán quyết của giám khảo, khoảng tin cậy, sanity accuracy, tỉ lệ thắng câu dài hơn hay win rate theo nhóm. Không thể kết luận DPO thắng SFT từ các số liệu hiện có.

| Nhóm | Số câu đã sinh | DPO thắng | SFT thắng | Hòa | Win rate / CI 95% | Cặp dài gần bằng | Câu dài hơn thắng |
|---|---:|---|---|---|---|---|---|
| Held-out | 50 | Chưa chấm | Chưa chấm | Chưa chấm | Chưa có | Chưa có | Chưa có |
| Hữu ích | 4 | Chưa chấm | Chưa chấm | Chưa chấm | Chưa có | Chưa có | Chưa có |
| An toàn | 4 | Chưa chấm | Chưa chấm | Chưa chấm | Chưa có | Chưa có | Chưa có |

Quan sát định tính không thay thế được đánh giá judge. Ở câu quicksort (h1), SFT và DPO trả lời gần như cùng nội dung, nên riêng ví dụ này chưa thể hiện cải thiện rõ. Ở câu yêu cầu công thức chất nổ (s1), cả hai đều từ chối cung cấp hướng dẫn; đây là hành vi an toàn tốt nhưng không chứng minh DPO tốt hơn SFT vì hai câu trả lời gần như trùng nhau. Ở câu hỏi gạo và trứng (h2), cả hai câu trả lời đều không bám tốt vào nguyên liệu người dùng nêu, và đầu ra DPO vẫn có nội dung không liên quan về thịt xông khói. Ngoài ra, một số câu trả lời được lưu có các chuỗi `<tool_call>` hoặc `</tool_call>` ở đầu; cần xem đây là lỗi định dạng đầu ra khi đánh giá mẫu. Chưa thể xác định mức độ phổ biến của lỗi này trên toàn bộ tập chỉ từ vài ví dụ.

Để hoàn tất mục này, cần chạy lại riêng bước chấm NB4 sau khi giải phóng/khởi động lại runtime Colab hoặc dùng cấu hình giám khảo phù hợp bộ nhớ, rồi tải `data/eval/judge_summary.json` về. Cho đến lúc có file đó, độ tin cậy của judge và các chỉ số thắng/thua vẫn chưa xác định.

## 5. Đánh đổi theo β (bonus)

Không chạy beta-sweep; chưa có số liệu thực nghiệm cho β = 0.05 và 0.5. Dự đoán trước khi chạy: β nhỏ có thể cho policy dịch chuyển xa reference hơn và làm reward margin thay đổi mạnh hơn. β lớn thường ràng buộc policy gần reference hơn, nhưng raw margin cũng phụ thuộc trực tiếp vào β. Vì vậy, khi so sánh cần xem reward accuracy held-out và chất lượng đầu ra, không kết luận chỉ từ độ lớn margin.

## 6. Quyết định quan trọng nhất

Quyết định được phân tích là dùng tier T4 và giữ cấu hình DPO mặc định β = 0.1, learning rate 5×10⁻⁶, một epoch, với reference là mô hình SFT đã gộp. Phương án thay thế là chạy BigGPU với mô hình lớn hơn hoặc huấn luyện nhiều epoch/đổi β; các phương án đó chưa được thử trong lần chạy này. T4 cho phép hoàn thành SFT, chuẩn bị dữ liệu, huấn luyện DPO và sinh đủ 50 câu held-out trong NB4. Kết quả huấn luyện cho thấy loss giảm từ khoảng 0.693 ở bước đầu xuống loss train cuối 0.6741; reward gap là dương trên train (+0.0983) và held-out (+0.0860), còn reward accuracy held-out là 0.67. Đây là tín hiệu phù hợp với mục tiêu tối ưu cặp preference, nhưng chưa đủ để nói chất lượng trả lời tổng thể đã tăng. Đặc biệt, bước judge bị OOM nên không có win rate hay sanity accuracy để xác nhận bằng một phép chấm độc lập; chiều dài trả lời trung bình DPO cũng cao hơn 23 ký tự, trong bối cảnh dữ liệu có `chosen` dài hơn ở 65.875% cặp. Nếu làm lại, tôi sẽ ưu tiên hoàn tất đánh giá judge trước khi đưa ra kết luận, lưu thời gian và mức dùng VRAM, kiểm tra các marker `<tool_call>` trong đầu ra, và so sánh ít nhất một cấu hình khác trên cùng held-out set. Tôi cũng cần tự hoàn thành hàm `my_dpo_loss` ở NB0 để phần giải thích công thức có bằng chứng thực thi đầy đủ.

## 7. Bộ đo chuẩn (bonus NB6)

NB6 chưa chạy; không có số liệu IFEval, GSM8K hoặc Global-MMLU-vi để so sánh SFT với SFT+DPO. Do đó chưa thể nhận xét chênh lệch có vượt hai lần stderr hay có alignment tax trên GSM8K hay không.

## 8. Biến thể loss (bonus NB3b)

Notebook có output tổng hợp cho bốn biến thể. Các mô hình được đo độ dài trên 20 prompt thử; margin bên dưới là hiệu số reward chosen − rejected, suy ra từ hai cột reward được in ra.

| Loss | Reward accuracy held-out | Margin held-out | Độ dài trung bình (ký tự) | Ghi chú |
|---|---:|---:|---:|---|
| DPO | 0.63 | 0.02622 | 377.15 | Chẩn đoán `INTENDED` |
| RPO | 0.62 | 0.03384 | 380.20 | Chẩn đoán `INTENDED` |
| DPO-norm | 0.63 | 0.00863 | 381.00 | Chẩn đoán `LIKELIHOOD DISPLACEMENT`; chosen/rejected reward đều âm |
| LD-DPO | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Không xuất hiện trong bảng output đã lưu |
| ORPO | 0.65 | Không áp dụng/không có reward margin | 370.75 | Không dùng reference; output có log-odds ratio -0.62336 |

Trong các biến thể có số đo, DPO-norm cho đầu ra trung bình dài nhất (381 ký tự), còn ORPO ngắn nhất (370.75 ký tự); chênh lệch nhỏ và chỉ dựa trên 20 prompt nên không nên khái quát quá mức. NB3b chưa hoàn chỉnh vì không có kết quả LD-DPO trong output đã tải xuống. Các reward accuracy này là chỉ số trainer trên held-out, không phải kết quả judge người dùng của NB4.

## 9. GRPO (bonus NB7)

NB7 chưa chạy; chưa có số liệu accuracy trước/sau, cỡ mẫu kiểm tra hoặc đường reward GRPO. Vì vậy chưa thể tính sai số chuẩn hay kết luận GRPO có cải thiện hay không.

## Danh sách bonus

- [x] NB3b — có kết quả DPO, RPO, DPO-norm và ORPO; LD-DPO chưa có output
- [ ] NB5 — GGUF
- [ ] NB6 — benchmark
- [ ] NB7 — GRPO
- [ ] β-sweep
- [ ] Chấm chéo bằng hai họ mô hình
- [ ] Đẩy lên HF Hub và tạo thẻ mô hình
- [ ] BONUS-CHALLENGE

## Điều bất ngờ nhất

Reward gap cuối trên held-out dương (+0.0860) và reward accuracy đạt 0.67, nhưng bước judge độc lập lại không hoàn tất vì thiếu bộ nhớ GPU. Điều này nhắc tôi rằng chỉ số huấn luyện tích cực chưa đủ để tuyên bố mô hình trả lời tốt hơn; cần hoàn tất đánh giá và kiểm tra trực tiếp chất lượng, định dạng câu trả lời.
