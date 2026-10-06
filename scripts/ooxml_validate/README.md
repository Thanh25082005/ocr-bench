# Kiểm tra DOCX theo chuẩn của Microsoft (Open XML SDK)

LibreOffice bỏ qua nhiều lỗi mà **Word không chấp nhận** (vd. sai thứ tự phần tử XML → Word báo file hỏng).
Công cụ này dùng `OpenXmlValidator` của Microsoft (cùng lược đồ Word dùng), kiểm theo Office 2010 và Office 2019.

```bash
curl -fsSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh
bash /tmp/dotnet-install.sh --channel 8.0 --install-dir ~/.dotnet-sdk
export DOTNET_ROOT=~/.dotnet-sdk DOTNET_CLI_TELEMETRY_OPTOUT=1
~/.dotnet-sdk/dotnet run --project scripts/ooxml_validate -c Release -- file1.docx file2.docx
```

Kết quả mong đợi: `lỗi: 0` cho mọi file.
