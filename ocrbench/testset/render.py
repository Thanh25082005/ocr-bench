"""Chụp ảnh trang HTML bằng Chromium chạy ẩn (Playwright).

Trình duyệt nối chữ Ả Rập và xử lý hướng phải-sang-trái chuẩn xác, điều mà vẽ chữ trực tiếp
bằng PIL thường làm sai.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


class Renderer:
    def __init__(self, scale: float = 1.5, width: int = 1000):
        self.scale = scale
        self.width = width

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        errors = []
        # Thử Chromium của Playwright trước, sau đó tới Google Chrome đã cài trên máy
        for kwargs in ({}, {"channel": "chrome"}, {"channel": "chromium"}):
            try:
                self._browser = self._pw.chromium.launch(**kwargs)
                break
            except Exception as e:
                errors.append(f"{kwargs or 'mặc định'}: {str(e).splitlines()[0]}")
        else:
            self._pw.stop()
            raise RuntimeError("Không mở được Chromium. Chạy `playwright install chromium`.\n" + "\n".join(errors))
        self._page = self._browser.new_page(
            viewport={"width": self.width, "height": 800}, device_scale_factor=self.scale
        )
        self._tmp = tempfile.TemporaryDirectory()
        return self

    def render(self, html: str, out: Path) -> None:
        # Ghi ra file để trang được nạp qua file://, nhờ vậy @font-face đọc được font cục bộ
        src = Path(self._tmp.name) / "page.html"
        src.write_text(html, encoding="utf-8")
        self._page.goto(src.as_uri())
        self._page.evaluate("document.fonts.ready")
        # full_page: không cắt mất phần nào, nếu cắt thì đáp án sẽ chứa chữ không có trong ảnh
        self._page.screenshot(path=str(out), full_page=True)

    def __exit__(self, *exc):
        self._browser.close()
        self._pw.stop()
        self._tmp.cleanup()
