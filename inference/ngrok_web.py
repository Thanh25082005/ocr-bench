"""Mở web demo (cổng 7860) ra ngoài bằng ngrok, in link https cho người khác vào.

    python inference/ngrok_web.py              # tạo (hoặc lấy lại) tunnel HTTP tới 127.0.0.1:7860, in link
    python inference/ngrok_web.py --port 7861
    python inference/ngrok_web.py --stop       # đóng tunnel web (KHÔNG đụng tunnel SSH)

Cách làm: Kaggle đã có một tiến trình ngrok chạy tunnel SSH; tài khoản ngrok miễn phí chỉ cho 1 tiến trình
(ERR_NGROK_108) → THÊM tunnel HTTP vào chính tiến trình đó qua API nội bộ http://127.0.0.1:4040/api/tunnels.
Không có tiến trình nào → chạy `ngrok http <cổng>` mới (cần ngrok đã có authtoken).

Web PHẢI đang chạy với đăng nhập (AUTH=ten:matkhau bash inference/start.sh): script kiểm tra và từ chối nếu
trang mở được mà không cần đăng nhập.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

NAME = "ocr-web"


def _api(base: str, path: str = "/api/tunnels", method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def find_agent() -> str | None:
    for port in range(4040, 4046):
        base = f"http://127.0.0.1:{port}"
        try:
            _api(base)
            return base
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return None


def web_requires_login(port: int) -> bool | None:
    """Gradio có auth: /config trả 401 hoặc trang có form đăng nhập."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/config", timeout=10):
            return False  # đọc được cấu hình mà không đăng nhập
    except urllib.error.HTTPError as e:
        return e.code in (401, 403)
    except (urllib.error.URLError, OSError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--allow-no-login", action="store_true", help="(không khuyến nghị) cho mở dù web không có mật khẩu")
    a = ap.parse_args()

    agent = find_agent()
    if a.stop:
        if agent:
            try:
                _api(agent, f"/api/tunnels/{NAME}", method="DELETE")
                print(f"✔ đã đóng tunnel '{NAME}'")
            except urllib.error.HTTPError as e:
                print(f"không có tunnel '{NAME}' ({e.code})")
        return 0

    login = web_requires_login(a.port)
    if login is None:
        print(f"✘ Web chưa chạy ở 127.0.0.1:{a.port}. Mở web trước: AUTH=ten:matkhau bash inference/start.sh")
        return 1
    if not login and not a.allow_no_login:
        print("✘ Web đang mở KHÔNG cần đăng nhập — đưa ra ngrok thì ai có link cũng xem được tài liệu và lịch sử.\n"
              "  Chạy lại web với mật khẩu: AUTH=ten:matkhau bash inference/start.sh  rồi chạy lại script này.")
        return 1

    if not agent:
        exe = shutil.which("ngrok")
        if not exe:
            print("✘ Không thấy tiến trình ngrok nào (API cổng 4040–4045) và không có lệnh `ngrok`.")
            return 1
        print("Không thấy tiến trình ngrok đang chạy → chạy `ngrok http` mới...")
        subprocess.Popen([exe, "http", str(a.port), "--log", "stdout"], stdout=open("/tmp/ngrok_web.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(20):
            time.sleep(1)
            agent = find_agent()
            if agent:
                break
        if not agent:
            print("✘ ngrok không khởi động được. Xem /tmp/ngrok_web.log (thường do thiếu authtoken hoặc ERR_NGROK_108).")
            return 1

    tunnels = _api(agent).get("tunnels", [])
    print("Tunnel đang có:", ", ".join(f"{t['name']} → {t['public_url']} ({t['config']['addr']})" for t in tunnels)
          or "(không có)")
    for t in tunnels:
        addr = t.get("config", {}).get("addr", "")
        if t.get("public_url", "").startswith("https") and addr.rstrip("/").endswith(f":{a.port}"):
            print(f"\n✔ LINK WEB (đã có sẵn): {t['public_url']}")
            return 0
    try:
        t = _api(agent, method="POST", body={"name": NAME, "proto": "http", "addr": str(a.port)})
    except urllib.error.HTTPError as e:
        print(f"✘ ngrok từ chối tạo tunnel: {e.code} {e.read().decode('utf-8', 'ignore')[:400]}")
        return 1
    print(f"\n✔ LINK WEB: {t['public_url']}")
    print("  Gửi link + tài khoản cho người xem. Lần đầu ngrok hiện trang cảnh báo → bấm 'Visit Site'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
