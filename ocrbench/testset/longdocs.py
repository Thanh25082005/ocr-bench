"""Tài liệu dài để đo khả năng giữ ngữ cảnh.

- Bảng dài (1–3 trang): sao kê tài khoản (có ô nợ/có để trống xen kẽ, dễ bị dồn cột)
  hoặc bảng kê hàng hóa 8 cột dày đặc số. Trang sau có thể in lại hàng tiêu đề hoặc không.
  Đáp án là MỘT bảng logic (tiêu đề một lần + mọi hàng).
- Văn bản dài (1–2 trang): văn xuôi thật từ Wikipedia, ~một nửa các con số bị thay ngẫu nhiên
  để phân biệt model "đọc thật" với model "nhớ lại" bài viết đã học.
"""

from __future__ import annotations

import random
import re

from . import content as C
from .templates import Block, base_css, esc, html_to_text, logo_svg, text_block, wrap_page

TABLE_BINS = [("1 trang · 30-45 dòng", 1, (30, 45)), ("2 trang · 60-90 dòng", 2, (60, 90)),
              ("3 trang · 100-140 dòng", 3, (100, 140))]
TEXT_BINS = [("1 trang · ~4.000 ký tự", 1, 4000), ("2 trang · ~8.500 ký tự", 2, 8500)]

AR_MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
MERCHANTS = ["Carrefour", "Amazon.sa", "Starbucks", "IKEA", "Noon.com", "Uber", "Apple Store", "Jarir Bookstore",
             "Lulu Hypermarket", "Careem"]
AR_MERCHANTS = ["هايبر بنده", "مكتبة جرير", "صيدلية النهدي", "مطعم البيك", "بقالة الحي", "محطة وقود الدريس"]
AR_UNITS = ["قطعة", "علبة", "كرتون", "متر", "كيلوغرام", "لتر"]
EN_UNITS = ["pcs", "box", "carton", "m", "kg", "L"]


# --- Bảng dài -----------------------------------------------------------------------


def _statement_rows(c: C.Content, n: int):
    rng = c.rng
    if c.ar:
        head = ["التاريخ", "رقم المرجع", "البيان", "مدين", "دائن", "الرصيد"]
        merchants = MERCHANTS if c.lang == "mixed" else AR_MERCHANTS
        debit_desc = [lambda: f"شراء عبر نقاط البيع - {c.pick(merchants)}", lambda: "سحب نقدي من الصراف الآلي",
                      lambda: "رسوم خدمة", lambda: "سداد فاتورة كهرباء", lambda: f"تحويل صادر إلى {c.company()}"]
        credit_desc = [lambda: f"تحويل وارد من {c.person()}", lambda: f"راتب شهر {c.pick(AR_MONTHS)}",
                       lambda: f"إيداع شيك رقم {c.num(rng.randint(100000, 999999))}"]
    else:
        head = ["Date", "Reference", "Description", "Debit", "Credit", "Balance"]
        debit_desc = [lambda: f"POS purchase - {c.pick(MERCHANTS)}", lambda: "ATM cash withdrawal",
                      lambda: "Service fee", lambda: "Electricity bill payment",
                      lambda: f"Outgoing transfer to {c.company()}"]
        credit_desc = [lambda: f"Incoming transfer from {c.person()}", lambda: f"Salary {c.pick(EN_MONTHS)}",
                       lambda: f"Cheque deposit No. {rng.randint(100000, 999999)}"]
    balance = rng.randint(5_000, 80_000) + 0.0
    year, month, day = rng.randint(2023, 2025), rng.randint(1, 9), 1
    rows = []
    for _ in range(n):
        day += rng.choice([0, 0, 1, 1, 2])
        if day > 28:
            day, month = 1, month + 1
        date = c.num(f"{year}/{month:02d}/{day:02d}" if c.ar else f"{day:02d}/{month:02d}/{year}")
        ref = f"TRX{rng.randint(10**7, 10**8 - 1)}"
        if rng.random() < 0.65:
            amt = round(rng.uniform(10, 3000), 2)
            balance -= amt
            rows.append([date, ref, c.pick(debit_desc)(), c.money(amt), "", c.money(balance)])
        else:
            amt = round(rng.uniform(500, 15000), 2)
            balance += amt
            rows.append([date, ref, c.pick(credit_desc)(), "", c.money(amt), c.money(balance)])
    title = "كشف حساب" if c.ar else "Account Statement"
    return title, head, rows, balance


def _inventory_rows(c: C.Content, n: int):
    rng = c.rng
    if c.ar:
        head = ["الرمز", "الصنف", "الوحدة", "الكمية", "سعر الوحدة", "الخصم %", "الضريبة", "الإجمالي"]
        units = AR_UNITS
    else:
        head = ["Code", "Item", "Unit", "Qty", "Unit Price", "Disc. %", "VAT", "Total"]
        units = EN_UNITS
    rows, grand = [], 0.0
    for _ in range(n):
        qty = rng.randint(1, 400)
        price = round(rng.uniform(1, 900), 2)
        disc = rng.choice([0, 0, 5, 10, 15])
        net = qty * price * (1 - disc / 100)
        vat = net * 0.15
        grand += net + vat
        rows.append([c.code(), c.product(), c.pick(units), c.num(qty), c.money(price), c.num(disc),
                     c.money(vat), c.money(net + vat)])
    title = "قائمة جرد المخزون" if c.ar else "Inventory List"
    return title, head, rows, grand


def _table_html(head, rows, with_head: bool, numeric_cols) -> tuple[str, str]:
    """(html hiển thị, html đáp án) cho một đoạn bảng."""
    th = "<tr>" + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr>" if with_head else ""
    body_page = "".join(
        "<tr>" + "".join(f'<td{" class=num" if j in numeric_cols else ""}>{esc(v)}</td>' for j, v in enumerate(r)) + "</tr>"
        for r in rows
    )
    body_gt = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in r) + "</tr>" for r in rows)
    return f"<table>{th}{body_page}</table>", f"{th}{body_gt}"


def long_table(lang: str, rng: random.Random, font_faces: str, bin_idx: int):
    label, n_pages, (lo, hi) = TABLE_BINS[bin_idx]
    c = C.Content(rng, lang)
    css, style = base_css(rng, lang)
    css += f"\n.page {{ font-size:{rng.randint(12, 14)}px; }} th, td {{ padding:3px 6px; }} td.num {{ text-align:end; }}"
    n_rows = rng.randint(lo, hi)
    kind = rng.choice(["statement", "inventory"])
    if kind == "statement":
        title, head, rows, final = _statement_rows(c, n_rows)
        numeric = {3, 4, 5}
    else:
        title, head, rows, final = _inventory_rows(c, n_rows)
        numeric = {3, 4, 5, 6, 7}
    header_repeat = n_pages > 1 and rng.random() < 0.5

    intro = [text_block("h1", c.company()), text_block("h2", title)]
    if c.ar:
        intro += [text_block("p", f"العميل: {c.company()}"),
                  text_block("p", f"رقم الحساب: SA{rng.randint(10, 99)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}"),
                  text_block("p", f"الفترة: من {c.date()} إلى {c.date()}")]
        outro = [text_block("p", f"{'الرصيد الختامي' if kind == 'statement' else 'الإجمالي العام'}: {c.money(final)} {c.currency()}"),
                 text_block("p", f"عدد البنود: {c.num(n_rows)}")]
    else:
        intro += [text_block("p", f"Customer: {c.company()}"),
                  text_block("p", f"Account No.: GB{rng.randint(10, 99)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}"),
                  text_block("p", f"Period: {c.date()} to {c.date()}")]
        outro = [text_block("p", f"{'Closing balance' if kind == 'statement' else 'Grand total'}: {c.money(final)} {c.currency()}"),
                 text_block("p", f"Number of items: {n_rows}")]
    if rng.random() < 0.6:
        intro.insert(0, logo_svg(rng))

    # chia hàng đều cho các trang
    chunks, start = [], 0
    for k in range(n_pages):
        end = start + (n_rows - start) // (n_pages - k)
        chunks.append(rows[start:end])
        start = end
    pages = []
    for k, chunk in enumerate(chunks):
        page_html, _ = _table_html(head, chunk, with_head=(k == 0 or header_repeat), numeric_cols=numeric)
        blocks = (intro if k == 0 else []) + [Block(page_html)] + (outro if k == n_pages - 1 else [])
        pages.append(wrap_page(blocks, css, lang, font_faces))
    _, table_gt = _table_html(head, rows, with_head=True, numeric_cols=numeric)
    gt = "\n".join(b.gt for b in intro if b.gt) + f"\n<table>{table_gt}</table>\n" + "\n".join(b.gt for b in outro)
    meta = {"kind": f"long_{kind}", "lang": lang, "length_bin": label, "n_rows": n_rows, "n_cols": len(head),
            "header_repeated": header_repeat, "arabic_digits": c.arabic_digits, **style}
    return pages, gt, "table_html", meta


# --- Văn bản dài từ Wikipedia ----------------------------------------------------------


_NUM = re.compile(r"[0-9٠-٩]+")
_AR_TO_EN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_EN_TO_AR = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def perturb_numbers(text: str, rng: random.Random, p: float = 0.5) -> tuple[str, int]:
    """Thay ngẫu nhiên các con số (giữ số chữ số, giữ kiểu chữ số). Model nào "nhớ" bài Wikipedia
    gốc thay vì đọc ảnh sẽ viết lại số cũ, và bị tính là sai."""
    changed = 0

    def repl(m):
        nonlocal changed
        s = m.group(0)
        if rng.random() >= p:
            return s
        arabic = s.translate(_AR_TO_EN) != s
        digits = s.translate(_AR_TO_EN)
        new = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(len(digits) - 1))
        new = new if len(digits) > 1 else str(rng.randint(0, 9))
        if new == digits:
            return s
        changed += 1
        return new.translate(_EN_TO_AR) if arabic else new

    return _NUM.sub(repl, text), changed


def wiki_articles(lang: str, n: int, min_chars: int = 12000):
    """n bài Wikipedia đủ dài, theo thứ tự cố định của dataset. Trả về (tiêu đề, [(loại, đoạn)])."""
    from datasets import load_dataset

    cfg = {"ar": "20231101.ar", "en": "20231101.en"}[lang]
    ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
    out = []
    for row in ds:
        if len(out) >= n:
            break
        text = row["text"]
        if len(text) < min_chars:
            continue
        parts = []
        for line in (ln.strip() for ln in text.split("\n")):
            if not line:
                continue
            is_heading = len(line) < 60 and not re.search(r"[.!?؟:،,;]$", line)
            parts.append(("h2" if is_heading else "p", line))
        paragraphs = [t for k, t in parts if k == "p"]
        # bỏ bài dạng danh sách (nhiều dòng ngắn)
        if len(paragraphs) < 8 or sum(len(t) for t in paragraphs) / len(paragraphs) < 150:
            continue
        out.append((row["title"], parts))
    return out


def long_text(lang: str, article, rng: random.Random, font_faces: str, bin_idx: int):
    label, n_pages, budget = TEXT_BINS[bin_idx]
    title, parts = article
    css, style = base_css(rng, lang)
    css += f"\n.page {{ font-size:{rng.randint(14, 16)}px; }} p {{ text-align:justify; margin:.35em 0; }}"
    # bỏ tiêu đề mục nằm cuối cùng (không có nội dung đi sau)
    chosen, total = [], 0
    for kind, t in parts:
        if total >= budget:
            break
        chosen.append((kind, t))
        total += len(t)
    while chosen and chosen[-1][0] == "h2":
        chosen.pop()
    changed = 0
    blocks = [text_block("h1", title)]
    for kind, t in chosen:
        t, k = perturb_numbers(t, rng)
        changed += k
        blocks.append(text_block(kind, t))
    # chia trang theo số ký tự, ngắt ở ranh giới khối
    pages_blocks, cur, acc = [], [], 0
    per_page = total / n_pages
    for b in blocks:
        cur.append(b)
        acc += len(b.gt or "")
        if len(pages_blocks) < n_pages - 1 and acc >= per_page * (len(pages_blocks) + 1):
            # không để tiêu đề mục đứng cuối trang
            if b.page.startswith("<h2"):
                cur.pop()
                pages_blocks.append(cur)
                cur = [b]
            else:
                pages_blocks.append(cur)
                cur = []
    if cur:
        pages_blocks.append(cur)
    pages = [wrap_page(pb, css, lang, font_faces) for pb in pages_blocks]
    gt = html_to_text("\n".join(b.gt for b in blocks if b.gt))
    meta = {"kind": "long_text", "lang": lang, "length_bin": label, "n_chars": len(gt), "pages": len(pages),
            "wiki_title": title, "numbers_changed": changed, **style}
    return pages, gt, "text", meta
