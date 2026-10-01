"""Mẫu tài liệu tổng hợp: hợp đồng, công văn, hóa đơn, biểu mẫu điền tay.

Mỗi tài liệu được dựng từ các "khối" (Block). Mỗi khối có phần hiển thị trên trang
(`page`) và phần đáp án (`gt`). Các hình trang trí (logo, chữ ký, con dấu) chỉ có phần
`page`, nên không xuất hiện trong đáp án. Thứ tự khối trong HTML cũng là thứ tự đọc
của đáp án.
"""

from __future__ import annotations

import html
import random
import re
from dataclasses import dataclass, field

from ..normalize import to_plain_text
from . import content as C
from .fonts import FONTS

KINDS = ("contract", "letter", "invoice", "form")


@dataclass
class Block:
    page: str
    gt: str | None = None  # None = trang trí, không có trong đáp án


@dataclass
class Doc:
    kind: str
    lang: str
    page_html: str
    gt: str
    gt_type: str  # "text" | "table_html"
    meta: dict = field(default_factory=dict)


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def text_block(tag: str, text: str, cls: str = "") -> Block:
    c = f' class="{cls}"' if cls else ""
    return Block(f"<{tag}{c}>{esc(text)}</{tag}>", f"<{tag}>{esc(text)}</{tag}>")


# --- Trang trí (không có chữ) --------------------------------------------------------


def logo_svg(rng: random.Random) -> Block:
    color = rng.choice(["#1f4e79", "#7b2d26", "#2e6b3a", "#5b3a8a", "#b8860b", "#333"])
    shape = rng.choice([
        f'<circle cx="30" cy="30" r="26" fill="{color}"/><circle cx="30" cy="30" r="14" fill="white"/>',
        f'<rect x="6" y="6" width="48" height="48" rx="10" fill="{color}"/><rect x="20" y="20" width="20" height="20" fill="white"/>',
        f'<polygon points="30,4 56,54 4,54" fill="{color}"/><polygon points="30,22 44,48 16,48" fill="white"/>',
    ])
    return Block(f'<div class="logo"><svg width="60" height="60">{shape}</svg></div>')


def signature_svg(rng: random.Random) -> str:
    x, y = 10, 40
    d = f"M{x},{y}"
    for _ in range(rng.randint(4, 7)):
        x += rng.randint(15, 35)
        d += f" C{x - 20},{rng.randint(0, 60)} {x - 10},{rng.randint(0, 60)} {x},{rng.randint(20, 50)}"
    ink = rng.choice(["#1a3a8a", "#111", "#0b2f6b"])
    return f'<svg class="sig" width="{x + 20}" height="60"><path d="{d}" stroke="{ink}" stroke-width="2" fill="none"/></svg>'


def stamp_svg(rng: random.Random) -> str:
    color = rng.choice(["#2045a8", "#7a1f6e", "#a82020"])
    rot = rng.randint(-25, 25)
    return (
        f'<svg class="stamp" width="120" height="120" style="transform:rotate({rot}deg);opacity:.75">'
        f'<circle cx="60" cy="60" r="54" stroke="{color}" stroke-width="4" fill="none"/>'
        f'<circle cx="60" cy="60" r="40" stroke="{color}" stroke-width="2" fill="none" stroke-dasharray="6 4"/>'
        f'<polygon points="60,38 66,54 83,54 69,64 74,81 60,71 46,81 51,64 37,54 54,54" fill="{color}"/></svg>'
    )


# --- Kiểu trình bày ---------------------------------------------------------------------


def base_css(rng: random.Random, lang: str) -> tuple[str, dict]:
    if lang == "en":
        font = rng.choice(FONTS["print_en"])
    else:
        font = rng.choice(FONTS["print_ar"])
    size = rng.randint(14, 18)
    pad = rng.randint(50, 90)
    accent = rng.choice(["#1f4e79", "#7b2d26", "#2e6b3a", "#444", "#5b3a8a"])
    table_style = rng.choice(["grid", "lines", "zebra", "header"])
    align_head = rng.choice(["start", "center"])
    css = f"""
    body {{ margin:0; background:#fff; }}
    .page {{ width:1000px; box-sizing:border-box; padding:{pad}px; font-family:'{font}', 'Noto Naskh Arabic', 'DejaVu Sans', sans-serif;
             font-size:{size}px; line-height:1.6; color:#111; }}
    h1 {{ font-size:1.6em; margin:.2em 0; color:{accent}; text-align:{align_head}; }}
    h2 {{ font-size:1.25em; margin:.8em 0 .4em; text-align:{align_head}; }}
    p {{ margin:.25em 0; }}
    .clause {{ text-align:justify; }}
    .logo {{ margin-bottom:6px; text-align:{align_head}; }}
    table {{ border-collapse:collapse; width:100%; margin:.6em 0; }}
    th, td {{ padding:6px 8px; text-align:start; vertical-align:top; }}
    td.num {{ white-space:nowrap; }}
    .totals {{ width:55%; }}
    .sigrow {{ display:flex; justify-content:space-between; margin-top:2.2em; align-items:flex-end; gap:30px; }}
    .sigbox {{ min-width:260px; }}
    .line {{ display:inline-block; border-bottom:1px solid #555; min-width:180px; height:1.2em; vertical-align:bottom; }}
    .field {{ margin:.55em 0; border-bottom:1px dotted #999; padding-bottom:2px; }}
    .label {{ font-weight:600; }}
    .hw {{ display:inline-block; margin-inline-start:14px; }}
    """
    css += {
        "grid": "table, th, td { border:1px solid #333; } th { background:#eee; }",
        "lines": "th { border-bottom:2px solid #222; } td { border-bottom:1px solid #bbb; }",
        "zebra": f"tr:nth-child(even) td {{ background:#f3f3f3; }} th {{ border-bottom:2px solid {accent}; }}",
        "header": f"th {{ background:{accent}; color:#fff; }} td {{ border:1px solid #ccc; }}",
    }[table_style]
    return css, {"font": font, "font_size": size, "table_style": table_style}


# Chuỗi Latinh/chữ số nhiều từ (số điện thoại, email, IBAN, tên sản phẩm tiếng Anh...)
_LTR_RUN = re.compile(r"[+]?[0-9\u0660-\u0669A-Za-z][0-9\u0660-\u0669A-Za-z@._:/+\-]*"
                      r"(?: +[0-9\u0660-\u0669A-Za-z@._:/+\-]+)+")


def isolate_ltr_runs(body: str) -> str:
    """Trong văn bản RTL, thuật toán bidi tách chuỗi '+966 58 440 7282' thành từng cụm và xếp
    ngược lại. Tài liệu thật giữ nguyên các chuỗi này theo chiều trái-sang-phải, nên bọc chúng
    trong <span dir="ltr">. Chỉ sửa phần hiển thị, đáp án giữ nguyên thứ tự logic."""
    def fix_text(m):
        return ">" + _LTR_RUN.sub(lambda r: f'<span dir="ltr">{r.group(0)}</span>', m.group(1)) + "<"
    return re.sub(r">([^<>]+)<", fix_text, body)


def wrap_page(blocks: list[Block], css: str, lang: str, font_faces: str) -> str:
    direction = "rtl" if lang in ("ar", "mixed") else "ltr"
    body = "\n".join(b.page for b in blocks)
    if direction == "rtl":
        body = isolate_ltr_runs(body)
    return (f'<!doctype html><html lang="{"en" if lang == "en" else "ar"}" dir="{direction}"><head><meta charset="utf-8">'
            f"<style>{font_faces}\n{css}</style></head><body><div class=\"page\">{body}</div></body></html>")


def gt_of(blocks: list[Block]) -> str:
    return "\n".join(b.gt for b in blocks if b.gt)


def html_to_text(gt_html: str) -> str:
    text = to_plain_text(gt_html)
    return re.sub(r"\n{2,}", "\n", "\n".join(line.strip() for line in text.splitlines())).strip()


# --- Các loại tài liệu ------------------------------------------------------------------


def contract(c: C.Content) -> list[Block]:
    rng = c.rng
    blocks = [logo_svg(rng)] if rng.random() < 0.6 else []
    p1, p2 = c.company(), c.company()
    city, _ = c.city_country()
    if c.ar:
        blocks.append(text_block("h1", c.pick(C.AR_CONTRACT_TITLES)))
        blocks.append(text_block("p", f"رقم العقد: {c.num(c.ref())}"))
        blocks.append(text_block("p", f"إنه في يوم {c.pick(C.AR_WEEKDAYS)} الموافق {c.date()}، تم الاتفاق بين كل من:"))
        blocks.append(text_block("p", f"الطرف الأول: {p1}، ومقرها {city}، ويمثلها السيد {c.person(male=True)} بصفته {c.job_title()}."))
        blocks.append(text_block("p", f"الطرف الثاني: {p2}، ويمثلها السيد {c.person(male=True)} بصفته {c.job_title()}."))
        if c.lang == "mixed":
            blocks.append(text_block("p", f"رقم المشروع: {c.code()}، البريد الإلكتروني: {c.email()}"))
        blocks.append(text_block("p", f"تمهيد: حيث إن الطرف الأول يرغب في {c.pick(C.AR_PURPOSES)}، وحيث إن الطرف الثاني "
                                      "لديه الخبرة والإمكانات اللازمة لذلك، فقد اتفق الطرفان على ما يلي:", "clause"))
        clauses = rng.sample(C.AR_CLAUSES, rng.randint(4, 7))
        for i, cl in enumerate(clauses):
            blocks.append(text_block("p", f"البند {C.AR_ORDINALS[i]}: {c.fill(cl)}", "clause"))
        labels = ("الطرف الأول", "الطرف الثاني", "الاسم", "التوقيع")
    else:
        blocks.append(text_block("h1", c.pick(C.EN_CONTRACT_TITLES)))
        blocks.append(text_block("p", f"Contract No.: {c.ref()}"))
        blocks.append(text_block("p", f"This Agreement is made on {c.pick(C.EN_WEEKDAYS)}, {c.date()}, between:"))
        blocks.append(text_block("p", f"First Party: {p1}, located in {city}, represented by {c.person()}, {c.job_title()}."))
        blocks.append(text_block("p", f"Second Party: {p2}, represented by {c.person()}, {c.job_title()}."))
        blocks.append(text_block("p", f"Whereas the First Party wishes to engage the Second Party for {c.pick(C.EN_PURPOSES)}, "
                                      "the parties have agreed as follows:", "clause"))
        clauses = rng.sample(C.EN_CLAUSES, rng.randint(4, 7))
        for i, cl in enumerate(clauses, 1):
            blocks.append(text_block("p", f"{i}. {c.fill(cl)}", "clause"))
        labels = ("First Party", "Second Party", "Name", "Signature")
    blocks.append(signature_row(c, labels))
    return blocks


def signature_row(c: C.Content, labels) -> Block:
    rng = c.rng
    boxes_page, boxes_gt = [], []
    for who in labels[:2]:
        sig = signature_svg(rng) if rng.random() < 0.8 else '<span class="line"></span>'
        boxes_page.append(
            f'<div class="sigbox"><p><b>{esc(who)}</b></p><p>{esc(labels[2])}: <span class="line"></span></p>'
            f"<p>{esc(labels[3])}: {sig}</p></div>"
        )
        boxes_gt.append(f"<p>{esc(who)}</p><p>{esc(labels[2])}:</p><p>{esc(labels[3])}:</p>")
    stamp = stamp_svg(rng) if rng.random() < 0.5 else ""
    return Block(f'<div class="sigrow">{boxes_page[0]}{stamp}{boxes_page[1]}</div>', "".join(boxes_gt))


def letter(c: C.Content) -> list[Block]:
    rng = c.rng
    blocks = [logo_svg(rng)] if rng.random() < 0.7 else []
    sender = c.company()
    blocks.append(text_block("h1", sender))
    blocks.append(text_block("p", c.address()))
    if c.ar:
        blocks.append(text_block("p", f"الرقم: {c.num(c.ref())}    التاريخ: {c.date()}"))
        blocks.append(text_block("p", f"السادة/ {c.company()} المحترمين"))
        blocks.append(text_block("p", "السلام عليكم ورحمة الله وبركاته، وبعد:"))
        blocks.append(text_block("p", f"الموضوع: {c.pick(C.AR_LETTER_SUBJECTS)}"))
        if c.lang == "mixed":
            blocks.append(text_block("p", f"المرجع: {c.code()} - {c.pick(C.MIXED_PRODUCTS)}"))
        bodies = rng.sample(C.AR_LETTER_BODIES, rng.randint(3, 5))
        closing = ["وتفضلوا بقبول فائق الاحترام والتقدير،", c.job_title(), c.person()]
    else:
        blocks.append(text_block("p", f"Ref.: {c.ref()}    Date: {c.date()}"))
        blocks.append(text_block("p", f"To: {c.company()}"))
        blocks.append(text_block("p", "Dear Sir or Madam,"))
        blocks.append(text_block("p", f"Subject: {c.pick(C.EN_LETTER_SUBJECTS)}"))
        bodies = rng.sample(C.EN_LETTER_BODIES, rng.randint(3, 5))
        closing = ["Yours faithfully,", c.person(), c.job_title()]
    for b in bodies:
        blocks.append(text_block("p", c.fill(b), "clause"))
    for line in closing:
        blocks.append(text_block("p", line))
    blocks.append(Block(signature_svg(rng)))
    if rng.random() < 0.5:
        blocks.append(Block(stamp_svg(rng)))
    return blocks


def invoice(c: C.Content) -> list[Block]:
    rng = c.rng
    blocks = [logo_svg(rng)] if rng.random() < 0.7 else []
    vat_rate = rng.choice([5, 15])
    blocks.append(text_block("h1", c.company()))
    blocks.append(text_block("p", c.address()))
    if c.ar:
        L = dict(title="فاتورة ضريبية", no="رقم الفاتورة", date="تاريخ الفاتورة", cust="العميل", addr="عنوان العميل",
                 vatno="الرقم الضريبي", phone="الهاتف", cols=["م", "الوصف", "الكمية", "سعر الوحدة", "الإجمالي"],
                 sub="المجموع قبل الضريبة", vat=f"ضريبة القيمة المضافة ({c.num(vat_rate)}%)", total="الإجمالي المستحق",
                 terms="شروط الدفع: خلال {d} يوماً من تاريخ الفاتورة", thanks="شكراً لتعاملكم معنا")
    else:
        L = dict(title="Tax Invoice", no="Invoice No.", date="Invoice Date", cust="Bill To", addr="Customer Address",
                 vatno="VAT No.", phone="Phone", cols=["#", "Description", "Qty", "Unit Price", "Amount"],
                 sub="Subtotal", vat=f"VAT ({vat_rate}%)", total="Total Due",
                 terms="Payment terms: within {d} days of the invoice date", thanks="Thank you for your business")
    currency = c.currency()
    blocks.append(text_block("p", f"{L['vatno']}: {c.num('3' + str(rng.randint(10**12, 10**13 - 1)) + '3')}"))
    blocks.append(text_block("p", f"{L['phone']}: {c.phone()}    {c.email()}"))
    blocks.append(text_block("h2", L["title"]))
    blocks.append(text_block("p", f"{L['no']}: {c.code()}"))
    blocks.append(text_block("p", f"{L['date']}: {c.date()}"))
    blocks.append(text_block("p", f"{L['cust']}: {c.company()}"))
    blocks.append(text_block("p", f"{L['addr']}: {c.address()}"))

    rows, subtotal = [], 0.0
    for i in range(1, rng.randint(3, 12) + 1):
        qty = rng.randint(1, 50)
        price = rng.randint(5, 4000) + rng.choice([0, 0.5, 0.25, 0.99])
        amount = qty * price
        subtotal += amount
        rows.append([c.num(i), c.product(), c.num(qty), c.money(price), c.money(amount)])
    head = "<tr>" + "".join(f"<th>{esc(h)}</th>" for h in L["cols"]) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f'<td{" class=num" if j in (0, 2, 3, 4) else ""}>{esc(v)}</td>' for j, v in enumerate(r)) + "</tr>"
        for r in rows
    )
    items_html = f"<table>{head}{body}</table>"
    blocks.append(Block(items_html, re.sub(r" class=num", "", items_html)))

    vat = subtotal * vat_rate / 100
    tot_rows = [(L["sub"], subtotal), (L["vat"], vat), (L["total"], subtotal + vat)]
    tot_html = "<table class=\"totals\">" + "".join(
        f"<tr><td>{esc(k)}</td><td class=num>{esc(c.money(v) + ' ' + currency)}</td></tr>" for k, v in tot_rows
    ) + "</table>"
    blocks.append(Block(tot_html, tot_html.replace(' class="totals"', "").replace(" class=num", "")))
    blocks.append(text_block("p", L["terms"].format(d=c.num(rng.choice([15, 30, 45, 60])))))
    iban = f"IBAN: SA{rng.randint(10, 99)} {' '.join(str(rng.randint(1000, 9999)) for _ in range(5))}"
    blocks.append(text_block("p", iban))
    blocks.append(text_block("p", L["thanks"]))
    if rng.random() < 0.6:
        blocks.append(Block(f'<div class="sigrow"><span></span>{stamp_svg(rng)}</div>'))
    return blocks


def form(c: C.Content, hw_font: str, hw_font_latin: str) -> list[Block]:
    """Biểu mẫu in sẵn, phần điền bằng font giả chữ viết tay (mực xanh, hơi nghiêng)."""
    rng = c.rng
    blocks = [logo_svg(rng)] if rng.random() < 0.5 else []
    city, country = c.city_country()
    if c.ar:
        title, reasons = c.pick(C.AR_FORMS)
        blocks.append(text_block("h1", title))
        fields = [("الاسم الكامل", c.person()), ("رقم الهوية", c.num(rng.randint(10**9, 10**10 - 1))),
                  ("تاريخ الميلاد", c.date((1960, 2004))), ("رقم الجوال", c.phone()), ("البريد الإلكتروني", c.email()),
                  ("العنوان", c.address()), ("جهة العمل", c.company()), ("المسمى الوظيفي", c.job_title()),
                  ("سبب الطلب", c.pick(reasons)), ("التاريخ", c.date())]
        sig_label = "التوقيع"
    else:
        title, reasons = c.pick(C.EN_FORMS)
        blocks.append(text_block("h1", title))
        fields = [("Full Name", c.person()), ("ID Number", str(rng.randint(10**9, 10**10 - 1))),
                  ("Date of Birth", c.date((1960, 2004))), ("Mobile", c.phone()), ("Email", c.email()),
                  ("Address", c.address()), ("Employer", c.company()), ("Job Title", c.job_title()),
                  ("Reason", c.pick(reasons)), ("Date", c.date())]
        sig_label = "Signature"
    fields = fields[:3] + rng.sample(fields[3:], rng.randint(4, len(fields) - 3))
    ink = rng.choice(["#1a3a8a", "#0b2f6b", "#222", "#1d4fa0"])
    size = rng.randint(20, 27)
    for label, value in fields:
        has_arabic = bool(re.search(r"[؀-ۿ]", value))
        font = hw_font if has_arabic else hw_font_latin
        rot = rng.uniform(-2.5, 2.5)
        page = (f'<div class="field"><span class="label">{esc(label)}:</span>'
                f'<span class="hw" style="font-family:\'{font}\';font-size:{size}px;color:{ink};'
                f'transform:rotate({rot:.1f}deg)">{esc(value)}</span></div>')
        blocks.append(Block(page, f"<p>{esc(label)}: {esc(value)}</p>"))
    blocks.append(Block(f'<div class="field"><span class="label">{esc(sig_label)}:</span> {signature_svg(rng)}</div>',
                        f"<p>{esc(sig_label)}:</p>"))
    return blocks


def make_doc(kind: str, lang: str, rng: random.Random, font_faces: str) -> Doc:
    c = C.Content(rng, lang)
    css, style = base_css(rng, lang)
    meta = {"kind": kind, "lang": lang, "arabic_digits": c.arabic_digits, **style}
    if kind == "contract":
        blocks = contract(c)
    elif kind == "letter":
        blocks = letter(c)
    elif kind == "invoice":
        blocks = invoice(c)
    elif kind == "form":
        hw = rng.choice(FONTS["hand_ar"])
        hw_latin = rng.choice(FONTS["hand_en"])
        meta["hand_font"] = hw if c.ar else hw_latin
        blocks = form(c, hw, hw_latin)
    else:
        raise ValueError(kind)
    gt_html = gt_of(blocks)
    if kind == "invoice":
        gt, gt_type = gt_html, "table_html"
    else:
        gt, gt_type = html_to_text(gt_html), "text"
    return Doc(kind, lang, wrap_page(blocks, css, lang, font_faces), gt, gt_type, meta)
