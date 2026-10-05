"""Dòng lệnh `ocrbench`.

    ocrbench make-manifest DATA_DIR -o data/manifest.jsonl     # tạo manifest từ thư mục
    ocrbench validate --config config.yaml                      # kiểm tra dữ liệu + config
    ocrbench run --config config.yaml --gpus 0,1                # chạy các model, rồi chấm điểm
    ocrbench score --config config.yaml                         # chỉ chấm lại + tạo báo cáo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import load_config
from .dataset import fingerprint, load_manifest, make_manifest, summarize


def _split_arg(a) -> str:
    if a.split == "holdout" and not a.confirm_holdout:
        sys.exit(
            "Từ chối chạy trên holdout. Holdout chỉ dùng để chấm lần cuối, sau khi đã chốt model và cấu hình.\n"
            "Nếu đúng là lần chấm cuối, thêm --confirm-holdout."
        )
    return a.split


def _csv(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def cmd_make_manifest(a):
    recs = make_manifest(a.data_dir, a.output, holdout_ratio=a.holdout_ratio, seed=a.seed, doc_sep=a.doc_sep)
    print(f"Đã ghi {len(recs)} mẫu vào {a.output}\n")
    print(summarize(load_manifest(a.output)))


def cmd_build_testset(a):
    from .testset.build import build_testset

    recs = build_testset(
        a.output, public_n=a.public_n, synth_n=a.synth_n, holdout_ratio=a.holdout_ratio, seed=a.seed,
        public=not a.no_public, synth=not a.no_synth, sources=_csv(a.sources), categories=_csv(a.categories),
        scale=a.scale,
    )
    print(f"\nĐã ghi {len(recs)} mẫu vào {a.output}/manifest.jsonl (mô tả: {a.output}/DATASET_CARD.md)\n")
    print(summarize(load_manifest(f"{a.output}/manifest.jsonl")), flush=True)
    # Thư viện `datasets` (chế độ streaming) để lại luồng nền, đôi khi làm Python crash lúc thoát
    # dù dữ liệu đã ghi xong. Mọi file đã đóng, nên thoát thẳng để mã thoát phản ánh đúng kết quả.
    os._exit(0)


def cmd_validate(a):
    cfg = load_config(a.config)
    items = load_manifest(cfg.dataset)
    missing = [it for it in items if not all(p.exists() for p in it.images)]
    print(f"Manifest: {cfg.dataset}")
    print(f"Dấu vân tay dữ liệu: {fingerprint(items)}  (hai máy có cùng dấu vân tay = cùng một bộ test)\n")
    print(summarize(items))
    print()
    if missing:
        print(f"✘ {len(missing)} mẫu thiếu ảnh, ví dụ: {[m.id for m in missing[:3]]}")
    empty = [it.id for it in items if not it.gt.strip()]
    if empty:
        print(f"⚠ {len(empty)} mẫu có đáp án rỗng, ví dụ: {empty[:3]}")
    small = sorted({it.category for it in items if it.split == 'dev'} )
    for c in small:
        n = sum(1 for it in items if it.category == c and it.split == "dev")
        if n < 30:
            print(f"⚠ nhóm '{c}' chỉ có {n} mẫu dev: khoảng tin cậy sẽ rất rộng")
    print(f"\nModel trong config ({len(cfg.models)}):")
    for m in cfg.models:
        flag = "" if m.enabled else "  (enabled: false)"
        print(f"  - {m.name:<28} adapter={m.adapter:<12} gpus={m.gpus}{flag}")
    if missing:
        sys.exit(1)


def cmd_vram(a):
    from .vram import free_vram_gb, report_plans

    cfg = load_config(a.config)
    specs = cfg.select(_csv(a.models)) if a.models else cfg.models
    free = [float(x) for x in a.vram_gb.split(",")] if a.vram_gb else free_vram_gb()
    impossible = report_plans(cfg, specs, free, a.long)
    sys.exit(2 if impossible else 0)


def cmd_decide(a):
    from .decide import decide, write_decision

    cfg = load_config(a.config)
    split = _split_arg(a)
    dec = decide(cfg, split, a.stage, models=_csv(a.models), categories=_csv(a.categories),
                 per_category=a.per_category, max_unstable=a.max_unstable, max_finalists=a.max_finalists)
    path = write_decision(cfg, dec)
    print(path.read_text(encoding="utf-8"))
    print(f"\nĐã ghi: {path} (và bản .json)")


def cmd_status(a):
    from .status import push_status, write_status

    cfg = load_config(a.config)
    if a.push:
        try:
            print(f"✔ STATUS: {push_status(cfg, note=a.note)}")
        except Exception as e:
            sys.exit(f"✘ ĐẨY STATUS THẤT BẠI: {e}")
    else:
        print(f"Đã ghi {write_status(cfg, note=a.note)}")


def cmd_restore(a):
    from .status import restore

    cfg = load_config(a.config)
    try:
        done = restore(cfg)
    except Exception as e:
        sys.exit(f"✘ KHÔI PHỤC THẤT BẠI: {e}")
    print(f"Đã khôi phục {len(done)} file từ nhánh '{cfg.status.branch}':")
    for d in done[:20]:
        print("  " + d)
    if len(done) > 20:
        print(f"  ... và {len(done) - 20} file khác")


def cmd_benchmark(a):
    from .benchmark import write_benchmark

    cfg = load_config(a.config)
    path = write_benchmark(cfg)
    print(path.read_text(encoding="utf-8"))
    print(f"Đã ghi {path}. Đẩy lên GitHub: ocrbench status --config {a.config} --push")


def cmd_goal_check(a):
    from .goal import report, write_lock

    cfg = load_config(a.config)
    if a.write_lock:
        print(f"Đã ghi khóa: {write_lock(cfg)}  (chỉ con người được làm việc này)")
        return
    text, ok = report(cfg)
    print(text)
    sys.exit(0 if ok else 1)


def cmd_clean_cache(a):
    from .goal import clean_cache

    cfg = load_config(a.config)
    try:
        print(clean_cache(cfg, a.item, force=a.force))
    except Exception as e:
        sys.exit(f"✘ {e}")


def cmd_convert(a):
    from .convert import Converter

    conv = Converter(a.config, a.model, dpi=a.dpi, params=_params(a.set), gpus=a.gpus)
    for f in a.inputs:
        res = conv.convert_file(f, a.output, force_ocr=a.force_ocr, pages=a.pages, doc_type=a.doc_type, mode=a.mode,
                                progress=lambda k, n, msg: print(f"  {msg} ({k + 1}/{n})", flush=True))
        n_ocr = sum(1 for p in res.pages if p.source == "OCR")
        n_err = sum(1 for p in res.pages if p.error)
        print(f"✔ {res.docx}  ({len(res.pages)} trang: {n_ocr} qua OCR, {len(res.pages) - n_ocr - n_err} từ lớp chữ"
              + (f", {n_err} LỖI" if n_err else "") + ")")


def cmd_dots_parse(a):
    """Như `python dots_ocr/parser.py <file> --prompt ...` của dots.ocr, chạy bằng HF transformers trên T4."""
    from .adapters.dots import DotsAdapter, parse_file
    from .convert import merge_params

    params = {}
    if a.config and a.model:
        params = load_config(a.config).model(a.model).params
    params = merge_params(params, _params(a.set))
    if a.no_fitz_preprocess:
        params["fitz_preprocess"] = False
    elif a.fitz_preprocess:
        params["fitz_preprocess"] = True
    adapter = DotsAdapter(**params)
    adapter.load()
    for f in a.inputs:
        res = parse_file(adapter, f, a.output, prompt_mode=a.prompt, bbox=a.bbox, dpi=a.dpi)
        print(f"✔ {f}: {len(res)} trang → {a.output}/{Path(f).stem}/"
              + (f"  (⚠ {sum(1 for r in res if r.get('filtered'))} trang JSON hỏng)" if any(r.get('filtered') for r in res)
                 else ""))
    adapter.close()


def cmd_serve(a):
    from .app import serve

    serve(a.config, a.model, host=a.host, port=a.port, share=a.share, dpi=a.dpi,
          params=_params(a.set), gpus=a.gpus, text_layer=not a.ocr_all, title=a.title)


def _params(sets):
    from .speedtest import parse_variant

    return parse_variant(",".join(sets)) if sets else None


def cmd_speedtest(a):
    from .speedtest import DEFAULT_VARIANTS, run_speedtest

    cfg = load_config(a.config)
    out = cfg.work_dir / f"speedtest_{a.model}.md"
    print("\n" + run_speedtest(a.config, a.model, a.variant or DEFAULT_VARIANTS, a.per_category, a.gpus, out))
    print(f"\nĐã ghi {out}")


def cmd_run(a):
    from .report import build_report
    from .runner import run_all

    cfg = load_config(a.config)
    split = _split_arg(a)
    specs = cfg.select(_csv(a.models))
    if not specs:
        sys.exit("Không có model nào để chạy (kiểm tra 'enabled' hoặc --models).")
    results = run_all(
        cfg, specs, split, categories=_csv(a.categories), limit=a.limit, retry_errors=a.retry_errors,
        gpus=_csv(a.gpus), inline=a.inline, per_category=a.per_category, shard=not a.no_shard,
    )
    failed = [m for m, code in results.items() if code != 0]
    if not a.no_score:
        items = load_manifest(cfg.dataset, split=split, categories=_csv(a.categories), limit=a.limit,
                              per_category=a.per_category)
        out = build_report(cfg, split, [s.name for s in specs], items)
        print(f"\nBáo cáo: {out / 'report.md'}")
    if failed:
        print(f"\n⚠ Model kết thúc với lỗi: {failed}. Chạy lại lệnh để tiếp tục từ chỗ dừng.")
        sys.exit(1)


def cmd_score(a):
    from .report import build_report
    from .score import list_scored_models

    cfg = load_config(a.config)
    split = _split_arg(a)
    models = _csv(a.models) or list_scored_models(cfg, split)
    if not models:
        sys.exit(f"Chưa có kết quả nào trong {cfg.output_dir / split}")
    items = load_manifest(cfg.dataset, split=split, categories=_csv(a.categories), limit=a.limit,
                              per_category=a.per_category)
    out = build_report(cfg, split, models, items)
    print(f"Báo cáo: {out / 'report.md'}\nCSV: {out / 'summary.csv'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ocrbench", description="So sánh nhiều model OCR trên cùng một bộ test")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-manifest", help="tạo manifest.jsonl từ thư mục DATA_DIR/<nhóm>/<file>")
    p.add_argument("data_dir")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--holdout-ratio", type=float, default=0.5, help="tỉ lệ doc_id đưa vào holdout (mặc định 0.5)")
    p.add_argument("--seed", default="ocrbench")
    p.add_argument("--doc-sep", default="__", help="ký tự ngăn doc_id và số trang trong tên file")
    p.set_defaults(func=cmd_make_manifest)

    p = sub.add_parser("build-testset", help="dựng bộ test từ dữ liệu công khai + tài liệu tổng hợp")
    p.add_argument("output")
    p.add_argument("--public-n", type=int, default=80, help="số mẫu lấy từ MỖI nguồn công khai")
    p.add_argument("--synth-n", type=int, default=80, help="số tài liệu tổng hợp cho MỖI nhóm syn_*")
    p.add_argument("--holdout-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-public", action="store_true")
    p.add_argument("--no-synth", action="store_true")
    p.add_argument("--sources", help="chỉ các nguồn công khai này (vd. misraj_dococr,iam_lines)")
    p.add_argument("--categories", help="chỉ các nhóm tổng hợp này (vd. syn_invoice_ar)")
    p.add_argument("--scale", type=float, default=1.5, help="độ phân giải ảnh tổng hợp (1.5 ≈ 180 dpi)")
    p.set_defaults(func=cmd_build_testset)

    p = sub.add_parser("validate", help="kiểm tra manifest và config")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("status", help="ghi status.md (tiến độ + kết quả nhanh); --push để đẩy lên GitHub")
    p.add_argument("--config", required=True)
    p.add_argument("--push", action="store_true")
    p.add_argument("--note", help="ghi chú ngắn kèm theo (vd. 'xong giai đoạn 2')")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("restore", help="kéo kết quả đã đẩy (nhánh results) về máy này để chạy tiếp")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser("convert", help="chuyển PDF/ảnh sang DOCX bằng một model (không cần giao diện)")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("-o", "--output", default="docx_out")
    p.add_argument("--force-ocr", action="store_true", help="OCR mọi trang, kể cả trang PDF đã có lớp chữ")
    p.add_argument("--pages", help="vd. 1-3,5")
    p.add_argument("--doc-type", choices=["text", "table"], default="text")
    p.add_argument("--mode", default=None, help="chế độ đọc trong params.modes của model (vd. 'Chỉ chữ')")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--set", action="append", help="ghi đè tham số model, vd. --set batch_size=4 --set stop_on_loop=true")
    p.add_argument("--gpus", default="auto", help="auto = mọi GPU (một bản model mỗi GPU); vd. 0 hoặc 0,1")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("dots-parse", help="chạy dots.ocr/dots.mocr giống tool gốc (json, jpg bố cục, md, _nohf.md)")
    p.add_argument("inputs", nargs="+", help="PDF / jpg / png")
    p.add_argument("-o", "--output", default="dots_out")
    p.add_argument("--prompt", default=None,
                   choices=["prompt_layout_all_en", "prompt_layout_only_en", "prompt_ocr", "prompt_grounding_ocr",
                            "prompt_web_parsing", "prompt_scene_spotting", "prompt_image_to_svg", "prompt_general"],
                   help="chế độ (mặc định: prompt_mode trong config, hoặc prompt_layout_all_en)")
    p.add_argument("--bbox", type=int, nargs=4, metavar=("x1", "y1", "x2", "y2"), help="cho prompt_grounding_ocr")
    p.add_argument("--config", help="lấy tham số từ mục model trong config (vd. dots_mocr)")
    p.add_argument("--model", help="tên mục model trong config")
    p.add_argument("--dpi", type=int, default=200)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fitz-preprocess", action="store_true", help="như tool gốc mặc định: render lại ảnh theo dpi")
    g.add_argument("--no-fitz-preprocess", action="store_true")
    p.add_argument("--set", action="append", help="ghi đè tham số, vd. --set max_new_tokens=16384")
    p.set_defaults(func=cmd_dots_parse)

    p = sub.add_parser("serve", help="giao diện web: kéo thả PDF/ảnh → tải DOCX")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1", help="mặc định chỉ máy này truy cập; 0.0.0.0 = mọi máy cùng mạng")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="tạo link công khai của Gradio (KHÔNG dùng với tài liệu mật)")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--set", action="append", help="ghi đè tham số model, vd. --set batch_size=4 --set stop_on_loop=true")
    p.add_argument("--gpus", default="auto", help="auto = mọi GPU (một bản model mỗi GPU); vd. 0 hoặc 0,1")
    p.add_argument("--ocr-all", action="store_true", help="mặc định bỏ chọn 'Dùng lớp chữ PDF' (mọi trang qua model)")
    p.add_argument("--title", help="tiêu đề trang web")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("speedtest", help="đo tốc độ + độ chính xác của các cấu hình tối ưu (chọn cấu hình triển khai)")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--variant", action="append",
                   help="vd. 'max_new_tokens=4096,batch_size=4,stop_on_loop=true' (lặp lại để thêm biến thể)")
    p.add_argument("--per-category", type=int, default=3)
    p.add_argument("--gpus", default="auto")
    p.set_defaults(func=cmd_speedtest)

    p = sub.add_parser("benchmark", help="sinh BENCHMARK.md: bảng xếp hạng các model trong hàng đợi /goal")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("goal-check", help="trạng thái hàng đợi /goal, việc tiếp theo, và GOAL: ĐẠT khi xong")
    p.add_argument("--config", required=True)
    p.add_argument("--write-lock", action="store_true", help="CHỈ CON NGƯỜI: tạo lại goal/GOAL_LOCK.sha256")
    p.set_defaults(func=cmd_goal_check)

    p = sub.add_parser("clean-cache", help="xóa trọng số Hugging Face của một model sau khi benchmark đã lên GitHub")
    p.add_argument("--config", required=True)
    p.add_argument("--item", required=True, help="tên model trong goal/queue.yaml (xóa cả các biến thể)")
    p.add_argument("--force", action="store_true", help="chỉ dùng cho model đã ghi vào goal_skips.yaml")
    p.set_defaults(func=cmd_clean_cache)

    p = sub.add_parser("decide", help="áp luật loại/chọn model theo từng nhóm (không phải tự tính)")
    p.add_argument("--config", required=True)
    p.add_argument("--stage", required=True, choices=["screening", "final"])
    p.add_argument("--models")
    p.add_argument("--split", default="dev", choices=["dev", "holdout"])
    p.add_argument("--confirm-holdout", action="store_true")
    p.add_argument("--categories")
    p.add_argument("--per-category", type=int, help="phải GIỐNG HỆT giá trị đã dùng khi run (vd. 20 ở vòng sàng lọc)")
    p.add_argument("--max-unstable", type=float, default=0.15)
    p.add_argument("--max-finalists", type=int, default=4)
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("vram", help="kiểm tra GPU, ước lượng VRAM và chọn cấu hình chạy được cho từng model")
    p.add_argument("--config", required=True)
    p.add_argument("--models", help="chỉ các model này (mặc định: mọi model trong config, kể cả đang tắt)")
    p.add_argument("--long", action="store_true", help="tính cho tài liệu dài 2–3 trang (cần nhiều VRAM hơn)")
    p.add_argument("--vram-gb", help="bỏ qua nvidia-smi, dùng VRAM trống cho trước, vd. 15,15")
    p.set_defaults(func=cmd_vram)

    for name, func, help_ in [("run", cmd_run, "chạy model rồi chấm điểm"), ("score", cmd_score, "chỉ chấm điểm + báo cáo")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", required=True)
        p.add_argument("--models", help="chỉ chạy các model này (phân cách bằng dấu phẩy)")
        p.add_argument("--split", default="dev", choices=["dev", "holdout"])
        p.add_argument("--confirm-holdout", action="store_true")
        p.add_argument("--categories", help="chỉ các nhóm này")
        p.add_argument("--limit", type=int, help="chỉ N mẫu đầu (để thử nhanh)")
        p.add_argument("--per-category", type=int, help="tập con: N mẫu mỗi nhóm, cố định cho mọi model")
        if name == "run":
            p.add_argument("--gpus", help="vd. 0,1: chạy song song, mỗi model một GPU")
            p.add_argument("--retry-errors", action="store_true", help="chạy lại các mẫu bị lỗi")
            p.add_argument("--inline", action="store_true", help="chạy ngay trong tiến trình này (để debug)")
            p.add_argument("--no-shard", action="store_true",
                           help="không chia mẫu cho nhiều GPU khi chỉ chạy một model vừa 1 GPU")
            p.add_argument("--no-score", action="store_true")
        p.set_defaults(func=func)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
