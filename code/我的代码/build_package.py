"""Assemble the submission package for the byte-domain experiment course.

Layout follows docs/student_report_template.md from the course repository:
required baseline/comparison directories keep the names the template asks for,
the full course source is excluded per the template's "do not submit" list.
"""

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(r"D:\zijie")
BUILD = ROOT / "submission_build"
PKG = BUILD / "pkg" / "学号_姓名_码流图像分类"
RES = ROOT / "results_basic"
REPO = BUILD / "_repo_dl" / "repo"

# (source, destination-relative-path)
FILE_MAP = [
    # --- baseline training run: course_clean/ ---
    (RES / "best.pt", "course_clean/best.pt"),
    (RES / "baseline_metrics.json", "course_clean/metrics.json"),
    (RES / "baseline_history.csv", "course_clean/history.csv"),
    (RES / "baseline_curves.png", "course_clean/curves.png"),
    (RES / "prediction_single.png", "course_clean/prediction_single.png"),
    (RES / "baseline_confusion_matrix.png", "course_clean/confusion_matrix.png"),
    # --- baseline independent test evaluation: course_clean_eval/ ---
    (RES / "baseline_evaluation.json", "course_clean_eval/evaluation.json"),
    (RES / "baseline_predictions.png", "course_clean_eval/predictions.png"),
    (RES / "baseline_confusion_matrix.png", "course_clean_eval/confusion_matrix.png"),
    (RES / "baseline_confusion_matrix.csv", "course_clean_eval/confusion_matrix.csv"),
    # --- batch-size comparison run: comparison/ ---
    (RES / "comparison_batch16_best.pt", "comparison/best.pt"),
    (RES / "comparison_metrics.json", "comparison/metrics.json"),
    (RES / "comparison_history.csv", "comparison/history.csv"),
    (RES / "comparison_curves.png", "comparison/curves.png"),
    (RES / "comparison_evaluation.json", "comparison/evaluation.json"),
    (RES / "comparison_predictions.png", "comparison/predictions.png"),
    (RES / "comparison_confusion_matrix.png", "comparison/confusion_matrix.png"),
    (RES / "comparison_confusion_matrix.csv", "comparison/confusion_matrix.csv"),
    # --- readme / reports ---
    (ROOT / "提交说明_码流图像分类实验提交.md", "提交说明.md"),
    (BUILD / "report" / "实验报告.pdf", "报告/实验报告.pdf"),
    (ROOT / "deliverables" / "字节域语义内容理解_实验报告.docx", "报告/实验报告.docx"),
    (ROOT / "实验汇报总结_ByteFormer_MNIST.md", "报告/实验汇报总结_ByteFormer_MNIST.md"),
    # --- student-authored code ---
    (ROOT / "bonus_augmentation.py", "code/我的代码/bonus_augmentation.py"),
    (ROOT / "make_deliverables.py", "code/我的代码/make_deliverables.py"),
    (BUILD / "md_to_pdf.py", "code/我的代码/md_to_pdf.py"),
    (BUILD / "build_package.py", "code/我的代码/build_package.py"),
]

COURSE_CODE = [
    "build_course_dataset.py", "byteformer_model.py", "course_corruption.py",
    "course_kaggle.ipynb", "data_utils.py", "evaluate.py",
    "evaluate_course_corruption.py", "predict.py", "prepare.py",
    "summarize_course_experiments.py", "train.py", "train_course_subset.py",
    "verify_model.py", "requirements.txt", "requirements-local.txt",
    "requirements-reproduce.txt",
]

EXCLUDE_FROM_SMALL = {"course_clean/best.pt", "comparison/best.pt"}


def human(num: int) -> str:
    return f"{num:,} B ({num / 1024 / 1024:.2f} MB)"


def main() -> int:
    if PKG.exists():
        shutil.rmtree(PKG)
    for src, rel in FILE_MAP:
        if not src.exists():
            print(f"[MISS] {src}")
            continue
        dst = PKG / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for name in COURSE_CODE:
        src = REPO / name
        if not src.exists():
            print(f"[MISS] course file {name}")
            continue
        dst = PKG / "code" / "课程仓库源码" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    entries = sorted(p for p in PKG.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in entries)
    print(f"\n[PKG ] {PKG}")
    print(f"[PKG ] {len(entries)} files, {human(total)}")

    manifest = {
        "files": [
            {"path": p.relative_to(PKG).as_posix(), "bytes": p.stat().st_size}
            for p in entries
        ],
        "total_bytes": total,
    }
    (BUILD / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    zips = [
        (
            ROOT / "学号_姓名_码流图像分类_完整版.zip",
            lambda rel: True,
        ),
        (
            ROOT / "学号_姓名_码流图像分类_小体积版_可作邮件附件.zip",
            lambda rel: rel not in EXCLUDE_FROM_SMALL,
        ),
    ]
    for zip_path, keep in zips:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in entries:
                rel = p.relative_to(PKG).as_posix()
                if not keep(rel):
                    continue
                zf.write(p, arcname=f"学号_姓名_码流图像分类/{rel}")
        print(f"[ZIP ] {zip_path.name}: {human(zip_path.stat().st_size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
