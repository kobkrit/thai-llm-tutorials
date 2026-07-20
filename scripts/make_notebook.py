#!/usr/bin/env python3
"""Generate a tutorial notebook from the shared skeleton.

Every notebook in the series is produced by this script so that the setup cell
and the baseline-evaluation cell cannot drift apart by hand-editing. The setup
cell body lives in notebooks/cell0_setup.ipy and is inserted verbatim.

    python3 scripts/make_notebook.py --slug _template --post 0 \\
        --title "แม่แบบ (template)"

Verify afterwards with:  python3 scripts/check_cell0.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
CELL0_PATH = NOTEBOOKS / "cell0_setup.ipy"

REPO = "kobkrit/thai-llm-tutorials"

# The ten sections every post follows. Section 2 is the baseline measurement and
# section 8 is the results table -- those two are fixed by the series' format.
SECTIONS = [
    "ปัญหาคืออะไร และทำไมต้องแก้",
    "ตั้งค่า และวัด baseline ก่อนลงมือทำอะไรทั้งสิ้น",
    "ทฤษฎีเท่าที่จำเป็น",
    "เตรียมข้อมูล",
    "ลงมือทำ (implementation)",
    "รันจริง",
    "วัดผลซ้ำด้วยชุดวัดเดิม",
    "ตารางผลลัพธ์",
    "วิเคราะห์: อะไรได้ผล อะไรไม่ได้ผล",
    "สรุป และตอนต่อไป",
]


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build(slug: str, post: int, title: str, subtitle: str) -> dict:
    if not CELL0_PATH.exists():
        raise SystemExit(f"missing {CELL0_PATH}")
    cell0 = CELL0_PATH.read_text(encoding="utf-8").rstrip("\n")

    badge_url = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{slug}.ipynb"
    outline = "\n".join(f"{i}. {name}" for i, name in enumerate(SECTIONS, 1))

    header = f"""# ตอนที่ {post}: {title}

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({badge_url})

{subtitle}

โดย **ดร.กอบกฤตย์ วิริยะยุทธกร** — ผู้สร้าง OpenThaiGPT, CEO บริษัท iApp Technology
บทความฉบับเต็ม: [kobkrit.com](https://kobkrit.com)

---

## โครงของบทความตอนนี้

{outline}

---

## ก่อนเริ่ม

- โน้ตบุ๊กนี้ออกแบบมาให้รันได้บน **Google Colab แบบฟรี (Tesla T4)**
- T4 เป็นสถาปัตยกรรม Turing (sm_75) ซึ่ง **ไม่รองรับ bfloat16 และไม่รองรับ FlashAttention-2**
  เราจึงใช้ `torch_dtype=torch.float16`, `attn_implementation="sdpa"` และ `fp16=True` เสมอ
  (รายละเอียดอยู่ใน Cell 0 ด้านล่าง — อ่านคอมเมนต์ให้ครบ)
- ทุกตอนในซีรีส์นี้ใช้ชุดวัดผลกลางตัวเดียวกันคือ `kobeval` และเบนช์มาร์ก **KobEval-TH**
  (120 ข้อ, 4 slice) เพื่อให้ตัวเลขของแต่ละตอนเทียบกันได้จริง
"""

    section_2 = """---

## 2. ตั้งค่า และวัด baseline ก่อนลงมือทำอะไรทั้งสิ้น

หลักการของทั้งซีรีส์: **วัดก่อนเสมอ**

เซลล์ถัดไปคือ Cell 1 ซึ่งวัดผลโมเดลตั้งต้นด้วย KobEval-TH ก่อนที่เราจะเทรน
หรือแก้อะไรทั้งนั้น ถ้าไม่มีตัวเลขตั้งต้น เราจะไม่มีทางรู้เลยว่าสิ่งที่ทำในตอนนี้
ทำให้ดีขึ้นจริงหรือแค่รู้สึกว่าดีขึ้น

สังเกตค่า `th_ratio` เป็นพิเศษ — มันคือสัดส่วนตัวอักษรไทยในคำตอบ
จุดที่ Qwen3-0.6B พังบ่อยที่สุดกับคำถามภาษาไทย ไม่ใช่การตอบผิด
แต่คือการ **ตอบเป็นภาษาอังกฤษหรือภาษาจีนอย่างมั่นใจ** ซึ่ง accuracy อย่างเดียวมองไม่เห็น
"""

    cell1 = '''# =============================================================================
# CELL 1 -- BASELINE (วัดผลก่อนเทรน/ก่อนแก้อะไรทั้งสิ้น)
# เซลล์นี้เป็นเซลล์โค้ดที่สองของทุกตอนในซีรีส์
# =============================================================================
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,             # <-- ห้ามใช้ "auto" เด็ดขาดบน T4 (ดูคอมเมนต์ Cell 0)
    attn_implementation=ATTN_IMPL, # <-- "sdpa" ไม่ใช่ flash_attention_2
    device_map="cuda:0",
)
model.eval()

print(f"โหลดโมเดลแล้ว: {MODEL_ID}")
print(f"dtype จริงของ parameter: {next(model.parameters()).dtype}")
print(f"จำนวนพารามิเตอร์: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# รันเบนช์มาร์กกลาง -- สัญญาการวัดผลถูกตรึงไว้ใน kobeval แล้ว
# (greedy, max_new_tokens=256, enable_thinking=False, seed=42)
baseline = evaluate(
    model,
    tokenizer,
    slices=["TH-KNOW", "TH-MATH", "TH-INSTR", "TH-SAFE"],
    seed=42,
    model_name=f"{MODEL_ID} (baseline)",
    out_path="results_baseline.json",
)

for name, s in baseline["slices"].items():
    print(
        f"{name:9s} acc={s['accuracy']:.3f} "
        f"[95% CI {s['ci_low']:.3f}-{s['ci_high']:.3f}]  "
        f"th_ratio={s['th_ratio']:.2f}  len={s['mean_output_len']:.0f}"
    )

print(f"\\nรวมทุก slice: acc={baseline['overall']['accuracy']:.3f}  "
      f"th_ratio={baseline['overall']['th_ratio']:.2f}")
'''

    middle_sections = []
    for i in (3, 4, 5, 6, 7):
        middle_sections.append(md(f"---\n\n## {i}. {SECTIONS[i - 1]}\n\n_(เนื้อหาของตอนนี้)_\n"))
        middle_sections.append(code(f"# ตอนที่ {post} -- ส่วนที่ {i}\n"))

    results_md = """---

## 8. ตารางผลลัพธ์

ตารางมาตรฐานที่ใช้เหมือนกันทุกตอน สร้างจาก `compare()` ซึ่งอ่านค่าจริงจาก
`results.json` ไม่ใช่ตัวเลขที่พิมพ์เอง

แถบ error bar ในกราฟคือ **Wilson 95% confidence interval** ไม่ใช่ normal approximation
เหตุผลสำคัญ: แต่ละ slice มีแค่ 30 ข้อ ความต่างระหว่าง 40% กับ 47% ดูเหมือนดีขึ้น
จนกว่าจะวาดช่วงความเชื่อมั่นแล้วพบว่ามันซ้อนทับกันเกือบทั้งหมด
"""

    results_code = '''# เปรียบเทียบ baseline กับผลหลังปรับปรุง
# (ในตอนที่เป็น template นี้ เราใช้ baseline ซ้ำสองครั้งเพื่อสาธิตรูปแบบตาราง
#  ในตอนจริงให้แทน `after` ด้วยผลของโมเดลหลังเทรน)
after = baseline  # TODO: แทนที่ด้วย evaluate(...) ของโมเดลหลังปรับปรุง

table = compare(baseline, after, out_path="results.json")

plot_before_after(
    baseline,
    after,
    slices=["TH-KNOW", "TH-MATH", "TH-INSTR"],
    title=f"ตอนที่ {post}: ก่อน vs หลัง",
    out_path="before_after.png",
    results_json="results.json",
)
'''.replace("{post}", str(post))

    analysis_md = f"""---

## 9. {SECTIONS[8]}

_(อภิปรายผล: ตัวเลขไหนขยับจริง ตัวเลขไหนขยับแค่ในสายตา)_

ข้อควรระวังในการอ่านผล:

- ดูค่า p จาก McNemar ที่ `compare()` พิมพ์ออกมาด้วย ไม่ใช่ดูแค่ accuracy
  accuracy ที่ขยับจาก 12/30 เป็น 14/30 อาจซ่อนการที่โมเดล "แก้ถูก 8 ข้อ
  แต่ทำพังเพิ่ม 6 ข้อ" ซึ่งไม่ใช่การพัฒนา
- ถ้า `th_ratio` ตก แปลว่าโมเดลเริ่มตอบเป็นภาษาอื่นมากขึ้น แม้ accuracy จะขึ้นก็ตาม

---

## 10. {SECTIONS[9]}

_(สรุปสิ่งที่ได้เรียนรู้ และเกริ่นตอนต่อไป)_
"""

    limitations_md = """---

## ข้อจำกัดของการทดลองนี้

เขียนไว้ตรง ๆ เพื่อไม่ให้ตัวเลขข้างบนถูกอ่านเกินกว่าที่มันบอกได้จริง

1. **ขนาดชุดทดสอบเล็กมาก** — KobEval-TH มี slice ละ 30 ข้อเท่านั้น
   ช่วงความเชื่อมั่นจึงกว้าง ความต่างระดับ 1-2 ข้อ **ไม่ใช่** ความต่างที่มีนัยสำคัญ
   ตัวเลขในซีรีส์นี้ใช้เพื่อ "สอนวิธีวัด" ไม่ใช่เพื่อประกาศว่าโมเดลไหนดีที่สุด

2. **โมเดลเล็ก** — Qwen3-0.6B มีพารามิเตอร์เพียง 0.6B ข้อสรุปหลายอย่างที่ได้จาก
   โมเดลขนาดนี้ อาจไม่เป็นจริงกับโมเดลขนาด 7B ขึ้นไป

3. **รันครั้งเดียว ไม่ได้ทำ multiple seeds** — เราใช้ greedy decoding และ seed=42
   ตายตัวเพื่อให้ทำซ้ำได้ แต่ไม่ได้รายงานความแปรปรวนจากการเทรนหลาย seed
   ซึ่งในงานวิจัยจริงจำเป็นต้องทำ

4. **การให้คะแนนเป็นแบบอัตโนมัติทั้งหมด** — TH-KNOW ใช้ exact match,
   TH-INSTR ใช้ rubric ตายตัว, TH-SAFE ใช้การตรวจจับการปฏิเสธด้วย keyword
   วิธีพวกนี้เร็วและทำซ้ำได้ แต่หยาบกว่าการให้มนุษย์ตรวจ และมีทั้ง
   false positive และ false negative แน่นอน

5. **ฮาร์ดแวร์จำกัด** — ทุกอย่างถูกบีบให้รันได้บน T4 ฟรี ซึ่งแปลว่าต้องใช้
   fp16 แทน bf16, ใช้ sdpa แทน FlashAttention-2, batch size เล็ก และ
   sequence length สั้น ผลลัพธ์บนฮาร์ดแวร์ที่ใหญ่กว่าอาจต่างออกไป

6. **ยังไม่ได้ตรวจการปนเปื้อนของข้อมูล (contamination)** — ข้อสอบ TH-MATH
   เขียนขึ้นใหม่เองทั้งหมดเพื่อลดความเสี่ยงนี้ แต่ TH-KNOW เป็นความรู้ทั่วไป
   ที่อาจอยู่ในข้อมูลเทรนของโมเดลอยู่แล้ว

---

*ซีรีส์นี้เผยแพร่ภายใต้สัญญาอนุญาต [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — ใช้ต่อได้โดยอ้างอิงที่มา ห้ามใช้เชิงพาณิชย์ และต้องเผยแพร่ต่อด้วยสัญญาเดียวกัน — [kobkrit.com](https://kobkrit.com)*
"""

    cells = [
        md(header),
        code(cell0),
        md(section_2),
        code(cell1),
        *middle_sections,
        md(results_md),
        code(results_code),
        md(analysis_md),
        md(limitations_md),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True, help="filename stem, e.g. 01-first-thai-eval")
    parser.add_argument("--post", type=int, required=True, help="post number 1-10 (0 = template)")
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", default="_(คำโปรยของตอนนี้)_")
    args = parser.parse_args()

    nb = build(args.slug, args.post, args.title, args.subtitle)
    out = NOTEBOOKS / f"{args.slug}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
