# สอนสร้างและปรับแต่ง LLM ภาษาไทย — โน้ตบุ๊กประกอบซีรีส์ 10 ตอน

**Thai LLM Tutorials — companion notebooks for the 10-part series on [kobkrit.com](https://kobkrit.com)**

โดย **ดร.กอบกฤตย์ วิริยะยุทธกร** — ผู้สร้าง [OpenThaiGPT](https://openthaigpt.aieat.or.th/), CEO [iApp Technology](https://iapp.co.th)

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/KobEval--TH-v1%20·%20120%20items-green.svg)](data/README.md)

---

## ซีรีส์นี้คืออะไร

ซีรีส์นี้พาคุณสร้าง ปรับแต่ง และ **วัดผล** โมเดลภาษาขนาดเล็กสำหรับภาษาไทย
ตั้งแต่ศูนย์ โดยใช้ `Qwen/Qwen3-0.6B` เป็นโมเดลหลัก และรันได้จริงบน
**Google Colab แบบฟรี (Tesla T4)** ทุกตอน

สิ่งที่ทำให้ซีรีส์นี้ต่างจากบทความ fine-tuning ทั่วไป: **ทุกตอนวัดผลด้วยชุดวัดเดียวกัน**
เราไม่ได้บอกว่า "โมเดลดีขึ้น" เราบอกว่าดีขึ้นกี่เปอร์เซ็นต์ ช่วงความเชื่อมั่นเท่าไร
และความต่างนั้นมีนัยสำคัญทางสถิติหรือไม่

*This series builds, fine-tunes and — above all — **measures** a small Thai
language model, end to end, on a free Colab T4. Every post runs the same
benchmark, so the numbers across all ten posts are directly comparable.*

---

## สารบัญ 10 ตอน

ทุกตอนเผยแพร่แล้วทั้งบทความและโน้ตบุ๊ก — กดปุ่ม Colab แล้วรันได้ทันทีบน T4 ฟรี

| # | ตอน | สิ่งที่วัด | อ่านบทความ | Colab |
|---|---|---|---|---|
| 1 | **Continue Pretraining** — สอนความรู้ใหม่ให้ LLM ภาษาไทย | domain/general held-out PPL, TH-KNOW, tokens/char | [kobkrit.com](https://kobkrit.com/blog/llm-01-continue-pretraining) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kobkrit/thai-llm-tutorials/blob/main/notebooks/01_continue_pretraining.ipynb) |
| 2 | **SFT + LoRA** — สอนโมเดลให้เป็นผู้ช่วย ด้วยการเทรน 1.69% ของพารามิเตอร์ | TH-INSTR + Wilson CI, `th_ratio`, สัดส่วนจบด้วย eos, ablation ไม่ mask | [kobkrit.com](https://kobkrit.com/blog/llm-02-sft-lora) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kobkrit/thai-llm-tutorials/blob/main/notebooks/02_sft_lora.ipynb) |
| 3 | **RLHF และ PPO** — เทรนโมเดลด้วยรางวัลที่หาอนุพันธ์ไม่ได้ | RM pairwise acc + Wilson CI, reward/KL/ความยาว, TH-MATH, ablation β=0 | [kobkrit.com](https://kobkrit.com/blog/llm-03-rlhf-ppo) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kobkrit/thai-llm-tutorials/blob/main/notebooks/03_rlhf_ppo.ipynb) |
| 4 | **DPO** — เมื่อโมเดลภาษากลายเป็น reward model ของตัวเอง | preference acc + Wilson CI, reward margin, `th_ratio`, TH-INSTR | [kobkrit.com](https://kobkrit.com/blog/llm-04-dpo) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kobkrit/thai-llm-tutorials/blob/main/notebooks/04_dpo.ipynb) |
| 5 | **GRPO** — ลบ value network ทิ้ง ให้กลุ่มคำตอบเป็น baseline ของกันเอง | pass@1/pass@8 (unbiased) + Wilson CI, mean reward, สัดส่วนกลุ่ม std > 0, TH-MATH, ดักจับ reward hacking | [kobkrit.com](https://kobkrit.com/blog/llm-05-grpo) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kobkrit/thai-llm-tutorials/blob/main/notebooks/05_grpo.ipynb) |
| 6 | **Context Distillation (OPCD)** — ย้าย system prompt ~400 token เข้า weights ด้วย reverse KL บน top-128 | persona-compliance 4 เงื่อนไข (รวม offline CD) + Wilson CI, OOD compliance, coverage, entropy/length canaries, prompt tokens/latency | [kobkrit.com](https://kobkrit.com/blog/llm-06-context-distillation) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kobkrit/thai-llm-tutorials/blob/main/notebooks/06_context_distillation.ipynb) |
| 7 | **Model Distillation** — กลั่น Qwen3-1.7B ลงสู่ 0.6B ด้วย top-64 logits | TH-INSTR ครู/นักเรียน + gap closed %, พิสูจน์ KD loss กับ float64, T² sweep, tok/sec, TH-SAFE | [kobkrit.com](https://kobkrit.com/blog/llm-07-model-distillation) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kobkrit/thai-llm-tutorials/blob/main/notebooks/07_model_distillation.ipynb) |
| 8 | **Guardrails** — classifier ขาเข้า + ตัวกรอง PII ขาออก และการเลือก τ* ด้วยต้นทุน | ROC-AUC + bootstrap CI, τ* ที่ 10:1, unsafe/benign-blocked บน TH-SAFE + Wilson CI, mod-11 checksum | [kobkrit.com](https://kobkrit.com/blog/llm-08-guardrails) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kobkrit/thai-llm-tutorials/blob/main/notebooks/08_guardrails.ipynb) |
| 9 | **Benchmarking** — ระบบให้คะแนน 3 โหมดจากศูนย์ (loglik γ / exact-match / judge) | ThaiExam + GSM8K-TH + KobEval-TH ครบ 4 slice, Wilson CI ทุกตัวเลข, McNemar คู่สูสี, pass@k, contamination scan k=5/20, ด่านตรวจ license | [kobkrit.com](https://kobkrit.com/blog/llm-09-benchmarking) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kobkrit/thai-llm-tutorials/blob/main/notebooks/09_benchmarking.ipynb) |
| 10 | **Deployment** — เพดาน ~266 tok/s จาก datasheet แล้ววัดจริงทีละชั้น | KV 112 KiB/token (assert จาก config), tok/s @B=1/16, continuous batching ~60 บรรทัด, p50/p99 vs λ แบบ open-loop, int8/nf4 พร้อมคอลัมน์ TH-KNOW บังคับ | [kobkrit.com](https://kobkrit.com/blog/llm-10-deployment) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kobkrit/thai-llm-tutorials/blob/main/notebooks/10_deployment.ipynb) |
| — | แม่แบบ (Template) — โครงมาตรฐานของทุกตอน | — | — | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kobkrit/thai-llm-tutorials/blob/main/notebooks/_template.ipynb) |

> โน้ตบุ๊กทุกตอนใช้ **Cell 0 ที่เหมือนกันทุกไบต์** และวัดผลด้วย `kobeval` + ชุด KobEval-TH ชุดเดียวกัน
> ตัวเลขข้ามตอนจึงเทียบกันได้ตรง ๆ

---

## แกนกลางร่วมของทุกตอน (the shared spine)

ทุกตอนใช้โครงเดียวกันเป๊ะ ๆ ซึ่งเป็นเหตุผลที่ตัวเลขข้ามตอนเทียบกันได้:

### 1. `kobeval` — ชุดวัดผลกลาง

```python
from kobeval import evaluate, compare, plot_before_after, th_ratio, wilson_ci

report = evaluate(model, tokenizer, slices=["TH-KNOW", "TH-MATH"], seed=42)
```

`evaluate()` คืนค่า accuracy, **Wilson 95% CI**, mean perplexity,
ความยาวคำตอบเฉลี่ย และ `th_ratio` แยกตาม slice พร้อมเขียน `results.json`
ให้วิดเจ็ตบนเว็บอ่านตัวเลขจริงไปแสดง

### 2. `th_ratio` — ตัววัดประจำซีรีส์

สัดส่วนตัวอักษรที่ไม่ใช่ช่องว่างซึ่งอยู่ในบล็อกยูนิโค้ดภาษาไทย `U+0E00–U+0E7F`

ทำไมต้องมีตัวนี้: จุดที่ Qwen3-0.6B พังบ่อยที่สุดกับคำถามภาษาไทย **ไม่ใช่การตอบผิด**
แต่คือการ *ตอบเป็นภาษาอังกฤษหรือภาษาจีนอย่างมั่นใจ* ซึ่ง accuracy อย่างเดียวมองไม่เห็นเลย
`th_ratio` เปลี่ยนความล้มเหลวแบบนั้นให้กลายเป็นตัวเลขที่พล็อตได้

```python
th_ratio("เมืองหลวงของไทยคือกรุงเทพมหานคร")  # 1.0
th_ratio("The capital of Thailand is Bangkok.")  # 0.0
th_ratio("泰国的首都是曼谷。")                      # 0.0
```

### 3. KobEval-TH — เบนช์มาร์ก 120 ข้อ

4 slice × 30 ข้อ: **TH-KNOW** (ความรู้ไทย), **TH-MATH** (โจทย์เลข),
**TH-INSTR** (ทำตามคำสั่ง), **TH-SAFE** (ความปลอดภัย)

รายละเอียดทั้งหมดอยู่ที่ [`data/README.md`](data/README.md) — รวมถึงกฎเหล็ก **"ห้ามนำไปเทรน"**

### 4. โครงบทความ 10 หัวข้อ + Cell 0 / Cell 1 ที่ตายตัว

- **Cell 0** (เซลล์โค้ดแรก) — เหมือนกันทุกตัวอักษรในทั้ง 10 ตอน:
  ตรวจ GPU, พิมพ์ `SUPPORTS_BF16`, ติดตั้ง dependency แบบ pin, ตั้ง seed, import `kobeval`
  ตรวจสอบด้วย `python3 scripts/check_cell0.py`
- **Cell 1** (เซลล์โค้ดที่สอง) — **วัด baseline ก่อนเสมอ** ทุกตอนจึงเริ่มด้วยการวัด
  สิ่งที่ตัวเองกำลังจะไปปรับปรุง
- ปิดท้ายทุกตอนด้วยหัวข้อ **"ข้อจำกัดของการทดลองนี้"**

---

## สัญญาการวัดผล (the evaluation contract)

ค่าเหล่านี้ถูก **ตรึงไว้ในโค้ด** (`kobeval.runner.EVAL_CONTRACT`) ไม่ใช่ตั้งใหม่ในแต่ละตอน
เพราะเบนช์มาร์กที่พารามิเตอร์การถอดรหัสเปลี่ยนไปมาระหว่างตอน ย่อมวัดอะไรไม่ได้เลย

| พารามิเตอร์ | ค่า | เหตุผล |
|---|---|---|
| `do_sample` | `False` (greedy) | ทำซ้ำได้ ไม่มีความสุ่ม |
| `max_new_tokens` | `256` | พอสำหรับคำตอบไทยที่มีเหตุผลประกอบ |
| `enable_thinking` | `False` | เปิดได้เมื่อจงใจศึกษาผลของ thinking mode |
| `torch.manual_seed` | `42` **ก่อนทุกข้อ** | ผลของข้อหนึ่งไม่ขึ้นกับจำนวนข้อที่รันมาก่อน |
| ช่วงความเชื่อมั่น | **Wilson 95%** | ไม่ใช้ normal approximation — ที่ n=30 มันให้ช่วงที่หลุดออกนอก [0,1] และกว้างเป็นศูนย์เมื่อ p=0 |
| การทดสอบนัยสำคัญ | **McNemar** (มี continuity correction) | accuracy 12/30 → 14/30 อาจซ่อนการ "แก้ถูก 8 พังเพิ่ม 6" |

ตารางผลลัพธ์มาตรฐานในหัวข้อที่ 8 ของทุกตอน:

| Model | TH-KNOW | TH-MATH | TH-INSTR | th_ratio | PPL | VRAM peak | Train time |
|---|---|---|---|---|---|---|---|

---

## ความต้องการด้านฮาร์ดแวร์

ออกแบบมาให้รันได้บน **Google Colab ฟรี — Tesla T4, 16 GB VRAM**

### ข้อจำกัดของ T4 ที่กำหนดรูปร่างของทั้งซีรีส์

T4 เป็นสถาปัตยกรรม **Turing (sm_75)** ซึ่ง **ไม่รองรับ**:

- ❌ **bfloat16** → ต้องใช้ `torch_dtype=torch.float16` และ `fp16=True`
- ❌ **FlashAttention-2** → ต้องใช้ `attn_implementation="sdpa"`

> ⚠️ **กับดักที่สำคัญที่สุด**
>
> ไฟล์ config ของ `Qwen/Qwen3-0.6B` ระบุ `torch_dtype: bfloat16` เอาไว้
> ดังนั้นถ้าคุณเขียน `torch_dtype="auto"` transformers จะเชื่อ config
> แล้วโหลดโมเดลเป็น bf16 **บนการ์ดที่ไม่รองรับ bf16**
>
> ผลคือ NaN, loss ไม่ลด หรือโมเดลพ่นข้อความมั่ว — **โดยไม่มี error ฟ้องให้เห็น**
>
> ทุกโน้ตบุ๊กในซีรีส์นี้จึงกำหนด dtype เองอย่างชัดเจนเสมอ ไม่ใช้ `"auto"` เด็ดขาด
> และมีเทสต์ (`tests/test_notebooks.py`) คอยกันไม่ให้ `"auto"`, `bf16=True`
> หรือ `flash_attention_2` หลุดเข้ามาในโค้ดเซลล์ใดก็ตาม

Cell 0 จะพิมพ์บรรทัดนี้ให้เห็นกับตาทุกครั้ง — บน T4 มันจะขึ้นว่า `False`:

```
SUPPORTS_BF16  : False
```

### โมเดลและชุดข้อมูลที่ใช้

**โมเดล:** `Qwen/Qwen3-0.6B` · `Qwen/Qwen3-0.6B-Base` · `Qwen/Qwen3-1.7B`

Qwen3-0.6B: `hidden_size=1024`, `intermediate_size=3072`, `num_hidden_layers=28`,
`num_attention_heads=16`, `num_key_value_heads=8` (GQA), `head_dim=128`,
`vocab_size=151936`, `max_position_embeddings=40960`

**ชุดข้อมูลภาษาไทยบน Hugging Face ที่ซีรีส์นี้ใช้:**

| Dataset | ใช้ในตอน |
|---|---|
| [`airesearch/wangchanx-seed-free-synthetic-instruct-thai-120k`](https://huggingface.co/datasets/airesearch/wangchanx-seed-free-synthetic-instruct-thai-120k) | 1 (replay), 2 (SFT), 4 (DPO), 6 (OPCD), 7 (distillation) |
| [`pythainlp/thaigov-v2-corpus-22032023`](https://huggingface.co/datasets/pythainlp/thaigov-v2-corpus-22032023) | 1 (โดเมนสำหรับ CPT), 9 (contamination scan) |
| [`iapp/dpo_thai_tutorial`](https://huggingface.co/datasets/iapp/dpo_thai_tutorial) | 3 (reward model), 4 (DPO) |
| [`VISAI-AI/gsm8k-thai`](https://huggingface.co/datasets/VISAI-AI/gsm8k-thai) | 3 (PPO), 5 (GRPO), 9 (วัดผล) |
| [`tmu-nlp/thai_toxicity_tweet`](https://huggingface.co/datasets/tmu-nlp/thai_toxicity_tweet) | 8 (guardrails — คลาส unsafe) |
| [`pythainlp/wisesight_sentiment`](https://huggingface.co/datasets/pythainlp/wisesight_sentiment) | 8 (guardrails — hard negative) |
| [`scb10x/thai_exam`](https://huggingface.co/datasets/scb10x/thai_exam) | 9 (benchmark ภายนอก) |
| **KobEval-TH** (ในรีโพนี้) | ทุกตอน — ห้ามนำไปเทรนเด็ดขาด |

---

## เริ่มใช้งาน

### บน Colab (แนะนำ)

กดปุ่ม Colab ของตอนที่ต้องการ แล้วรัน Cell 0 — มันจะจัดการติดตั้งทุกอย่างให้เอง

### บนเครื่องตัวเอง

```bash
git clone https://github.com/kobkrit/thai-llm-tutorials.git
cd thai-llm-tutorials

# ติดตั้ง torch ให้ตรงกับ CUDA ของคุณก่อน (ดู https://pytorch.org)
pip install -r requirements.txt
pip install -e .
```

### ใช้ `kobeval` อย่างเดียว (ไม่ต้องมี GPU)

ชั้นสถิติและ metric เป็น Python ล้วน ไม่พึ่ง torch จึงติดตั้งและใช้บนเครื่อง CPU ได้:

```bash
pip install -e .
python3 -c "
from kobeval import th_ratio, wilson_ci, pass_at_k, mcnemar
print(th_ratio('สวัสดีครับ'))        # 1.0
print(wilson_ci(50, 100))            # (0.4038, 0.5962)
print(pass_at_k(10, 3, 5))           # 0.9167
print(mcnemar(2, 10)['p_value'])     # 0.0433
"
```

---

## โครงสร้างรีโพ

```
thai-llm-tutorials/
├── kobeval/                  # แพ็กเกจวัดผลกลาง (pip install -e .)
│   ├── stats.py              # wilson_ci, mcnemar, pass_at_k
│   ├── metrics.py            # th_ratio, การ normalize, grader ของแต่ละ slice
│   ├── data.py               # โหลดข้อมูล + ตรวจ checksum
│   ├── runner.py             # evaluate(), compare(), EVAL_CONTRACT
│   └── plotting.py           # plot_before_after() + ฟอนต์ไทย
├── data/
│   ├── README.md             # ที่มา สัญญาอนุญาต กฎห้ามเทรน checksum
│   └── kobeval_th/           # เบนช์มาร์ก 120 ข้อ (JSONL) + checksums.json
├── notebooks/
│   ├── cell0_setup.ipy       # Cell 0 ฉบับต้นแบบ (ต้องเหมือนกันทุกตอน)
│   └── _template.ipynb       # โน้ตบุ๊กแม่แบบ
├── scripts/
│   ├── make_notebook.py      # สร้างโน้ตบุ๊กตอนใหม่จากแม่แบบ
│   ├── check_cell0.py        # ตรวจว่า Cell 0 เหมือนกันทุกไฟล์
│   ├── verify_math.py        # แก้โจทย์ TH-MATH ซ้ำอย่างเป็นอิสระ
│   └── make_checksums.py     # สร้าง checksums.json ใหม่ (ใช้ตอนขึ้นเวอร์ชัน)
├── tests/                    # 105 เทสต์ รันได้บน CPU ล้วน
├── requirements.txt
└── pyproject.toml
```

## การตรวจสอบ (CI)

```bash
pytest tests/ -q                    # 105 tests
python3 scripts/verify_math.py      # แก้โจทย์เลขทั้ง 30 ข้อซ้ำ
python3 scripts/check_cell0.py      # Cell 0 ตรงกันทุกโน้ตบุ๊ก
```

---

## สัญญาอนุญาต

โค้ด โน้ตบุ๊ก และชุดข้อมูล KobEval-TH เผยแพร่ภายใต้ **Apache-2.0** — ดู [LICENSE](LICENSE)

โมเดลและชุดข้อมูลจากภายนอกอยู่ภายใต้สัญญาอนุญาตของเจ้าของแต่ละราย
โปรดตรวจสอบก่อนนำไปใช้ในเชิงพาณิชย์

## ติดต่อ

- เว็บไซต์: [kobkrit.com](https://kobkrit.com)
- บริษัท: [iApp Technology](https://iapp.co.th)
- ปัญหา/ข้อเสนอแนะ: เปิด [issue](https://github.com/kobkrit/thai-llm-tutorials/issues) ได้เลย

---

*ถ้าซีรีส์นี้มีประโยชน์กับคุณ ฝากกดดาวให้รีโพนี้ด้วยครับ 🙏*
