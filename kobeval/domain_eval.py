"""Build a domain-matched, held-out evaluation from your own corpus.

Why this module exists
----------------------
Notebook 01 continues pretraining on Thai government documents and then scored
the result on TH-KNOW -- general Thai facts like "what is the capital". That
measures the wrong thing: TH-KNOW asks about knowledge the corpus never taught,
so the number sits still no matter how well CPT worked, while domain perplexity
drops tenfold.

The rule this teaches: **evaluate on the distribution you trained on.** The
cloze items below are generated from documents deliberately held out of
training, so the eval is domain-matched by construction and leak-free by split.
"""
from __future__ import annotations

import random
import re
from typing import Iterable, List

__all__ = ["build_domain_cloze", "MASK"]

MASK = "____"

# Thai writes without spaces, so a pattern like r"กระทรวง[ก-๙]{2,20}" runs past
# the end of the entity and slices a word in half -- it produced answers such as
# "จังหวัดเชียงรายเป็นเจ้าภาพจ". Rather than guess word boundaries (which needs a
# Thai segmenter), match a curated list of real entity names literally,
# longest-first. Deterministic, and every answer is a complete, checkable name.
_MINISTRIES = [
    "กระทรวงการพัฒนาสังคมและความมั่นคงของมนุษย์",
    "กระทรวงทรัพยากรธรรมชาติและสิ่งแวดล้อม",
    "กระทรวงดิจิทัลเพื่อเศรษฐกิจและสังคม",
    "กระทรวงการท่องเที่ยวและกีฬา",
    "กระทรวงเกษตรและสหกรณ์",
    "กระทรวงการต่างประเทศ",
    "กระทรวงศึกษาธิการ",
    "กระทรวงอุตสาหกรรม",
    "กระทรวงสาธารณสุข",
    "สำนักนายกรัฐมนตรี",
    "กระทรวงยุติธรรม",
    "กระทรวงมหาดไทย",
    "กระทรวงคมนาคม",
    "กระทรวงวัฒนธรรม",
    "กระทรวงพาณิชย์",
    "กระทรวงพลังงาน",
    "กระทรวงการคลัง",
    "กระทรวงกลาโหม",
    "กระทรวงแรงงาน",
]
_PATTERNS = [re.escape(m) for m in sorted(_MINISTRIES, key=len, reverse=True)]
_PATTERNS.append(r"พ\.ศ\.\s?\d{4}")
_SENT_SPLIT = re.compile(r"(?<=[\s])(?=[ก-๙])|\n+")


def _sentences(text: str, min_len: int, max_len: int) -> List[str]:
    parts = [p.strip() for p in re.split(r"[\n]+|(?<=\.)\s+", text) if p.strip()]
    return [p for p in parts if min_len <= len(p) <= max_len]


def build_domain_cloze(
    texts: Iterable[str],
    n: int = 30,
    seed: int = 42,
    min_len: int = 60,
    max_len: int = 300,
) -> List[dict]:
    """Return up to ``n`` cloze items shaped like KobEval-TH entries.

    Each item masks one salient span and asks the model to restore it:

        {"id", "prompt", "answers": [span], "slice": "TH-DOMAIN", "source_len"}

    Pass ONLY held-out documents. Passing training documents turns this into a
    memorisation check and the resulting number means nothing.
    """
    rng = random.Random(seed)
    items: List[dict] = []
    seen_answers: set[str] = set()

    for text in texts:
        if len(items) >= n:
            break
        if not isinstance(text, str):
            continue
        for sent in _sentences(text, min_len, max_len):
            if len(items) >= n:
                break
            hits = []
            for pat in _PATTERNS:
                hits.extend(m for m in re.finditer(pat, sent))
            if not hits:
                continue
            m = rng.choice(hits)
            answer = m.group(0).strip()
            # Skip spans that appear more than once: the answer would still be
            # visible elsewhere in the same prompt.
            if sent.count(answer) != 1 or answer in seen_answers:
                continue
            seen_answers.add(answer)
            masked = sent[: m.start()] + MASK + sent[m.end():]
            items.append({
                "id": f"TH-DOMAIN-{len(items) + 1:03d}",
                "prompt": f"เติมข้อความในช่องว่างให้ถูกต้อง ตอบเฉพาะข้อความที่หายไป\n\n{masked}",
                "answers": [answer],
                "slice": "TH-DOMAIN",
                "source_len": len(sent),
            })
    return items
