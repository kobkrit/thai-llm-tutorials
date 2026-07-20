"""Tests for the domain-matched cloze builder."""
from kobeval.domain_eval import build_domain_cloze, MASK

DOCS = [
    "กระทรวงมหาดไทยได้ออกประกาศเรื่องการจัดสรรงบประมาณให้แก่องค์กรปกครองส่วนท้องถิ่นทั่วประเทศในปีนี้",
    "สำนักนายกรัฐมนตรีได้พิจารณาร่างกฎหมายฉบับนี้เสร็จเรียบร้อยแล้วและส่งกลับไปยังหน่วยงานที่เกี่ยวข้องทุกแห่ง",
    "การประชุมคณะรัฐมนตรีเมื่อ พ.ศ. 2567 มีมติเห็นชอบแผนพัฒนาโครงสร้างพื้นฐานด้านคมนาคมของประเทศ",
]


def test_builds_items_with_masked_answer_removed():
    items = build_domain_cloze(DOCS, n=5)
    assert items, "should build at least one item"
    for it in items:
        assert MASK in it["prompt"]
        ans = it["answers"][0]
        assert ans not in it["prompt"], "the answer must not remain visible in the prompt"
        assert it["slice"] == "TH-DOMAIN"


def test_is_deterministic_for_a_given_seed():
    a = build_domain_cloze(DOCS, n=5, seed=7)
    b = build_domain_cloze(DOCS, n=5, seed=7)
    assert [x["prompt"] for x in a] == [x["prompt"] for x in b]


def test_respects_n_and_does_not_repeat_answers():
    items = build_domain_cloze(DOCS * 20, n=3)
    assert len(items) <= 3
    answers = [i["answers"][0] for i in items]
    assert len(answers) == len(set(answers)), "answers must be unique"


def test_empty_and_junk_input_is_safe():
    assert build_domain_cloze([]) == []
    assert build_domain_cloze(["", None, "short"]) == []


def test_answers_are_complete_entity_names_not_word_fragments():
    """Thai has no spaces; a regex like [ก-๙]{2,20} slices words in half.

    That produced answers such as "จังหวัดเชียงรายเป็นเจ้าภาพจ" from the real
    corpus. Matching a curated list literally is what prevents it.
    """
    from kobeval.domain_eval import _MINISTRIES
    docs = ["นายกรัฐมนตรีมอบหมายให้กระทรวงยุติธรรมเป็นเจ้าภาพจัดการประชุมร่วมกับหน่วยงานที่เกี่ยวข้องในสัปดาห์หน้า"]
    items = build_domain_cloze(docs, n=3)
    for it in items:
        ans = it["answers"][0]
        assert ans in _MINISTRIES or ans.startswith("พ.ศ."), f"not a complete entity: {ans}"
