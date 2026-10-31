#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""归因去重的回归测试。

为什么单独写这个：去重的判据是「证据集包含 + 有没有达标记录」两件事的组合，
边界情形靠肉眼是看不全的。这套用例把每条判据钉成断言，改判据时先跑一遍。

    python test_dedup.py
"""

import json

from agent import dedup_patterns, enrich_patterns


def scores(*totals):
    """C01..Cn 依次给分。"""
    return [{"case": "C%02d" % i, "总分": v} for i, v in enumerate(totals, 1)]


def dedup(pats, sc):
    out = dedup_patterns({"模式": json.loads(json.dumps(pats))}, sc)
    return [p["名称"] for p in out["模式"]], out.get("合并", []), out


def test_subset_all_low():
    """真子集 + 子集内无达标记录 → 合并。"""
    names, merged, out = dedup([
        {"名称": "A", "证据": ["C01", "C02", "C03"], "建议": "a"},
        {"名称": "B", "证据": ["C01", "C02"], "建议": "b"},
    ], scores(2, 2, 2))
    assert names == ["A"], names
    assert merged == [{"子": "B", "父": "A"}], merged
    assert out["模式"][0]["补充"][0]["来源"] == "B"


def test_subset_has_pass():
    """真子集 + 子集内有达标记录 → 保留，那是「策略部分生效」的另一件事。"""
    names, merged, _ = dedup([
        {"名称": "A", "证据": ["C01", "C02", "C03"], "建议": "a"},
        {"名称": "B", "证据": ["C01", "C02"], "建议": "b"},
    ], scores(6, 2, 2))
    assert names == ["A", "B"], names
    assert merged == [], merged


def test_identical_has_pass():
    """证据完全相同 + 含达标记录 → 仍然合并。

    同一批记录说两遍，跟有没有高分无关；这里若套用「部分生效」的判据，
    会把重复归因当成新发现留下来。第一版就漏在这。
    """
    names, merged, _ = dedup([
        {"名称": "A", "证据": ["C01", "C02"], "建议": "a"},
        {"名称": "B", "证据": ["C01", "C02"], "建议": "b"},
    ], scores(6, 2))
    assert names == ["A"], names
    assert merged == [{"子": "B", "父": "A"}], merged


def test_identical_all_low():
    names, merged, _ = dedup([
        {"名称": "A", "证据": ["C01", "C02"], "建议": "a"},
        {"名称": "B", "证据": ["C01", "C02"], "建议": "b"},
    ], scores(2, 2))
    assert names == ["A"], names


def test_empty_evidence():
    """空证据不参与比较，也不能让函数炸掉。"""
    names, merged, _ = dedup([
        {"名称": "A", "证据": []},
        {"名称": "B", "证据": ["C01"]},
    ], scores(2))
    assert sorted(names) == ["A", "B"], names
    assert merged == [], merged


def test_disjoint():
    """证据互不相交 → 一条都不并。保险那轮就是这种形状。"""
    names, merged, _ = dedup([
        {"名称": "A", "证据": ["C01", "C02"]},
        {"名称": "B", "证据": ["C03", "C04"]},
    ], scores(2, 2, 2, 2))
    assert names == ["A", "B"], names
    assert merged == [], merged


def test_chain():
    """链式 A⊃B⊃C：B、C 都并进 A，不能出现 C 并进已消失的 B。"""
    names, merged, out = dedup([
        {"名称": "A", "证据": ["C01", "C02", "C03"]},
        {"名称": "B", "证据": ["C01", "C02"]},
        {"名称": "C", "证据": ["C01"]},
    ], scores(2, 2, 2))
    assert names == ["A"], names
    assert [m["子"] for m in merged] == ["B", "C"], merged
    assert [c["来源"] for c in out["模式"][0]["补充"]] == ["B", "C"]


def test_child_first():
    """输入顺序颠倒也要并，不能靠「父在前」这个巧合。"""
    names, merged, _ = dedup([
        {"名称": "B", "证据": ["C01"]},
        {"名称": "A", "证据": ["C01", "C02", "C03"]},
    ], scores(2, 2, 2))
    assert names == ["A"], names
    assert merged == [{"子": "B", "父": "A"}], merged


def test_partial_overlap():
    """部分重叠但不包含 → 不并。两个不同的问题模式。"""
    names, merged, _ = dedup([
        {"名称": "A", "证据": ["C01", "C02", "C03"]},
        {"名称": "B", "证据": ["C02", "C03", "C04"]},
    ], scores(2, 2, 2, 2))
    assert names == ["A", "B"], names


def test_keep_model_order():
    """合并后输出必须保持模型给的严重度序，不能按证据数重排。

    提示词第 4 条要求模型「按严重度排序，影响面大的在前」。这里为了判定
    包含关系，遍历顺序是按证据数降序的（证据多的先入列，后面的才判得了
    包含），但那个顺序只该用于判断，不该漏进报告 —— 一旦漏了，标题写着
    「按严重度排序」而实际按覆盖数排，就是报告自己在说假话。

    构造：模型认为最严重的是只有 1 条证据的 Y；A 有 5 条但排在第二；
    X 是 A 的子集。若按证据数输出会变成 [A, Y]，正确的是 [Y, A]。
    """
    names, merged, _ = dedup([
        {"名称": "Y", "证据": ["C06"]},
        {"名称": "A", "证据": ["C01", "C02", "C03", "C04", "C05"]},
        {"名称": "X", "证据": ["C01", "C02"]},
    ], scores(2, 2, 2, 2, 2, 2))
    assert names == ["Y", "A"], names
    assert merged == [{"子": "X", "父": "A"}], merged


def test_idempotent():
    """对已去重的结果再跑一次，不应继续减少。"""
    pats = {"模式": [
        {"名称": "A", "证据": ["C01", "C02", "C03"], "建议": "a"},
        {"名称": "B", "证据": ["C01", "C02"], "建议": "b"},
    ]}
    once = dedup_patterns(json.loads(json.dumps(pats)), scores(2, 2, 2))
    twice = dedup_patterns(json.loads(json.dumps(once)), scores(2, 2, 2))
    assert len(twice["模式"]) == len(once["模式"]) == 1, (once, twice)


def test_enrich_counts():
    """enrich 的用例数 / 格数 / 记录数必须是算出来的，不是模型写的。"""
    pats = {"模式": [{"名称": "A", "证据": ["C01", "C02", "C99"],
                      "出现次数": 99, "建议": "a"}]}
    out = enrich_patterns(pats, [
        {"case": "C01", "总分": 2}, {"case": "C02", "总分": 6},
        {"case": "C02", "总分": 3},
    ])
    p = out["模式"][0]
    assert p["证据"] == ["C01", "C02"], p["证据"]   # C99 是编的，要剔掉
    assert p["用例数"] == 2
    assert p["格数"] == 3                            # C02 出现两次
    assert p["记录数"] == 2                          # ≤4 分的两条
    assert "出现次数" not in p


def test_enrich_drops_fabricated():
    """证据全是编的 → 整条不留。"""
    out = enrich_patterns({"模式": [{"名称": "A", "证据": ["C77"]}]},
                          [{"case": "C01", "总分": 2}])
    assert out["模式"] == [], out


if __name__ == "__main__":
    cases = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for n, f in cases:
        f()
        print("  ok  %s" % n)
    print("\n%d 项通过" % len(cases))
