"""claim-tracer 的测试。

重点不在"正常路径"，而在**我自己造出来的三个 bug**：
每一个都曾让工具给出看起来正常的错误结论。

1. **"零发现"被当成"通过"** —— 论文里一个数值 token 都没找到时，
   早期版本因"0 个未溯源"返回 0（成功）。实测踩到：数字都在
   ``\\input`` 进来的表格里而工具没读到，于是它报了一个绿色结果。
2. **``\\input`` 解析基准错了** —— LaTeX 相对**主文件目录**解析，
   我按"包含它的那个文件"的目录解析，于是 ``sections/results.tex`` 里的
   ``\\input{figures/evaluation_table}`` 找的是 ``sections/figures/``，
   整篇的数字一个都没读到。
3. **孤儿搜索范围太窄** —— 只扫 ``paper/``，而结果表在兄弟目录
   ``figures/`` 下，**恰好漏掉最该报的那个东西**。

运行：``python -m pytest`` 或 ``python tests/test_claim_tracer.py``
"""

from __future__ import annotations

import itertools
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim_tracer.cli import (  # noqa: E402
    analyze,
    derived_candidates,
    expand_paper,
    extract_numbers,
    find_orphans,
    main,
    render,
)

_CHECKS = 0
_FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    global _CHECKS
    _CHECKS += 1
    if not cond:
        _FAILURES.append(f"{label}{(' :: ' + str(detail)) if detail else ''}")


def eq(actual, expected, label: str) -> None:
    check(actual == expected, label, f"expected {expected!r}, got {actual!r}")


#: 临时目录根。**不用 tempfile.mkdtemp**：在某些受限环境（含本项目的开发沙箱）
#: 系统临时目录不可写，而测试应当只依赖工作区。可用 CLAIM_TRACER_TMP 覆盖。
_TMP_ROOT = Path(os.environ.get("CLAIM_TRACER_TMP") or Path(__file__).resolve().parents[1] / ".scratch")
_tmp_seq = itertools.count()


def _tmp() -> Path:
    d = _TMP_ROOT / f"t{next(_tmp_seq):03d}"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
def test_number_extraction_excludes_noise() -> None:
    """只收"值得核查"的 token —— 否则报告会被噪声淹没。"""
    text = (
        "In 2024 we used 3 datasets and 12 layers. "  # 2024/3/12 都不该被收
        "Accuracy reached 0.9412 and latency 12.5 ms, a 3.4% gain. "
        "See Section 4 and Eq. 7. Model f1 scored 0.88."
    )
    nums = extract_numbers(text)
    values = {round(n["value"], 4) for n in nums}

    check(0.9412 in values, "收下 4 位小数", str(sorted(values)))
    check(12.5 in values, "收下带小数的实数", str(sorted(values)))
    check(3.4 in values, "收下百分号的数值部分", str(sorted(values)))
    check(2024 not in values, "**排除年份**（否则报告满是噪声）", str(sorted(values)))
    check(3 not in values, "排除裸整数（数据集个数）", str(sorted(values)))
    check(7 not in values, "排除编号（Eq. 7）", str(sorted(values)))
    check(
        all(n["percent"] or "." in n["raw"] or "e" in n["raw"].lower() for n in nums),
        "全部命中项都有小数位或百分号",
        str([n["raw"] for n in nums]),
    )


def test_latex_input_resolution_uses_main_dir() -> None:
    """``\\input`` 必须相对**主文件目录**解析（bug 2 的回归）。"""
    work = _tmp()
    paper = work / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "figures").mkdir(parents=True)

    (paper / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}\n"
        "\\input{sections/results}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    # 关键：这个 \input 出现在 sections/ 里，但目标在 paper/figures/ 下
    (paper / "sections" / "results.tex").write_text(
        "\\section{Results}\nAccuracy was 0.9412.\n"
        "\\input{figures/evaluation_table}\n",
        encoding="utf-8",
    )
    (paper / "figures" / "evaluation_table.tex").write_text(
        "\\begin{table}\n\\begin{tabular}{lc}\nbaseline & 0.7712 \\\\\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    text, files = expand_paper(paper / "main.tex")
    check("evaluation_table.tex" in files, "跟随到了分节里的 \\input", str(files))
    check("0.7712" in text, "**表格里的数字被读到**（bug 2 的核心）", text[-200:])
    values = {n["value"] for n in extract_numbers(text)}
    check(0.7712 in values, "表格数字进入可核查集合", str(sorted(values)))


def test_zero_tokens_is_not_success() -> None:
    """零发现必须返回 3，而不是 0（bug 1 的回归）。"""
    work = _tmp()
    paper = work / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}\n"
        "We observe a small improvement.\n"  # 刻意一个数字都没有
        "\\end{document}\n",
        encoding="utf-8",
    )
    ev = work / "results"
    ev.mkdir()
    (ev / "metrics.csv").write_text("epoch,acc\n0,0.91\n", encoding="utf-8")

    report = analyze(paper / "main.tex", ev, 0.01)
    eq(report["counts"]["numbers_in_paper"], 0, "确实没有可核查的 token")
    eq(report["counts"]["unmatched"], 0, "因而也没有未溯源项")

    rendered = render(report)
    check(
        "NOTHING WAS CHECKED" in rendered or "nothing was checked" in rendered.lower(),
        "**报告里必须响亮地说「什么都没核查」**（bug 1）",
        rendered[:300],
    )
    check(
        "NOT a pass" in rendered or "NOT A PASS" in rendered.upper(),
        "必须明确写出这不是通过",
        rendered[:300],
    )
    rc = main(["--paper", str(paper / "main.tex"), "--evidence", str(ev)])
    eq(rc, 3, "**退出码 3，与「全部通过=0」区分**（bug 1 的回归）")


def test_orphan_table_detected_across_sibling_dir() -> None:
    """孤儿表要能跨兄弟目录被发现（bug 3 的回归）。"""
    work = _tmp()
    paper = work / "paper"
    (paper / "sections").mkdir(parents=True)
    (work / "figures").mkdir(parents=True)  # 兄弟目录，不在 paper/ 下

    (paper / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}\n"
        "\\input{sections/results}\n\\end{document}\n",
        encoding="utf-8",
    )
    (paper / "sections" / "results.tex").write_text(
        "\\section{Results}\nA small gain.\n", encoding="utf-8"
    )
    (work / "figures" / "evaluation_table.tex").write_text(
        "\\begin{table}\\begin{tabular}{lc}\nbaseline & 0.35 \\\\\n\\end{tabular}\\end{table}\n",
        encoding="utf-8",
    )

    report = analyze(paper / "main.tex", work, 0.01)
    orphans = report["orphans"]
    check(bool(orphans["tables"]), "**发现了未被引用的结果表**（bug 3）", str(orphans))
    check(
        any("evaluation_table" in x for x in orphans["tables"]),
        "报告里指出了是哪一张表",
        str(orphans["tables"]),
    )
    check(
        "ORPHAN" in render(report),
        "报告里出现孤儿段",
    )


def test_tracing_and_derived_candidates() -> None:
    """可溯源 / 派生候选 / 未溯源 三桶要分开。"""
    work = _tmp()
    (work / "paper").mkdir()
    (work / "results").mkdir()
    (work / "paper" / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}\n"
        "Accuracy was 0.9412, latency 12.5 ms, and the gain was 3.4.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    (work / "results" / "metrics.json").write_text(
        '{"acc": 0.9412, "lat": 12.5, "before": 0.91, "after": 0.9412}', encoding="utf-8"
    )

    report = analyze(work / "paper" / "main.tex", work / "results", 0.01)
    c = report["counts"]
    check(c["traced"] >= 2, "0.9412 与 12.5 被溯源", str(c))
    check(c["unmatched"] + c["derived_candidates"] >= 1, "3.4 落进派生或未溯源", str(c))

    # 派生候选必须打印出它用的是哪两个数（可核查，而不是"可能是比值"）
    rows = [r for r in report["rows"] if r["status"] == "missing" and r["derived"]]
    if rows:
        a, b, op = rows[0]["derived"][0]
        check(
            isinstance(a, float) and isinstance(b, float) and op in ("a-b", "a/b"),
            "派生候选带着两个证据值和一个算子（可人工复核口径）",
            str(rows[0]["derived"]),
        )
    else:
        check(True, "（本次没有派生候选，跳过该断言）")

    # 容差：四舍五入后仍应匹配
    (work / "paper" / "r.tex").write_text(
        "\\documentclass{article}\\begin{document}Accuracy 0.94.\\end{document}",
        encoding="utf-8",
    )
    r2 = analyze(work / "paper" / "r.tex", work / "results", 0.01)
    check(r2["counts"]["traced"] >= 1, "0.94 能以 1% 容差匹配到 0.9412", str(r2["counts"]))


def test_missing_evidence_dir_is_usage_error() -> None:
    """用法错误返回 2，与"有未溯源=1"区分。"""
    work = _tmp()
    (work / "p.tex").write_text("x 0.5", encoding="utf-8")
    rc = main(["--paper", str(work / "p.tex"), "--evidence", str(work / "nope")])
    eq(rc, 2, "证据目录不存在 -> 2")


def main_() -> int:
    tests = [
        test_number_extraction_excludes_noise,
        test_latex_input_resolution_uses_main_dir,
        test_zero_tokens_is_not_success,
        test_orphan_table_detected_across_sibling_dir,
        test_tracing_and_derived_candidates,
        test_missing_evidence_dir_is_usage_error,
    ]
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - 一个测试崩了不该掩盖其余的
            import traceback

            _FAILURES.append(f"{fn.__name__} 崩溃: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print()
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)} of {_CHECKS} checks:")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1
    print(f"PASSED {_CHECKS} checks in {len(tests)} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())
