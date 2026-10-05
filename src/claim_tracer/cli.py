#!/usr/bin/env python3
"""claim-tracer - trace every number in a paper back to an artifact on disk.

## The problem

AI 辅助写作最危险的失效不是"写得差"，是**数字对不上产物**，而且**不会响**：
论文读起来通顺、图表齐全、数字精确到小数点后两位——只是那些数字在任何
结果文件里都找不到。

这类错误没有人会查（人手核对一篇论文的数值 token 不现实），审稿人也不会查
（他们没有你的实验目录）。所以它会一路走到发表。

## What it does

1. 从论文（Markdown / LaTeX）抽出全部数值 token，连同上下文
2. 从证据目录（CSV / JSON / JSONL / TSV / npz-元数据）抽出全部数值
3. 逐个比对，输出**未溯源清单**

## What it does NOT do (this matters more)

**它是范围检查，不是正确性检查。**

* 能证明：论文里没有编造**超出证据范围**的数字
* **不能**证明：每个数字都对。把 accuracy 写成 f1、把均值写成最大值、
  引用错行——这些都抓不出来

同理，"找到了"不等于"用对了"：一个数字可能恰好在证据里存在，
但被安在了错误的结论上。

**别把这份报告当成"论文没问题"的证明。** 它只能告诉你"哪些数字我没有出处"。

## Derived numbers (where it is easiest to fool yourself)

论文里天然存在派生值：百分比、差值、求和、平均。它们不会逐字出现在结果文件里。

本工具**不假装能验证它们**，而是：
* 先尝试把论文数字解释为证据里两个数的**简单派生**（比值、差值）
* 能解释的归入 `derived`，并**标出用的是哪两个数**
* 解释不了的归入 `unmatched`

关键：`derived` 是**候选解释**，不是验证通过。报告里两者分开列，
因为它们的不确定性完全不同——一个"没出处"的数字和一个"可能是商"的数字，
需要你花的核查成本差很多。

Usage:

    python claim_tracer.py --paper paper.md --evidence ./results
    python claim_tracer.py --paper paper.tex --evidence ./results --tolerance 0.02
    python claim_tracer.py --paper paper.md --evidence ./results --json report.json

Exit codes: 0 = all traced; 1 = untraceable numbers or orphan artifacts; 2 = usage error; 3 = nothing was checked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path


def pathlib_relative(p, base):
    '''相对路径，跨 root 时退化成相对上层的路径。'''
    import os
    return Path(os.path.relpath(p, base))

# --- 数值 token -------------------------------------------------------------- #

#: 匹配论文里的数字。刻意要求**有小数位或百分号**才算"值得核查的数值token"——
#: 否则会把年份、编号、章节号、"3 个数据集"这类非结果数字全部卷进来，
#: 报告立刻被噪声淹没，工具也就没人用了。
_NUMBER = re.compile(
    r"""
    (?<![A-Za-z0-9_.])          # 左边界：不是标识符的一部分（避开 f1、v2、2024a）
    (?P<num>
        \d{1,3}(?:,\d{3})+(?:\.\d+)?   # 1,234 / 1,234.5
      | \d+\.\d+                       # 12.34
      | \.\d+                          # .5
      | \d+(?:\.\d+)?\s*%              # 12% / 12.3 %
      | \d+(?:\.\d+)?\s*[eE][-+]?\d+   # 1.2e-3
    )
    (?![A-Za-z])                # 右边界
    """,
    re.VERBOSE,
)

#: 数值所在位置附近的上下文长度（用于人工核查）
_CONTEXT = 90


def extract_numbers(text: str) -> list[dict]:
    """抽出论文里的数值 token 及上下文。"""
    found: list[dict] = []
    for m in _NUMBER.finditer(text):
        raw = m.group("num")
        value = _to_float(raw)
        if value is None:
            continue
        start = max(0, m.start() - _CONTEXT)
        end = min(len(text), m.end() + _CONTEXT)
        line = text.count("\n", 0, m.start()) + 1
        found.append(
            {
                "raw": raw.strip(),
                "value": value,
                "percent": raw.strip().endswith("%"),
                "line": line,
                "context": " ".join(text[start:end].split()),
            }
        )
    return found


def _to_float(raw: str) -> float | None:
    s = raw.strip().rstrip("%").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


# --- 证据索引 ---------------------------------------------------------------- #

#: 证据文件扩展名 -> 读取方式
_TEXT_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".md", ".log", ".ndjson"}


def collect_evidence(root: Path) -> tuple[list[float], int, list[str]]:
    """从证据目录抽出全部数值。

    返回 ``(数值列表, 扫描文件数, 跳过的文件说明)``。

    刻意做到**宽进**：任何文本文件里的任何数字都算证据。
    理由：这个工具要回答的是"论文里的数字有没有出处"，
    过窄的证据索引会产生大量假阳性，报告就没人看了。
    """
    values: list[float] = []
    skipped: list[str] = []
    n_files = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            # 二进制（.npz/.npy/.pt 等）不解析内部——本工具不假装懂它们
            if path.suffix.lower() in {".npz", ".npy", ".pt", ".pth", ".h5", ".pdf"}:
                skipped.append(f"{path.name} (binary; contents not parsed)")
            continue
        n_files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            skipped.append(f"{path.name} (read failed: {exc})")
            continue
        for m in re.finditer(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text):
            v = _to_float(m.group(0))
            if v is not None and math.isfinite(v):
                values.append(v)

    return values, n_files, skipped


def build_index(values: list[float]) -> dict:
    """把证据数值建成可快速查询的结构（含派生候选用的排序数组）。"""
    uniq = sorted(set(values))
    return {"values": uniq, "count": len(values)}


def _close(a: float, b: float, tol: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) <= tol * scale


def match(value: float, index: dict, tolerance: float) -> tuple[str, float | None]:
    """判断一个论文数字是否有出处。

    返回 ``(状态, 命中的证据值)``，状态为 ``exact`` / ``rounded`` / ``missing``。
    """
    vals: list[float] = index["values"]
    if not vals:
        return "missing", None

    # 精确或容差内
    for v in vals:
        if _close(value, v, tolerance):
            return ("exact" if value == v else "rounded"), v

    # 百分比形式：论文写 12.3%，证据里可能是 0.123
    if value > 1.0:
        for v in vals:
            if _close(value / 100.0, v, tolerance):
                return "rounded", v

    return "missing", None


def derived_candidates(
    value: float, index: dict, tolerance: float, max_pairs: int = 400
) -> list[tuple[float, float, str]]:
    """尝试把数字解释为证据里两个数的**简单派生**。

    只试比值与差值——这两个覆盖了论文里绝大多数派生值（相对提升、绝对差值）。
    刻意不做加权组合、回归拟合之类：那会给出大量无意义的"解释"，
    让报告失去可信度。**宁可说"没出处"，也不要给一个凑出来的解释。**
    """
    vals: list[float] = index["values"]
    out: list[tuple[float, float, str]] = []
    n = len(vals)
    if n < 2 or n > max_pairs:
        return out

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = vals[i], vals[j]
            # 差值
            if _close(a - b, value, tolerance):
                out.append((a, b, "a-b"))
            # 比值（百分比形式）
            if b != 0:
                for cand in (a / b, (a / b - 1.0) * 100.0):
                    if _close(cand, value, tolerance):
                        out.append((a, b, "a/b"))
                        break
            if len(out) >= 3:
                return out
    return out[:3]


# --- 报告 -------------------------------------------------------------------- #

def expand_paper(paper: Path) -> tuple[str, list[str]]:
    """读取论文全文，**跟随 LaTeX 的 ``\\input`` / ``\\include``**。

    为什么必须跟随：实测一篇真实论文的 ``main.tex`` 只有 4.3 KB 骨架，
    正文与数字全在 ``\\input{sections/...}`` 进来的分节文件里。
    只看 ``main.tex`` 会得到"数值 token = 0"，而工具当时把它报成了**通过**——
    「什么都没查到」被当成「全部可溯源」。这是本工具最不能犯的错误，
    因为它自己就是用来防这类事的。

    返回 ``(全文, 读取的文件列表)``。
    """
    parts: list[str] = []
    seen: list[str] = []
    root_dir = paper.parent

    def _read(path: Path, depth: int = 0) -> None:
        if depth > 8 or not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        seen.append(path.name)
        # 先把 \input/\include 展开，再交给调用方
        def _sub(m: re.Match) -> str:
            target = m.group(2).strip()
            if not target:
                return ""
            # **解析顺序很关键。** LaTeX 的 \input 相对**主文件所在目录**解析，
            # 而不是相对"包含它的那个文件"的目录。
            #
            # 实测踩过：main.tex 里 \input{sections/results}，而 results.tex 里
            # \input{figures/evaluation_table} —— 表格在 paper/figures/ 下。
            # 按"包含文件"的目录解析会去找 paper/sections/figures/，找不到，
            # 于是**全篇的数字都在表格里却一个都没读到**，工具报"数值 token = 0"。
            # 这正是本工具要防的那类错误，所以顺序与注释都留在这里。
            candidates = [
                path.parent / target,
                path.parent / (target + ".tex"),
                root_dir / target,
                root_dir / (target + ".tex"),
                root_dir / target if target.endswith(".tex") else root_dir / (target + ".tex"),
            ]
            for cand in candidates:
                if cand.is_file():
                    _read(cand, depth + 1)
                    return ""
            return ""

        text = re.sub(r"\\(input|include)\s*\{([^}]*)\}", _sub, text)
        parts.append(text)

    _read(paper)
    return "\n".join(parts), seen


def find_orphans(paper_root: Path, reached: list[str], text: str) -> dict:
    """找出**生成了却没有进入论文**的产物。

    为什么这是独立且重要的一类问题：绝大多数检查问的是
    "论文里的东西有没有依据"，而没人问"**你生成的东西有没有进论文**"。

    实测踩到过真实案例：管线生成了一张完整的结果表
    （``figures/evaluation_table.tex``，带 ``mean ± std`` 真值），
    但 ``main.tex`` 只 ``\\input`` 了 8 个分节，**那张表从未被引用**。
    于是论文正文里一个数字都没有，全是"小幅增益"这类定性说法——
    而**审稿人不会知道有这张表**，作者自己也未必意识到。

    这类产物的共同特征：**没有任何东西会报错**。文件存在、内容正确、
    编译通过，只是没人用它。
    """
    reached_names = set(reached)
    tables: list[str] = []
    # 搜索范围要**向上扩一层**：实测的布局是 deliverables/paper/ 与
    # deliverables/figures/ 并列，结果表在 figures/ 下——只扫 paper/ 够不到，
    # 于是恰恰漏掉了孤儿表这个最该报的东西。
    roots = [paper_root]
    if paper_root.parent != paper_root:
        roots.append(paper_root.parent)
    seen_tex: set = set()
    candidates: list = []
    for root in roots:
        for tex in sorted(root.rglob("*.tex")):
            if tex in seen_tex:
                continue
            seen_tex.add(tex)
            candidates.append(tex)
    for tex in candidates:
        if tex.name == "main.tex" or tex.name in reached_names:
            continue
        try:
            body = tex.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "\\begin{tabular}" in body or "\\begin{table}" in body:
            try:
                rel = tex.relative_to(paper_root)
            except ValueError:
                rel = pathlib_relative(tex, paper_root)
            tables.append(str(rel))

    missing: list[str] = []
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]*)\}", text):
        raw = m.group(1).strip()
        if not raw:
            continue
        cands = [paper_root / raw, paper_root / "figures" / raw, paper_root / "figs" / raw]
        found = False
        for c in cands:
            if c.is_file() or any(c.with_suffix(e).is_file() for e in (".pdf", ".png", ".jpg", ".eps")):
                found = True
                break
        if not found:
            missing.append(raw)

    unused: list[str] = []
    for fdir in (paper_root / "figures", paper_root / "figs"):
        if not fdir.is_dir():
            continue
        for img in sorted(fdir.iterdir()):
            if img.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".eps", ".tex"}:
                continue
            if img.stem not in text:
                unused.append(str(img.relative_to(paper_root)))

    return {"tables": tables, "figures_missing": missing, "figures_unused": unused}


def analyze(
    paper: Path, evidence_dir: Path, tolerance: float, check_orphans: bool = True
) -> dict:
    text, paper_files = expand_paper(paper)
    numbers = extract_numbers(text)
    values, n_files, skipped = collect_evidence(evidence_dir)
    index = build_index(values)
    orphans = (
        find_orphans(paper.parent, paper_files, text)
        if check_orphans
        else {"tables": [], "figures_missing": [], "figures_unused": []}
    )

    rows: list[dict] = []
    for item in numbers:
        status, hit = match(item["value"], index, tolerance)
        row = dict(item)
        row["status"] = status
        row["evidence"] = hit
        if status == "missing":
            row["derived"] = derived_candidates(item["value"], index, tolerance)
        else:
            row["derived"] = []
        rows.append(row)

    unmatched = [r for r in rows if r["status"] == "missing" and not r["derived"]]
    derived_only = [r for r in rows if r["status"] == "missing" and r["derived"]]
    traced = [r for r in rows if r["status"] != "missing"]

    return {
        "paper": str(paper),
        "paper_files": paper_files,
        "orphans": orphans,
        "evidence_dir": str(evidence_dir),
        "tolerance": tolerance,
        "evidence": {"files_scanned": n_files, "values": index["count"],
                     "skipped": skipped},
        "counts": {
            "numbers_in_paper": len(rows),
            "traced": len(traced),
            "derived_candidates": len(derived_only),
            "unmatched": len(unmatched),
        },
        "rows": rows,
    }


def render(report: dict) -> str:
    c = report["counts"]
    ev = report["evidence"]
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("claim-tracer - paper numbers vs. artifacts on disk")
    add("=" * 78)
    add(f"  paper    : {report['paper']}")
    add(f"  evidence : {report['evidence_dir']}")
    add(f"  index    : {ev['files_scanned']} text files / {ev['values']} numeric values")
    add(f"  tolerance: {report['tolerance']:.0%} (relative)")
    for s in ev["skipped"][:4]:
        add(f"    - skipped: {s}")
    add("")
    add(f"  numeric tokens in paper : {c['numbers_in_paper']}")
    add(f"  traced                  : {c['traced']}")
    add(f"  derived candidates      : {c['derived_candidates']}  (candidate explanations, NOT verified)")
    add(f"  UNTRACED                : {c['unmatched']}")
    add("")
    add(f"  files read: {', '.join(report.get('paper_files') or []) or '(none)'}")
    add("")

    # **"什么都没查到"必须响亮地说出来，不能报成通过。**
    # 实测踩过：main.tex 只有骨架、数字都在 \input 的分节里，
    # 工具得到 0 个数值 token，却因为"0 个未溯源"而退出码 0。
    # 零发现 ≠ 已验证——这是本工具最不能犯的错误。
    if c["numbers_in_paper"] == 0:
        add("!" * 78)
        add("!! WARNING: no checkable numeric token was found in the paper.")
        add("!! **NOTHING WAS CHECKED.** This is NOT a pass.")
        add("!!   - the paper is a LaTeX skeleton and numbers live in \\input-ed files")
        add("!!   - the numbers are inside images (not parsed by this tool)")
        add("!!   - every number is a bare integer (only decimals/percents are checked)")
        add("!" * 78)
        add("")

    if c["traced"]:
        add("-" * 78)
        add("TRACED - present in the artifacts (NOT the same as used correctly)")
        add("-" * 78)
        for r in [x for x in report["rows"] if x["status"] != "missing"][:12]:
            add(f"  {r['value']:<14.6g} <- evidence {r['evidence']:<14.6g} "
                f"[line {r['line']}, {r['status']}]")
        if c["traced"] > 12:
            add(f"  ... and {c['traced'] - 12} more")

    if c["derived_candidates"]:
        add("")
        add("-" * 78)
        add("DERIVED CANDIDATES - candidate explanations, NOT verified")
        add("-" * 78)
        for r in [x for x in report["rows"] if x["status"] == "missing" and x["derived"]][:10]:
            cands = "; ".join(f"{a:g} {op} {b:g}" for a, b, op in r["derived"])
            add(f"  {r['value']:<14.6g} [line {r['line']}] could be {cands}")
            add(f"      ... {r['context'][:120]}")

    if c["unmatched"]:
        add("")
        add("-" * 78)
        add("UNTRACED - no source in any result file, and not a simple derivation")
        add("-" * 78)
        for r in [x for x in report["rows"] if x["status"] == "missing" and not x["derived"]]:
            add(f"  {r['value']:<14.6g} [line {r['line']}]")
            add(f"      ... {r['context'][:150]}")

    orph = report.get("orphans") or {}
    if orph.get("tables") or orph.get("figures_missing") or orph.get("figures_unused"):
        add("")
        add("-" * 78)
        add("ORPHAN ARTIFACTS - generated but never referenced by the paper")
        add("-" * 78)
        for name in orph.get("tables", []):
            add(f"  [table never referenced] {name}")
            add("      这张表存在且内容完整，但论文里没有任何地方 \\input 它，")
            add("      so none of its numbers appear in the body. A reviewer will never know.")
        for name in orph.get("figures_unused", []):
            add(f"  [figure never referenced] {name}")
        for name in orph.get("figures_missing", []):
            add(f"  [图片缺失]   {name} 被 \\includegraphics 引用但文件不存在")
        add("")

    add("")
    add("=" * 78)
    add("This is a RANGE check, not a correctness check:")
    add("  proves     - no number in the paper exceeds what the artifacts support")
    add("  does NOT   - prove every number is right (wrong metric name, wrong row:")
    add("               none of that is detectable here).")
    add("  Also: 'traced' does NOT mean 'used correctly' - a number can exist in the") 
    add("  artifacts and still be attached to the wrong claim.")
    add("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Trace every number in a paper back to an artifact on disk. "
                    "A RANGE check, not a correctness check: it proves no number "
                    "exceeds what the artifacts support; it does NOT prove every "
                    "number is right."
    )
    ap.add_argument("--paper", required=True, type=Path)
    ap.add_argument("--evidence", required=True, type=Path)
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="relative tolerance; default 1%%, so rounded numbers still match")
    ap.add_argument("--json", type=Path, default=None, help="also write the full report as JSON")
    args = ap.parse_args(argv)

    if not args.paper.is_file():
        sys.stderr.write(f"paper not found: {args.paper}\n")
        return 2
    if not args.evidence.is_dir():
        sys.stderr.write(f"evidence directory not found: {args.evidence}\n")
        return 2

    report = analyze(args.paper, args.evidence, args.tolerance)
    print(render(report))
    if args.json:
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nreport written to {args.json}")
    # 退出码语义：0=全部可溯源 / 1=有未溯源 / 2=用法错误 / **3=什么都没核查**
    if report["counts"]["numbers_in_paper"] == 0:
        return 3
    orph = report.get("orphans") or {}
    if report["counts"]["unmatched"]:
        return 1
    # 有孤儿产物也返回 1：主结果表没进论文，与"数字没出处"同级严重
    if orph.get("tables") or orph.get("figures_missing"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
