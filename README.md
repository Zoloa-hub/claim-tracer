# claim-tracer

**Trace every number in a paper back to an artifact on disk.**

```bash
pip install claim-tracer
claim-tracer --paper paper/main.tex --evidence ./results
```

Exit codes: `0` all traced · `1` untraceable numbers or orphan artifacts · `2` usage error · **`3` nothing was checked**

---

## The failure it catches

AI-assisted writing does not usually fail by being badly written. It fails by
being **fluent, well-formatted, numerically precise — and untethered**:

> The paper reports an improvement, a p-value, a runtime. Those numbers appear
> nowhere in any result file. Nothing errors. Nothing looks wrong. The prose
> reads fine, the figures render, the PDF compiles.

Nobody checks this by hand — manually cross-referencing every numeric token in a
paper is impractical. Reviewers cannot check it either: they do not have your
results directory.

So it ships.

## What it does

1. Extracts every numeric token from the paper (Markdown or LaTeX), with context
2. Indexes every number in your results directory (CSV/TSV/JSON/JSONL/TXT)
3. Reports what cannot be traced

It also follows `\input` / `\include`, so multi-file LaTeX papers are read whole.

## What it does **not** do

This part matters more than the feature list.

**It is a range check, not a correctness check.**

- Proves: the paper contains no number that exceeds what the artifacts support
- **Does not** prove: every number is right. Writing `accuracy` where you meant
  `f1`, citing the wrong row, swapping mean and max — none of that is detectable here.

And **"traced" does not mean "used correctly"**: a number can exist in the
artifacts and still be attached to the wrong claim.

**Do not read a clean report as "the paper is fine".** It only tells you which
numbers have no source.

## The three buckets

The report separates findings by how much human attention they deserve — this is
the whole design, not a formatting choice:

| Bucket | Meaning | Your cost |
|---|---|---|
| **traced** | The number appears in the artifacts | Low — but check it is attached to the right claim |
| **derived candidates** | Not found directly, but explainable as a ratio or difference of two evidence values. **Each candidate is printed with the two numbers it used.** | Medium — confirm the denominator/definition |
| **untraced** | No source, and not a simple derivation | High — this is the list you came for |

Derived candidates are **candidate explanations, not verifications**. They are
kept separate precisely because "probably a ratio" and "no source at all" warrant
very different levels of scrutiny.

## Orphan artifacts

The tool also asks the opposite question, which almost nothing else asks:
**did you generate something that never made it into the paper?**

A real case, found by this tool:

```
孤儿产物（生成了但没进论文 —— 没有任何东西会为此报错）
  [table never referenced] ../figures/evaluation_table.tex
```

The pipeline had generated a complete results table with real `mean ± std`
values. `main.tex` only `\input`-ed the eight section files. **The table was
never referenced anywhere.** The paper's body therefore contained no numbers at
all — only qualitative phrasing like "a small improvement".

The reviewer would never know the table existed. Neither, apparently, would the
author.

Nothing errors in this scenario. The file exists, its content is correct, the
PDF compiles. It is simply never used.

## Exit code 3: nothing was checked

If the paper yields zero checkable numeric tokens, the tool **does not report
success** — it prints a loud warning and exits `3`.

This is deliberate. An early version returned `0` ("no untraced numbers") for a
paper whose numbers were all in an `\input`-ed table the tool failed to read.
**"Found nothing" was being reported as "all clear"** — the exact class of error
this tool exists to catch.

Zero findings is not a pass. It is a failure to check.

## Usage

```bash
claim-tracer --paper paper/main.tex --evidence ./results
claim-tracer --paper paper.md --evidence ./results --tolerance 0.02
claim-tracer --paper paper/main.tex --evidence ./results --json report.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--paper` | required | Paper file (`.tex`, `.md`, `.txt`) |
| `--evidence` | required | Directory of result files to trace against |
| `--tolerance` | `0.01` | Relative tolerance, so rounded numbers still match |
| `--json` | — | Also write the full report as JSON |

Use it as a CI gate: exit code `1` or `3` should fail the build.

## Install

```bash
pip install claim-tracer      # no dependencies
```

Requires Python 3.10+. There are **no mandatory third-party dependencies** — the
tool has to run against a paper and a results directory on any machine, offline.

## Scope and honesty

The evidence index is deliberately **broad**: any number in any text file under
`--evidence` counts as evidence. A narrow index produces false positives, and a
report full of false positives is a report nobody reads.

Consequences you should know:

- Binary results (`.npz`, `.npy`, `.pt`, `.pdf`, `.h5`) are **not** parsed. They
  are listed as skipped. Numbers inside images are invisible to this tool.
- Only tokens with a decimal point or a percent sign are checked. Bare integers
  (years, counts, section numbers) are excluded on purpose — including them
  buries the real findings in noise.
- Matching is by value, not semantics. Two different metrics that happen to share
  a value will both "trace".

Everything the tool cannot see is reported as a skip. Silence is not evidence.

## License

Apache-2.0
