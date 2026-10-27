"""Build the LaTeX rows of Tables 5 and 6 from the evaluation / training JSONs.

Every entry is mean (seed 0, seed 1): three significant figures, scientific notation below 1e-3.
Usage: python make_final_tables.py q2/results/final   (prints both tables as two blocks of columns each)
"""
import os
import sys
import json
import math

COLS = ["E5", "C_M", "E_M", "C_R", "E_R", "E2_1", "E2_2", "E2_3", "params", "train", "infer"]


def plain(x):
    return f"{x:#.3g}"


def sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def mean_cell(x, key):
    if key == "params":
        return f"{int(round(x)):,}".replace(",", "{,}")
    return sci(x) if 0 < abs(x) < 1e-3 else plain(x)


def seed_cell(mean, vals):
    """(a, b) with a shared exponent when the mean is in scientific notation."""
    if 0 < abs(mean) < 1e-3:
        e = int(f"{mean:.2e}".split("e")[1])
        mant = [v / 10 ** e for v in vals]
        if all(1 <= m < 10 for m in mant):
            return f"({plain(mant[0])}, {plain(mant[1])})\\times10^{{{e}}}"
        return "(" + ", ".join(f"${sci(v)}$" for v in vals) + ")"
    return "(" + ", ".join(f"${sci(v)}$" if 0 < abs(v) < 1e-3 else plain(v) for v in vals) + ")"


def load(d, name, seed, sym):
    ev = json.load(open(os.path.join(d, f"eval_{name}{'_sym' if sym else ''}_seed{seed}.json")))
    tr = json.load(open(os.path.join(d, f"train_{name}_seed{seed}.json")))
    ev["train"] = tr["mean_epoch_time"]
    ev["infer"] = ev["inference_time"]
    return ev


HEAD1 = r"""\begin{tabular}{@{}l ccccc@{}}
  \toprule
  Model & $E^{(5)}$ & $C_M$ & $E_M$ & $C_R$ & $E_R$ \\
  \midrule
"""
HEAD2 = r"""\begin{tabular}{@{}l ccc rcc@{}}
  \toprule
  Model & $E^{(2)}_{1\times}$ & $E^{(2)}_{2\times}$ & $E^{(2)}_{3\times}$ & Params & Train (s/epoch) & Infer (s) \\
  \midrule
"""


def cell(mean, k):
    c = mean_cell(mean, k)
    return c.replace("\\times", "$\\times", 1).replace("}", "}$", 1) if "\\times" in c else c


def sub(mean, vals, k):
    if k == "params":
        return ""
    c = seed_cell(mean, vals)
    if "$" in c:
        return "\\scriptsize" + c
    if "\\times" in c:
        a, b = c.split("\\times", 1)
        return "\\scriptsize" + a + "$\\times" + b + "$"
    return "\\scriptsize" + c


def block(d, items, sym, cols, head):
    out = []
    for name, label in items:
        ev = [load(d, name, s, sym) for s in (0, 1)]
        mean = {k: (ev[0][k] + ev[1][k]) / 2 for k in COLS}
        out.append(f"{label} & " + " & ".join(cell(mean[k], k) for k in cols) + " \\\\\n & "
                   + " & ".join(sub(mean[k], [ev[0][k], ev[1][k]], k) for k in cols) + " \\\\")
    return head + "\n\\addlinespace[2pt]\n".join(out) + "\n  \\bottomrule\n\\end{tabular}"


def two_blocks(d, items, sym):
    return (block(d, items, sym, ["E5", "C_M", "E_M", "C_R", "E_R"], HEAD1) + "\n\n\\vspace{8pt}\n\n"
            + block(d, items, sym, ["E2_1", "E2_2", "E2_3", "params", "train", "infer"], HEAD2))


T5 = [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")]
T6 = [("B", "B, symmetrised"), ("D", "D, symmetrised (final)")]

if __name__ == "__main__":
    d = sys.argv[1]
    print("% ---- Table 5")
    print(two_blocks(d, T5, False))
    print("% ---- Table 6")
    print(two_blocks(d, T6, True))
