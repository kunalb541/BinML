#!/usr/bin/env python3
"""Render paper/draft_gulls_section.tex with every \\bml* macro replaced by its value and the generated
tables inlined -> paper/outputs/draft_gulls_rendered.txt, for reading and reviewing (not for the build)."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
vals = {}
for f in ("paper_macros.tex", "gulls_macros.tex"):
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", open(os.path.join(HERE, f)).read(), re.M):
        vals[m.group(1)] = m.group(2).replace("{,}", ",")
s = open(os.path.join(HERE, "draft_gulls_section.tex")).read()
s = re.sub(r"\\input\{(outputs/[\w_.]+)\}", lambda m: open(os.path.join(HERE, m.group(1))).read() if os.path.exists(os.path.join(HERE, m.group(1))) else f"[[missing {m.group(1)}]]", s)
missing = sorted(set(re.findall(r"\\(bml\w+)", s)) - set(vals))
s = re.sub(r"\\(bml\w+)", lambda m: vals.get(m.group(1), f"[[{m.group(1)}?]]"), s)
out = os.path.join(HERE, "outputs", "draft_gulls_rendered.txt")
open(out, "w").write(s)
print(f"wrote {out}; undefined macros: {missing}")
