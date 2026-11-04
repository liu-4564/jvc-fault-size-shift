# -*- coding: utf-8 -*-
"""Check that the committed JSON reproduces the numbers printed in the paper.

Standard library only, no dataset and no GPU needed:

    python tests/test_data_consistency.py

Every assertion below is a number that appears in the manuscript or the
supplementary material, compared against data/*.json. If a script is ever
re-run and the results change, this test fails and the paper has to be
updated -- which is the point.
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

DOMAINS = ["0.014@0", "0.014@1", "0.014@2", "0.014@3",
           "0.021@0", "0.021@1", "0.021@2", "0.021@3"]

failures = []


def check(label, got, want, tol=0.005):
    ok = abs(got - want) <= tol
    print("  %-58s got %8.3f  paper %8.3f  %s" % (label[:58], got, want, "ok" if ok else "FAIL"))
    if not ok:
        failures.append(label)


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


uni = load("unified_p3_final.json")
ms = load("multiseed_recall.json")
fisher = load("fisher_12domains.json")

print("Table 4, seed 42 (plain / env / order)")
table4 = {
    "plain": [50.0, 50.0, 50.0, 50.0, 75.0, 73.0, 75.0, 75.0],
    "env": [50.0, 49.5, 50.0, 43.5, 70.0, 73.0, 64.5, 53.5],
    "order": [50.0, 50.0, 49.5, 42.5, 70.0, 74.0, 65.5, 52.0],
}
for rep, want in table4.items():
    for dom, w in zip(DOMAINS, want):
        check("%s  %s" % (rep, dom), uni["%s|42" % rep][dom]["acc"], w)

print()
print("Table 4, three-seed mean of the plain representation")
plain_mean = [50.67, 50.17, 50.00, 48.50, 75.00, 74.33, 76.50, 75.00]
for dom, w in zip(DOMAINS, plain_mean):
    check("plain 3-seed mean  %s" % dom, sum(ms[dom]["acc"]) / 3, w, tol=0.006)

print()
print("Headline figures in the abstract and Section 4.3")
single = sum(uni["plain|42"][d]["acc"] for d in DOMAINS) / 8
three = sum(sum(ms[d]["acc"]) for d in DOMAINS) / 24
acc3 = sum(sum(ms[d]["acc3"]) for d in DOMAINS) / (8 * 3)
check("severity drift, seed 42 mean (62.25)", single, 62.25)
check("severity drift, three-seed mean (62.52)", three, 62.52, tol=0.006)
check("three-class mean over the eight domains (50.03)", acc3, 50.03, tol=0.006)

print()
print("Per-domain three-class means, Section 4.3")
printed = [34.23, 33.56, 33.33, 31.33, 66.67, 65.77, 68.67, 66.67]
for dom, w in zip(DOMAINS, printed):
    # the JSON holds per-seed values rounded to 2 dp, so a recomputed mean can
    # differ from the printed one by up to half a unit in the last place of each
    # seed; 0.02 absorbs that without hiding a real change
    check("acc3  %s" % dom, sum(ms[dom]["acc3"]) / 3, w, tol=0.02)

print()
print("Abstract: per-domain SD of the four-class accuracy")
popsd = [statistics.pstdev(ms[d]["acc"]) for d in DOMAINS]
sampsd = [statistics.stdev(ms[d]["acc"]) for d in DOMAINS]
check("max population SD (abstract: <= 2.12)", max(popsd), 2.12, tol=0.006)
check("max sample SD (n-1, for information: 2.60)", max(sampsd), 2.60, tol=0.01)

print()
print("Table S2, Welch-Fisher ratio (band order 0-500 Hz ... 3.5-4 kHz)")
check("0.007@0HP, 1-1.5 kHz (printed 9.94)", fisher["0.007@0"][2], 9.94)
check("0.021@0HP, 3.5-4 kHz (printed 5.96)", fisher["0.021@0"][7], 5.96)
check("0.014@0HP, 2.5-3 kHz (printed 6.37)", fisher["0.014@0"][5], 6.37)
check("0.014@1HP, 0.5-1 kHz (printed 0.71)", fisher["0.014@1"][1], 0.71)

print()
print("Table S1, confusion matrices of the two representative domains")
for dom, want_acc in (("0.014@0", 50.0), ("0.021@0", 75.0)):
    cell = uni["plain|42"][dom]
    check("S1 accuracy  %s" % dom, cell["acc"], want_acc)
    check("S1 inner-race recall  %s" % dom, cell["recall"]["inner_race"], 0.0, tol=1e-9)
    check("S1 normal recall  %s" % dom, cell["recall"]["normal"], 100.0)

print()
if failures:
    print("%d check(s) failed:" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all checks passed: the committed JSON reproduces the paper's numbers")
