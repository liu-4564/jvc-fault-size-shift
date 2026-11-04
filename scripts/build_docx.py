# -*- coding: utf-8 -*-
"""Build the submission .docx files from the Markdown sources in manuscript/.

NOT RUNNABLE FROM A CLEAN CLONE. This file is kept for provenance only: it is
the driver that produced the submitted .docx files, but it imports a
journal-specific post-processing helper that is not distributed with the
repository, so it exits immediately after cloning. The submitted .docx files
are the artefacts of record; see README, Reproduce, step 3.

Usage:
    python scripts/build_docx.py

Requirements:
    - pandoc on PATH
    - a helper module named ``docx_build_helper`` importable on sys.path
      (e.g. placed next to this script). It must provide ``HDR_CFG``,
      ``build(md_path, out_path, lang)`` and ``report(path)`` - the pandoc
      post-processing wrapper used for style patching, running heads and
      table fitting. The helper performs journal-specific docx plumbing and
      is kept outside this repository; the task list and paths below are the
      project-specific driver logic.

Inputs are read from <repo>/manuscript/ and the .docx files are written next
to the Markdown sources in the same folder.
"""
import os, sys

try:
    import docx_build_helper as B
except ImportError:
    sys.exit('docx_build_helper module not found; see the header comment.')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANU = os.path.join(ROOT, 'manuscript')

B.HDR_CFG['jvc'] = ('Fault-Size Shift in Rolling-Bearing Diagnosis', 'Page ', ' / ', '')
# let the helper resolve figure files relative to the manuscript folder
B.ROOT = MANU

tasks = [
    ('manuscript_main.md', 'Manuscript.docx'),
    ('supplementary_material.md', 'Supplementary_Material.docx'),
    ('cover_letter.md', 'Cover_Letter.docx'),
]
for md, out in tasks:
    B.build(os.path.join(MANU, md), os.path.join(MANU, out), 'jvc')
    B.report(os.path.join(MANU, out))
