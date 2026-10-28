"""D1 (in-house UWB, DWM1001, three rooms) loader — withheld.

The D1 measurements are not public (available from the authors on reasonable request; see the paper's Data
Availability Statement). The original loader read the raw D1 workbooks and is not included. Every analysis on
D1 in this repository therefore cannot be re-run; its primary outputs are included under results/.
"""


def _unavailable(*_, **__):
    raise RuntimeError("D1 data and loader are not public; see src/data/d1.py")


load_raw = prep = cycles = files = _unavailable
