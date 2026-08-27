"""Build ``t200_table.npz`` from Blue Robotics' published T200 performance data.

Source
------
https://cad.bluerobotics.com/T200-Public-Performance-Data-10-20V-September-2019.xlsx

201 PWM samples (1100-1900 us, 4 us step) of measured RPM, current, power and
force, at each of 10/12/14/16/18/20 V.  This replaces the hand-fitted
PWM->RPM->thrust polynomials that ``thruster.py`` used to carry: those matched
the 20 V curve (RMSE 0.55 N) but overstated thrust by ~44 % at the 14 V a 4S
pack actually delivers under load, and their affine RPM intercept inflated the
first producible thrust to 1.44 N where the measurement says 0.44 N.

Run once; commit the resulting ``.npz`` (about 7 kB).  The downloaded workbook
is written to a temporary directory and is not kept in the repository.
"""

from __future__ import annotations

import argparse
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


URL = (
    "https://cad.bluerobotics.com/"
    "T200-Public-Performance-Data-10-20V-September-2019.xlsx"
)
# Worksheet order in the published workbook: sheet1 is "READ ME FIRST".
SHEETS = {10: "sheet2", 12: "sheet3", 14: "sheet4",
          16: "sheet5", 18: "sheet6", 20: "sheet7"}
KGF_TO_N = 9.80665
PWM_NEUTRAL_US = 1500.0
PWM_HALF_RANGE_US = 400.0
# Load-cell noise floor.  Below this the measurement cannot separate thrust
# from zero, so it is snapped to exactly zero.  This is what pins the dead-zone
# edge at a repeatable force instead of a noise-dependent one, which the
# inverse lookup needs in order to decide "not producible" consistently.
NOISE_FLOOR_N = 0.20


def _cell_rows(archive: zipfile.ZipFile, sheet: str):
    """Yield each worksheet row as a list of cell strings."""
    shared: list[str] = []
    if "xl/sharedStrings.xml" in archive.namelist():
        blob = archive.read("xl/sharedStrings.xml").decode("utf8")
        shared = [
            re.sub(r"<[^>]+>", "", item)
            for item in re.findall(r"<si>(.*?)</si>", blob, re.S)
        ]
    body = archive.read(f"xl/worksheets/{sheet}.xml").decode("utf8")
    for row in re.findall(r"<row[^>]*>(.*?)</row>", body, re.S):
        cells = []
        for cell in re.finditer(
            r'<c r="[A-Z]+\d+"(.*?)(?:/>|>(.*?)</c>)', row, re.S
        ):
            attributes, inner = cell.group(1), cell.group(2) or ""
            value = re.search(r"<v>(.*?)</v>", inner)
            text = value.group(1) if value else ""
            if 't="s"' in attributes and text != "":
                text = shared[int(text)]
            cells.append(text)
        yield cells


def _read_curve(archive: zipfile.ZipFile, sheet: str):
    """Return ``(pwm_us, force_n)`` for one supply-voltage worksheet."""
    samples = []
    for row in list(_cell_rows(archive, sheet))[1:]:
        try:
            samples.append((float(row[0]), float(row[5]) * KGF_TO_N))
        except (ValueError, IndexError):
            continue
    if not samples:
        raise ValueError(f"no numeric rows parsed from {sheet}")
    samples.sort()
    return (
        np.array([pwm for pwm, _ in samples]),
        np.array([force for _, force in samples]),
    )


def build(out_path: Path, workbook: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        if workbook is None:
            workbook = Path(scratch) / "t200.xlsx"
            urllib.request.urlretrieve(URL, workbook)
        archive = zipfile.ZipFile(workbook)

        volts = sorted(SHEETS)
        pwm_us, curves = None, []
        for volt in volts:
            grid, force = _read_curve(archive, SHEETS[volt])
            if pwm_us is None:
                pwm_us = grid
            elif not np.array_equal(grid, pwm_us):
                raise ValueError(f"{volt} V worksheet uses a different PWM grid")
            curves.append(force)

    force = np.array(curves)
    force[np.abs(force) < NOISE_FLOOR_N] = 0.0
    # The raw data contains a handful of sub-0.14 N reversals near the peaks.
    # ``T200ThrustTable.pwm()`` inverts by binary search, which requires a
    # non-decreasing curve, so enforce that here rather than at load time.
    force = np.maximum.accumulate(force, axis=1)

    out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        volts=np.array(volts, dtype=np.float32),
        pwm_norm=((pwm_us - PWM_NEUTRAL_US) / PWM_HALF_RANGE_US).astype(np.float32),
        force_n=force.astype(np.float32),
        noise_floor_n=np.float32(NOISE_FLOOR_N),
    )

    for index, volt in enumerate(volts):
        dead = pwm_us[force[index] == 0.0]
        print(
            f"{volt:2d} V  {force[index].min():+7.2f} / {force[index].max():+7.2f} N   "
            f"dead zone {dead.min():.0f}-{dead.max():.0f} us "
            f"(+/-{(dead.max() - dead.min()) / 2:.0f})"
        )
    print(f"wrote {out_path} {force.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("t200_table"),
        help="output .npz path (suffix added automatically)",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=None,
        help="use a local copy of the .xlsx instead of downloading it",
    )
    args = parser.parse_args()
    build(args.out, args.workbook)


if __name__ == "__main__":
    main()
