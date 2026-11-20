"""Export the bake-off training data as nnU-Net v2 datasets, and the test
photos as the images every contender predicts.

    venv/Scripts/python bakeoff/export_nnunet.py

  Dataset601_TiS3_B   agreement pseudo-labels + the two hand-labelled photos
  Dataset602_TiS3_C   B + the copy-paste photos (bakeoff/copy_paste.py), if they exist
Labels: background 0, wire 1, ignore 2. "ignore" is nnU-Net's ignore label:
those pixels add nothing to the loss, so the network is never told what they are.
nnU-Net samples training photos uniformly, so each hand-labelled photo is
written --hand-copies times (default 8) to be drawn about as often as 16
pseudo-labelled photos.

Images are 8-bit PNGs at the 50k-equivalent scale, one channel (nnU-Net
z-scores it). The test photos (bakeoff/test_photos.txt) are written to
results/bakeoff/test_images/<case>_0000.png with test_images/manifest.json.

nnU-Net's folders come from $nnUNet_raw; if it is unset, results/bakeoff/nnunet/raw
(the Slurm jobs set the same).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from common import (COPYPASTE, OUT, TEST_IMAGES, TRAIN, case_name, find_photo, img_to_u8,
                    load_photo, read_json, save_u8, test_photos, write_json)

DATASETS = {"B": (601, "TiS3_B"), "C": (602, "TiS3_C")}


def nnunet_raw() -> Path:
    return Path(os.environ.get("nnUNet_raw", OUT / "nnunet" / "raw"))


def write_dataset(key: str, cases: list[tuple[Path, Path, str]], hand_copies: int) -> Path:
    did, name = DATASETS[key]
    root = nnunet_raw() / f"Dataset{did:03d}_{name}"
    if root.exists():
        shutil.rmtree(root)
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir(parents=True)
    n = 0
    for img, lab, case in cases:
        copies = hand_copies if case.startswith("HAND_") else 1
        for k in range(copies):
            cid = f"{case}_c{k}" if copies > 1 else case
            shutil.copyfile(img, root / "imagesTr" / f"{cid}_0000.png")
            shutil.copyfile(lab, root / "labelsTr" / f"{cid}.png")
            n += 1
    write_json(root / "dataset.json", {
        "name": name,
        "description": f"TiS3 wire bake-off contender {key} (bakeoff/README.md)",
        "channel_names": {"0": "SEM"},
        "labels": {"background": 0, "wire": 1, "ignore": 2},
        "numTraining": n,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    })
    print(f"  Dataset{did:03d}_{name}: {n} training images -> {root}")
    return root


def export_tests() -> None:
    if TEST_IMAGES.exists():
        shutil.rmtree(TEST_IMAGES)
    meta = {}
    for photo in test_photos():
        img01, m = load_photo(find_photo(photo))
        case = case_name(photo)
        save_u8(TEST_IMAGES / f"{case}_0000.png", img_to_u8(img01))
        meta[case] = {"photo": photo, "shape": list(img01.shape),
                      "nm_per_px": m["pixel_size_m"] * 1e9 / m["rescale_to_50k"],
                      "rescale_to_50k": m["rescale_to_50k"]}
    write_json(TEST_IMAGES / "manifest.json", meta)
    print(f"  {len(meta)} test photos -> {TEST_IMAGES}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hand-copies", type=int, default=8)
    args = ap.parse_args()

    def listed(folder: Path) -> list[tuple[Path, Path, str]]:
        return [(folder / "images" / f"{c['case']}.png", folder / "labels" / f"{c['case']}.png",
                 c["case"]) for c in read_json(folder / "manifest.json")["cases"]]

    train = listed(TRAIN)
    tests = {case_name(p) for p in test_photos()}
    leaked = [c for _, _, c in train if c in tests]
    if leaked:
        raise SystemExit(f"test photos in the training data: {leaked}. Re-run pseudo_labels.py "
                         f"after editing test_photos.txt.")
    write_dataset("B", train, args.hand_copies)
    if (COPYPASTE / "manifest.json").exists():
        write_dataset("C", train + listed(COPYPASTE), args.hand_copies)
    else:
        print("  no copy-paste photos yet (bakeoff/copy_paste.py): dataset C skipped")
    export_tests()


if __name__ == "__main__":
    main()
