"""Level 1 demo: print valid examples and hard negatives for every label.

Run:  python level1_demo.py
"""

from aupii import validators as v
from aupii.generators import PIIGenerator
from aupii.labels import CHECKSUM_LABELS, ENTITY_TYPES


def main() -> None:
    gen = PIIGenerator(seed=7)

    print("VALID EXAMPLES (3 per label)\n")
    for et in ENTITY_TYPES:
        samples = [gen.generate(et.label).replace("\n", " / ") for _ in range(3)]
        tag = "checksum" if et.has_checksum else "format"
        print(f"  {et.label:<18} [{tag:<8}]  " + "  |  ".join(samples))

    print("\nHARD NEGATIVES (look right, fail the checksum)\n")
    for label in CHECKSUM_LABELS:
        good = gen.generate(label, style="spaced")
        bad = gen.corrupt(good, v.VALIDATORS[label])
        print(f"  {label:<12}  valid {good:<16} -> corrupted {bad:<16} "
              f"validator says: {v.VALIDATORS[label](bad)}")


if __name__ == "__main__":
    main()
