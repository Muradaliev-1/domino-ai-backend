"""Bağımlılıksız test çalıştırıcı (pytest varsa `pytest -q tests` de çalışır). Backend klasöründen: python tests/run.py"""
import os
import sys
import time
import traceback
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_demo as mod  # noqa: E402


def main():
    passed = failed = skipped = 0
    t0 = time.time()
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        t = time.time()
        try:
            getattr(mod, name)()
            passed += 1
            print(f"  ok   {name}  ({time.time() - t:.2f}s)")
        except unittest.SkipTest as ex:
            skipped += 1
            print(f"  atlandı {name}: {ex}")
        except Exception:
            failed += 1
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n{passed} geçti, {failed} kaldı" + (f", {skipped} atlandı" if skipped else "") + f", {time.time() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
