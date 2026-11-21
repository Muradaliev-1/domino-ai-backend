"""Araştırma çekirdeğini (domino-research/domino) backend'e sabit bir commit'ten kopyalar (D-15, D-26).

Backend `domino` paketini kütüphane olarak kullanır; Render'da özel repoya erişim gerekmesin diye paket
`backend/domino/` altına AYNEN kopyalanır ve kaynağın commit hash'i `DOMINO_VERSION` dosyasına yazılır.
Kopyada elle değişiklik yapılmaz: değişiklik domino-research'te yapılır, testlenir, sonra yeniden kopyalanır.

Kullanım (backend klasöründen):
    python tools/sync_domino.py --src C:/Users/murad/Projects/domino-research          # kopyala
    python tools/sync_domino.py --src C:/Users/murad/Projects/domino-research --check  # yalnız karşılaştır
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(HERE, "domino")
VERSION_FILE = os.path.join(HERE, "DOMINO_VERSION")


def src_files(src_pkg):
    return sorted(f for f in os.listdir(src_pkg) if f.endswith(".py"))


def git_head(src):
    out = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "-C", src, "status", "--porcelain", "--", "domino"],
                           capture_output=True, text=True, check=True).stdout.strip()
    return out.stdout.strip(), bool(dirty)


def check(src):
    """Kopya kaynakla birebir mi? Farklı dosyaların listesini döndürür (boş = aynı)."""
    src_pkg = os.path.join(src, "domino")
    a, b = set(src_files(src_pkg)), set(src_files(DST)) if os.path.isdir(DST) else set()
    diff = sorted(a ^ b)
    diff += [f for f in sorted(a & b) if not filecmp.cmp(os.path.join(src_pkg, f), os.path.join(DST, f), shallow=False)]
    return diff


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        diff = check(args.src)
        print("AYNI" if not diff else "FARKLI: " + ", ".join(diff))
        return 1 if diff else 0
    head, dirty = git_head(args.src)
    if dirty:
        print("domino-research/domino içinde commit edilmemiş değişiklik var: önce commit et.")
        return 2
    src_pkg = os.path.join(args.src, "domino")
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)
    for f in src_files(src_pkg):
        shutil.copy2(os.path.join(src_pkg, f), os.path.join(DST, f))
    with open(VERSION_FILE, "w", encoding="utf-8") as fh:
        fh.write(head + "\n")
    print(f"{len(src_files(src_pkg))} dosya kopyalandı, domino-research {head[:7]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
