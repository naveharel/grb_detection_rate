"""Render report pages and contact sheets for the mandatory visual check.

Uses the local pypdfium2 installation in tmp/toolchain/python when present.
Rendering creates inspection artifacts, not an assertion that inspection passed.
"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tmp" / "toolchain" / "python"))
import pypdfium2 as pdfium
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", default=str(ROOT / "docs" / "simplified_luminosity_function.pdf"))
    parser.add_argument("--out", default=str(ROOT / "tmp" / "pdfs" / "simplified_lf" / "pages"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(args.pdf)
    paths = []
    for i, page in enumerate(doc):
        pil = page.render(scale=1.7).to_pil()
        path = out / f"page-{i+1:02}.png"
        pil.save(path)
        paths.append(path)
        page.close()
    for start in range(0, len(paths), 6):
        sheet = Image.new("RGB", (1200, 1770), "#e8e8e8")
        draw = ImageDraw.Draw(sheet)
        for k, path in enumerate(paths[start:start+6]):
            page = Image.open(path).convert("RGB")
            page.thumbnail((570, 545))
            x, y = 15+(k%2)*600, 25+(k//2)*590
            sheet.paste(page, (x, y+18))
            draw.text((x, y), f"Page {start+k+1}", fill="black")
        sheet.save(out / f"contact-{start//6+1:02}.png")
    print(f"Rendered {len(paths)} pages to {out}")


if __name__ == "__main__":
    main()
