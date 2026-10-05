# Report toolchain

The report was built with portable **Tectonic 0.17.0** (XeTeX/xdvipdfmx) and
rendered with **pypdfium2 5.13.0**. These are local tools under `tmp/toolchain/`;
the app and project venv packages were not changed. The existing MiKTeX directory
could not be inspected within the filesystem sandbox, so a portable compiler was
used. The temporary tools/cache are not source deliverables.

Official compiler archive:
https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip

Downloaded archive SHA256:
`f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f`

Executable SHA256:
`99ffcfdbf1ebf8bdda9e791942e3d06aedb12463fddc33f07de6f5211c8bf08d`

Tectonic downloaded its standard `default_bundle_v33` resources on the first
compile. `TECTONIC_CACHE_DIR` was set to this checkout's
`tmp/toolchain/tectonic-cache` to keep downloaded resources local.

## Rebuild

For the current analytical report, run `build_pedagogical_figures.py`.
The old numerical TeX includes are no longer manuscript dependencies.
From the `docs/` directory:

```powershell
$env:TECTONIC_CACHE_DIR = 'C:/Users/naveh/PycharmProjects/grb_detections/tmp/toolchain/tectonic-cache'
../tmp/toolchain/tectonic/tectonic.exe --only-cached --keep-logs --keep-intermediates --outdir ../tmp/pdfs/simplified_lf simplified_luminosity_function.tex
```

Tectonic automatically repeats TeX passes until references stabilize. Omit
`--only-cached` for the first build on a new installation, which needs network
access. A normal `pdflatex` installation can alternatively compile the same
source twice from `docs/`.

The renderer was installed without changing the venv:

```powershell
.venv/Scripts/python.exe -m pip install --target tmp/toolchain/python pypdfium2==5.13.0
```

From the repository root, render the candidate PDF:

```powershell
.venv/Scripts/python.exe -B analysis/simplified_luminosity_function/render_report.py tmp/pdfs/simplified_lf/simplified_luminosity_function.pdf
```

Inspect every page in `tmp/pdfs/simplified_lf/pages/` and the individual figure
PNGs before delivering a rebuilt version. Then copy the inspected PDF to
`docs/simplified_luminosity_function.pdf`.

Matplotlib used the project's documented STIX mathtext fallback because
`latex` and `dvipng` were absent; portable Tectonic is not a replacement for
Matplotlib's DVI toolchain. The report itself is fully typeset by LaTeX.

## Pedagogical revision, 2026-10-04

The revised report also uses Tectonic 0.17.0; its additional `cmmi12.pfb` and
`cmr12.pfb` fonts were fetched into the existing local cache. Cached rebuilds
then work normally. Candidate outputs and page renders for this revision are
in `tmp/pdfs/simplified_lf_revision/`. Use that output directory in the
commands above, and pass `--out tmp/pdfs/simplified_lf_revision/pages` to the
renderer. The installed MiKTeX `latex`/`dvipng` tools now work, so the two
pedagogical figures use LaTeX rather than the original STIX fallback.
