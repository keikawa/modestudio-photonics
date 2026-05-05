# ModeStudio Photonics

ModeStudio Photonics is a desktop application for waveguide mode analysis in integrated photonics.

It runs locally on your computer and provides a graphical interface for loading cross-section geometry, running mode analysis, visualizing mode fields, and exporting results.

## Installation

Prebuilt application packages are available from the GitHub Releases page.

Download the latest release for your operating system, extract or install it, and launch ModeStudio.

## Quick start

Open ModeStudio.

Load a Section JSON file and a Materials JSON file. Sample files are included in the repository under `python_backend/`.

After loading the files, ModeStudio displays the cross-section model and initializes analysis settings such as reference regions, analysis window, and mesh preset.

Click `Run analysis` to calculate waveguide modes.

The calculated modes are shown in the results table. You can inspect mode fields, effective indices, TE/TM fractions, propagation loss, and reference-region power fractions.

## License

ModeStudio is licensed under the GNU General Public License v3.0 or later.

```text
SPDX-License-Identifier: GPL-3.0-or-later
```

Copyright (C) 2026 Keisuke Kawahara.

The full GPLv3 license text is available in `LICENSE`.

ModeStudio uses third-party open-source software. See `THIRD_PARTY_NOTICES.md` for dependency license notes.

## Disclaimer

ModeStudio is provided as-is, without warranty. Users are responsible for validating simulation settings and results for their own applications.
