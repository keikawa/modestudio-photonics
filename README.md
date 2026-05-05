# ModeStudio Photonics

ModeStudio Photonics is a desktop application for waveguide mode analysis in integrated photonics.

It runs locally on your computer and provides a graphical interface for loading cross-section geometry, running mode analysis, visualizing mode fields, and exporting results.

<img width="1859" height="1173" alt="image" src="https://github.com/user-attachments/assets/2342b8c2-5ca1-4a26-838b-444f2fa0d930" />

<br>

<img width="1856" height="1181" alt="image" src="https://github.com/user-attachments/assets/57acdde7-6056-4364-bef0-6cb241d8cd86" />

<br>

<img width="1861" height="1168" alt="image" src="https://github.com/user-attachments/assets/12d34e48-868e-417c-8453-ff0d7605bcd3" />


## Installation

Prebuilt application packages are available from the GitHub Releases page.

Download the latest release for your operating system, extract or install it, and launch ModeStudio.

## Quick start

1. Open ModeStudio.

1. Load a Section JSON file and a Materials JSON file. Sample files are included in the repository under `python_backend/`.

1. After loading the files, ModeStudio displays the cross-section model and initializes analysis settings such as reference regions, analysis window, and mesh preset.

1. Click `Run analysis` to calculate waveguide modes.

1. The calculated modes are shown in the results table. You can inspect mode fields, effective indices, TE/TM fractions, propagation loss, and reference-region power fractions.

## License

ModeStudio is licensed under the GNU General Public License v3.0 or later.

```text
SPDX-License-Identifier: GPL-3.0-or-later
```

Copyright (C) 2026 Keisuke Kawahara.

The full GPLv3 license text is available in `LICENSE`.

ModeStudio uses third-party open-source software. See `THIRD_PARTY_NOTICES.md` for dependency license notes.
