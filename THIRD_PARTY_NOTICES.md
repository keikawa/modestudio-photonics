# Third-party notices

ModeStudio Photonics uses third-party open-source software.

This file is a starting point for dependency license notices. For each release, review the exact dependency versions used in `requirements.txt`, `package.json`, and `package-lock.json`.

## Important dependencies

| Component | Role | License note |
| --- | --- | --- |
| FEMWELL | Mode-analysis backend | GPLv3-family license. Review the exact version used for each release. |
| gmsh | Meshing backend | GPL-family license, with separate commercial licensing available from the project. Review the exact distribution form. |
| Streamlit | Local web GUI framework | Apache-2.0. |
| scikit-fem | Finite-element operations | BSD-family license. |
| Shapely | Geometry processing | BSD-family license. |
| NumPy | Numerical arrays | BSD-family license. |
| SciPy | Scientific computing | BSD-family license. |
| pandas | Data tables and CSV handling | BSD-family license. |
| Plotly | Interactive plotting | MIT-family license. |
| meshio | Mesh/data conversion | MIT-family license. |
| Electron | Desktop application wrapper | MIT-family license. |

## Redistribution note

If ModeStudio Photonics is redistributed as source only, keep the ModeStudio Photonics license, this notice file, and dependency declarations.

If ModeStudio Photonics is redistributed as a binary, installer, PyInstaller bundle, Docker image, or other packaged artifact that includes third-party components, include the applicable third-party license texts and notices for the bundled components.

For release packaging, consider adding a `licenses/` directory containing the license text for each bundled component and the exact version list used to build the release.
