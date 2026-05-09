# ModeStudio

ModeStudio is an open-source waveguide mode analysis tool.

It provides a GUI for loading waveguide models from JSON files, running mode analysis, and visualizing fields.

<img width="1859" height="1173" alt="ModeStudio cross-section view" src="https://github.com/user-attachments/assets/2342b8c2-5ca1-4a26-838b-444f2fa0d930" />

<br>

<img width="1856" height="1181" alt="ModeStudio mode-field view" src="https://github.com/user-attachments/assets/57acdde7-6056-4364-bef0-6cb241d8cd86" />

<br>

<img width="1861" height="1168" alt="ModeStudio wavelength-sweep view" src="https://github.com/user-attachments/assets/12d34e48-868e-417c-8453-ff0d7605bcd3" />

## Download

→ See [releases](https://github.com/keikawa/modestudio-photonics/releases).

## Quick start

1. Open ModeStudio.

2. Load a Section JSON file and a Materials JSON file. Sample files are included in the repository under `python_backend/`.

3. Check the cross-section view. ModeStudio displays the loaded regions, assigned materials, and current simulation domain.

4. Select one or more reference regions. For a simple SOI waveguide, this is usually the core region.

5. Adjust the analysis domain, wavelength, number of modes, solver order, and mesh setting.

6. Click `Run mode analysis`.

## Input files

ModeStudio uses two JSON files as model inputs. The Section JSON file describes the cross-section geometry, and the Materials JSON file describes refractive indices.

### Section JSON

The Section JSON file contains a `regions` array. Each region has a name, material name, mesh order (optional), and one or more polygons.

```json
{
  "unit": "um",
  "regions": [
    {
      "name": "core",
      "material": "si",
      "mesh_order": 1,
      "polygons": [
        {
          "hull": [[5.75, 2.00], [5.75, 2.22], [6.25, 2.22], [6.25, 2.00]],
          "holes": []
        }
      ]
    }
  ]
}
```

| Field | Description |
|:--|:--|
| `unit` | Must currently be `"um"`. |
| `regions` | A non-empty array of region definitions. |
| `name` | Region name shown in the GUI and used for reference-region selection. |
| `material` | Material key that must be defined in the Materials JSON file. |
| `mesh_order` | Optional priority value used when resolving overlaps. Smaller values have higher priority. |
| `polygons` | Array of polygon objects. Each polygon has a `hull` and optional `holes`. |

If regions overlap, ModeStudio resolves the overlaps before mesh generation. Regions with smaller `mesh_order` values have higher priority. Omitted `mesh_order` values are treated as `2`. For regions with the same `mesh_order`, later entries in the Section JSON have higher priority.

### Materials JSON

The Materials JSON file contains refractive-index definitions. Each material must have the real and imaginary parts of the refractive index.

```json
{
  "materials": {
    "si": {
      "label": "Silicon",
      "n": {
        "real": 3.4777,
        "imag": 0.0
      }
    },
    "sio2": {
      "label": "Silicon Dioxide",
      "n": {
        "real": 1.444,
        "imag": 0.0
      }
    }
  }
}
```

| Field | Description |
|:--|:--|
| `materials` | Object keyed by material names. |
| `label` | Optional display label. |
| `n.real` | Real part of the refractive index. |
| `n.imag` | Imaginary part of the refractive index. |

Material names in the Section JSON must match material keys in the Materials JSON file. The sample files use lowercase names such as `si` and `sio2`.

## Analysis settings

The left pane contains the main settings used to prepare and run the mode analysis.

| Section | Setting                              | Description                                                  |
| :------ | :----------------------------------- | :----------------------------------------------------------- |
| Domain  | Reference regions                    | Regions used as the main target of the analysis. They are used for fitting the simulation domain, refining the mesh, and calculating the reference-region power fraction. |
| Domain  | Fit to reference regions             | Updates `x min`, `x max`, `y min`, and `y max` from the selected reference regions. |
| Domain  | `x min`, `x max`, `y min`, `y max`   | Bounds of the simulation domain in micrometers.              |
| Domain  | Background material                  | Material assigned to empty areas inside the simulation domain. An existing material or a user-defined refractive index can be used. |
| Solver  | Wavelength mode                      | Selects either a single-wavelength calculation or a wavelength sweep. |
| Solver  | Start, Stop, Points                  | Wavelength range and number of sample points for sweep calculations. |
| Solver  | Modes                                | Number of modes to calculate.                                |
| Solver  | Order (`1` or `2`)                   | Finite-element order used by the mode solver.                |
| Mesh    | `Coarse`, `Normal`, `Fine`,  `Ultra` | Mesh resolution level used to compute mesh sizes from the reference regions and simulation domain. |

## Results

The right pane shows the model preview and analysis results.

| View                  | Description                                                  |
| :-------------------- | :----------------------------------------------------------- |
| `Cross-section model` | Shows the loaded cross-section, region labels, material assignment, current simulation domain, and tables of materials and regions. |
| `Mode field`          | Shows the selected mode field on the cross-section. The field quantity can be selected from `|E|^2`, electric-field components, and magnetic-field components, and the display scale can be switched between `linear` and `log`.<br />The `Modes` table includes `Mode`, `n_eff`, `Im(n_eff)`, `Loss [dB/cm]`, `TE`, `TM`, and `P_ref`; `n_g` is also shown when available. |
| `Wavelength sweep`    | Plots the selected graph quantity as a function of wavelength. Modes checked in the table are drawn in the sweep graph, and the modes at the selected wavelength are shown below the graph. |

## Export

ModeStudio can export results and working state from the GUI.

| Export | File name | Description |
|:--|:--|:--|
| Mode table | `modes.csv` | CSV table of the currently displayed mode results. |
| Mode metadata | `modes_metadata.json` | JSON metadata containing the table, model input, analysis settings, and results. |
| Sweep table | `wavelength_sweep.csv` | CSV table of wavelength-sweep results. |
| Sweep metadata | `wavelength_sweep_metadata.json` | JSON metadata for wavelength-sweep results. |
| Field data | `mode_<index>_<field>_field_data.json` | JSON data for the selected mode-field map. |
| Python script bundle | `<project>_python_export.zip` | Reusable Python script bundle for rerunning the current analysis. |
| Project archive | `<project>.modestudio.zip` | Project archive containing model files, UI state, and optionally analysis output. |

The Python script bundle contains `run_analysis.py` and the files needed to reproduce the current GUI analysis. The generated script contains `SECTION_DATA`, `MATERIALS_DATA`, and `BASE_CONFIG`, so it can be edited after export.

The project archive is used to save and reopen a ModeStudio session. It contains `project.json`, `section.json`, and `materials.json`, and also includes `analysis_output.json` when analysis results are available.

## License

```text
SPDX-License-Identifier: GPL-3.0-or-later
```

See `THIRD_PARTY_NOTICES.md` for dependency license notes.

Copyright (C) 2026 Keisuke Kawahara.
