from __future__ import annotations

import hashlib
from datetime import datetime, timezone
import html
import json
import math
import re
import subprocess
import sys
import tempfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from shapely.ops import unary_union
import streamlit as st
import streamlit.components.v1 as components
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile

from engine import (
    DEFAULT_EMPTY_AREA_MATERIAL,
    InputValidationError,
    apply_analysis_window,
    auto_domain_bounds,
    build_shapes,
    get_geometry_bounds,
    loads_json,
    suggest_focus_region_names,
)

APP_DIR = Path(__file__).resolve().parent
SECTION_SAMPLE_PATH = APP_DIR / 'section_sample.json'
MATERIALS_SAMPLE_PATH = APP_DIR / 'materials_sample.json'
SOLVER_RUNNER_PATH = APP_DIR / 'solver_runner.py'
SOLVER_CLI_FLAG = '--modestudio-solver-runner'
PROJECT_SCHEMA_VERSION = 2
USER_DEFINED_EMPTY_AREA_LABEL = 'User-defined'
CROSS_SECTION_VIEW_LABEL = 'Cross-section model'
MODE_FIELD_VIEW_LABEL = 'Mode field'
WAVELENGTH_SWEEP_VIEW_LABEL = 'Wavelength sweep'
APP_VERSION = 'v118'

FIELD_QUANTITY_LABELS = {
    # Streamlit selectbox choices are plain text, not Markdown/HTML.
    # Keep them readable and stable rather than forcing partial math styling.
    'intensity': '|E|^2',
    'abs_Ex': '|E_x|',
    'abs_Ey': '|E_y|',
    'abs_Ez': '|E_z|',
    'abs_Hx': '|H_x|',
    'abs_Hy': '|H_y|',
    'abs_Hz': '|H_z|',
}
FIELD_QUANTITY_ORDER = ['intensity', 'abs_Ex', 'abs_Ey', 'abs_Ez', 'abs_Hx', 'abs_Hy', 'abs_Hz']
FIELD_QUANTITY_OPTIONS = [FIELD_QUANTITY_LABELS[key] for key in FIELD_QUANTITY_ORDER]
FIELD_LABEL_TO_KEY = {label: key for key, label in FIELD_QUANTITY_LABELS.items()}


PHYS_LABEL_HTML = {
    'wavelength_nm': '<i>λ</i> [nm]',
    'n_eff': '<i>n</i><sub>eff</sub>',
    'group_index': '<i>n</i><sub>g</sub>',
    'k_eff': 'Im(<i>n</i><sub>eff</sub>)',
    'loss_dB_per_cm': 'Loss [dB cm<sup>−1</sup>]',
    'TE_fraction': 'TE',
    'TM_fraction': 'TM',
    'reference_power_fraction': '<i>P</i><sub>ref</sub>',
}

PHYS_LABEL_PLAIN = {
    # Streamlit native widgets do not render Markdown, LaTeX, or HTML labels.
    # Keep native widget labels as plain text. Plotly/HTML-rendered areas use
    # PHYS_LABEL_HTML for proper subscript/superscript styling.
    'n_eff': 'n_eff',
    'group_index': 'n_g',
    'k_eff': 'Im(n_eff)',
    'loss_dB_per_cm': 'Loss [dB/cm]',
    'reference_power_fraction': 'P_ref',
}


FEMWELL_INFERNO_COLORSCALE = [
    [0.0000, 'rgb(0, 0, 4)'],
    [0.0625, 'rgb(12, 7, 44)'],
    [0.1250, 'rgb(32, 12, 75)'],
    [0.1875, 'rgb(59, 15, 112)'],
    [0.2500, 'rgb(87, 15, 109)'],
    [0.3125, 'rgb(114, 31, 94)'],
    [0.3750, 'rgb(141, 44, 73)'],
    [0.4375, 'rgb(168, 55, 55)'],
    [0.5000, 'rgb(193, 70, 38)'],
    [0.5625, 'rgb(215, 89, 20)'],
    [0.6250, 'rgb(232, 113, 6)'],
    [0.6875, 'rgb(244, 140, 9)'],
    [0.7500, 'rgb(250, 171, 21)'],
    [0.8125, 'rgb(252, 202, 42)'],
    [0.8750, 'rgb(248, 232, 77)'],
    [0.9375, 'rgb(244, 249, 135)'],
    [1.0000, 'rgb(252, 255, 164)'],
]

REGION_COLOR_PALETTE = [
    '#38bdf8', '#f59e0b', '#22c55e', '#fb7185', '#a78bfa', '#94a3b8', '#14b8a6', '#f97316'
]

MESH_PRESETS = {
    'Coarse': {
        'selected_divisions': 3.5,
        'surrounding_ratio': 5.0,
        'refined_distance_ratio': 2.6,
        'surrounding_distance_ratio': 2.4,
        'default_max_ratio': 2.0,
    },
    'Normal': {
        'selected_divisions': 6.0,
        'surrounding_ratio': 5.5,
        'refined_distance_ratio': 3.6,
        'surrounding_distance_ratio': 3.4,
        'default_max_ratio': 2.2,
    },
    'Fine': {
        'selected_divisions': 9.0,
        'surrounding_ratio': 5.5,
        'refined_distance_ratio': 4.8,
        'surrounding_distance_ratio': 4.2,
        'default_max_ratio': 2.4,
    },
    'Ultra': {
        'selected_divisions': 12.0,
        'surrounding_ratio': 5.0,
        'refined_distance_ratio': 5.2,
        'surrounding_distance_ratio': 4.6,
        'default_max_ratio': 2.2,
    },
}

def solver_subprocess_command(
    section_path: Path,
    materials_path: Path,
    config_path: Path,
    output_path: Path,
) -> list[str]:
    """Return the command used to run the solver in a separate process.

    In normal Python execution, solver_runner.py is launched with the current
    Python interpreter.  In a PyInstaller build, sys.executable points to the
    ModeStudio backend executable, not python.exe.  The backend launcher must
    therefore recognize SOLVER_CLI_FLAG and run solver_runner.main() instead of
    starting Streamlit again.
    """
    if getattr(sys, 'frozen', False):
        return [
            sys.executable,
            SOLVER_CLI_FLAG,
            str(section_path),
            str(materials_path),
            str(config_path),
            str(output_path),
        ]
    return [
        sys.executable,
        str(SOLVER_RUNNER_PATH),
        str(section_path),
        str(materials_path),
        str(config_path),
        str(output_path),
    ]


def run_solver_subprocess(section_data: dict, materials_data: dict, config: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix='streamlit_femwell_') as tmpdir:
        tmpdir_path = Path(tmpdir)
        section_path = tmpdir_path / 'section.json'
        materials_path = tmpdir_path / 'materials.json'
        config_path = tmpdir_path / 'config.json'
        output_path = tmpdir_path / 'output.json'

        section_path.write_text(json.dumps(section_data, ensure_ascii=False), encoding='utf-8')
        materials_path.write_text(json.dumps(materials_data, ensure_ascii=False), encoding='utf-8')
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')

        completed = subprocess.run(
            solver_subprocess_command(section_path, materials_path, config_path, output_path),
            capture_output=True,
            text=True,
            cwd=str(APP_DIR),
        )

        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or 'Solver subprocess failed.'
            raise RuntimeError(message)
        if not output_path.exists():
            raise RuntimeError('Solver subprocess did not create output.json.')

        payload = json.loads(output_path.read_text(encoding='utf-8'))
        if not payload.get('ok'):
            raise RuntimeError(payload.get('error', 'Solver subprocess failed.'))

        return {
            'results': payload['results'],
            'mode_field_maps': payload.get('mode_field_maps', []),
            'sweep_field_maps': payload.get('sweep_field_maps', []),
        }

def load_default_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def decode_upload(uploaded_file) -> tuple[str, str]:
    raw = uploaded_file.getvalue()
    token = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8'), token


def upload_token_from_state(key: str) -> str | None:
    uploaded_file = st.session_state.get(key)
    if uploaded_file is None:
        return None
    try:
        return hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    except Exception:
        return None


def reset_domain_state() -> None:
    for key in ('manual_left', 'manual_right', 'manual_bottom', 'manual_top', 'focus_regions'):
        st.session_state.pop(key, None)


def init_state() -> None:
    if st.session_state.get('app_version') != APP_VERSION:
        for key in ('sweep_metric_key', 'sweep_metric_label'):
            st.session_state.pop(key, None)
        st.session_state['app_version'] = APP_VERSION

    defaults = {
        'section_text': load_default_text(SECTION_SAMPLE_PATH),
        'materials_text': load_default_text(MATERIALS_SAMPLE_PATH),
        'analysis_output': None,
        'analysis_error': None,
        'section_upload_token': None,
        'materials_upload_token': None,
        'section_file_label': 'section_sample.json',
        'materials_file_label': 'materials_sample.json',
        'empty_area_n_real': 1.444,
        'empty_area_k': 0.0,
        'project_name': 'sample_project',
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def table_height(row_count: int, *, max_rows: int = 8) -> int:
    visible_rows = max(1, min(row_count, max_rows))
    return 38 + 35 * visible_rows + 4



def _round_mesh_value(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 0.05
    exponent = math.floor(math.log10(value))
    scaled = value / (10 ** exponent)
    for mantissa in (1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0):
        if scaled <= mantissa:
            return float(mantissa * (10 ** exponent))
    return float(10 ** (exponent + 1))


def _shape_feature_size(shapes: dict, region_names: list[str] | tuple[str, ...] | None) -> float:
    selected_names = [name for name in (region_names or []) if name in shapes]
    selected_shapes = [shapes[name] for name in selected_names] if selected_names else list(shapes.values())
    if not selected_shapes:
        return 0.1

    # Estimate the mesh feature size from the union of the selected reference
    # regions, not from each JSON region separately.  This keeps the automatic
    # mesh preset stable when one material/structure is split into multiple
    # adjacent regions in the Section JSON.  Disconnected islands remain separate
    # parts of a MultiPolygon and are still checked independently.
    reference_shape = unary_union(selected_shapes)

    dimensions: list[float] = []
    for geom in _iter_geoms(reference_shape):
        if geom.is_empty:
            continue
        left, bottom, right, top = geom.bounds
        width = float(right - left)
        height = float(top - bottom)
        if width > 1e-9:
            dimensions.append(width)
        if height > 1e-9:
            dimensions.append(height)
    if dimensions:
        return min(dimensions)
    return 0.1


def _domain_span(bounds: dict[str, float] | None, shapes: dict | None) -> float:
    if bounds:
        width = abs(float(bounds['right']) - float(bounds['left']))
        height = abs(float(bounds['top']) - float(bounds['bottom']))
    elif shapes:
        geometry_bounds = get_geometry_bounds(shapes)
        width = abs(float(geometry_bounds['right']) - float(geometry_bounds['left']))
        height = abs(float(geometry_bounds['top']) - float(geometry_bounds['bottom']))
    else:
        width = height = 1.0
    positive = [value for value in (width, height) if value > 1e-9]
    return min(positive) if positive else 1.0


def build_mesh_preset_config(
    preset: str,
    *,
    shapes: dict | None,
    focus_regions: list[str] | tuple[str, ...] | None,
    domain_bounds: dict[str, float] | None,
) -> dict[str, float]:
    factors = MESH_PRESETS[preset]
    feature = _shape_feature_size(shapes or {}, focus_regions)
    span = _domain_span(domain_bounds, shapes)

    refined = feature / float(factors['selected_divisions'])
    refined = max(0.003, min(refined, max(span / 18.0, 0.003)))
    refined = _round_mesh_value(refined)

    surrounding = refined * float(factors['surrounding_ratio'])
    surrounding = max(refined * 3.0, min(surrounding, max(span / 4.0, refined * 3.0)))
    surrounding = _round_mesh_value(surrounding)

    size_max = surrounding * float(factors['default_max_ratio'])
    size_max = max(surrounding, min(size_max, max(span / 2.0, surrounding)))
    size_max = _round_mesh_value(size_max)

    refined_distance = _round_mesh_value(max(refined, refined * float(factors['refined_distance_ratio'])))
    surrounding_distance = _round_mesh_value(max(refined, refined * float(factors['surrounding_distance_ratio'])))

    return {
        'refined_resolution': float(refined),
        'refined_distance': float(refined_distance),
        'surrounding_resolution': float(surrounding),
        'surrounding_distance': float(surrounding_distance),
        'default_resolution_max': float(size_max),
        'feature_size': float(feature),
    }


def build_materials_dataframe(materials_data: dict, empty_choice: str, empty_n_real: float, empty_k: float) -> pd.DataFrame:
    rows = []
    for name, entry in materials_data['materials'].items():
        rows.append(
            {
                'Material': name,
                'Material Label': entry.get('label', ''),
                'n': float(entry['n']['real']),
                'k': float(entry['n']['imag']),
            }
        )
    if empty_choice == USER_DEFINED_EMPTY_AREA_LABEL:
        rows.append(
            {
                'Material': DEFAULT_EMPTY_AREA_MATERIAL,
                'Material Label': 'User-defined empty area',
                'n': float(empty_n_real),
                'k': float(empty_k),
            }
        )
    return pd.DataFrame(rows)


def build_regions_dataframe(shapes: dict, region_materials: dict[str, str]) -> pd.DataFrame:
    rows = []
    for name, shape in shapes.items():
        left, bottom, right, top = shape.bounds
        width = max(0.0, float(right) - float(left))
        height = max(0.0, float(top) - float(bottom))
        rows.append(
            {
                'Region': name,
                'Material': region_materials.get(name, ''),
                'Size [um × um]': f'{width:.4g} × {height:.4g}',
                'Area [um²]': float(shape.area),
            }
        )
    return pd.DataFrame(rows)



def build_modes_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for mode in results.get('modes', []):
        wavelength_um = float(results.get('wavelength_um', float(results.get('wavelength_nm', 1550.0)) / 1000.0))
        n_eff_imag = float(mode['n_eff']['imag'])
        if 'propagation_loss_dB_per_cm' in mode:
            propagation_loss = float(mode['propagation_loss_dB_per_cm'])
        elif wavelength_um > 0.0:
            propagation_loss = float(20.0 * math.log10(math.e) * (2.0 * math.pi / wavelength_um) * abs(n_eff_imag) * 1.0e4)
        else:
            propagation_loss = float('nan')
        row = {
            'mode': int(mode['mode_index']),
            'wavelength_nm': float(results.get('wavelength_nm', wavelength_um * 1000.0)),
            'n_eff': float(mode['n_eff']['real']),
            'k_eff': n_eff_imag,
            'loss_dB_per_cm': propagation_loss,
            'TE_fraction': float(mode['te_fraction']),
            'TM_fraction': float(mode['tm_fraction']),
        }
        if 'group_index' in mode:
            try:
                row['group_index'] = float(mode['group_index'])
            except Exception:
                row['group_index'] = float('nan')
        if 'power_reference_fraction' in mode:
            row['reference_power_fraction'] = float(mode['power_reference_fraction']['real'])
        elif 'confinement_focus' in mode:
            # Backward-compatible fallback for older result payloads. New runs use
            # power_reference_fraction, not Femwell's calculate_confinement_factor.
            row['reference_power_fraction'] = float(mode['confinement_focus']['real'])
        rows.append(row)
    return pd.DataFrame(rows)


def build_modes_display_dataframe(results: dict) -> pd.DataFrame:
    modes_df = build_modes_dataframe(results)
    if modes_df.empty:
        return modes_df

    rename_map = {
        'mode': 'Mode',
        'n_eff': 'n_eff',
        'k_eff': 'Im(n_eff)',
        'loss_dB_per_cm': 'Loss [dB/cm]',
        'group_index': 'n_g',
        'TE_fraction': 'TE',
        'TM_fraction': 'TM',
        'reference_power_fraction': 'P_ref',
    }
    columns = ['mode', 'n_eff', 'k_eff', 'loss_dB_per_cm']
    if 'group_index' in modes_df.columns:
        columns.append('group_index')
    columns += ['TE_fraction', 'TM_fraction']
    if 'reference_power_fraction' in modes_df.columns:
        columns.append('reference_power_fraction')
    return modes_df[columns].rename(columns=rename_map)



def _format_float(value: Any, fmt: str) -> str:
    try:
        number = float(value)
    except Exception:
        return '—'
    if not math.isfinite(number):
        return '—'
    return format(number, fmt)


def render_modes_cards(results: dict) -> pd.DataFrame:
    modes_df = build_modes_dataframe(results)
    if modes_df.empty:
        st.caption('No mode result is available.')
        return modes_df

    cards: list[str] = ['<div class="modes-list">']
    has_pref = 'reference_power_fraction' in modes_df.columns
    for row in modes_df.to_dict('records'):
        mode = int(row['mode'])
        stats = [
            (PHYS_LABEL_HTML['k_eff'], _format_float(row.get('k_eff'), '.3e')),
            (PHYS_LABEL_HTML['loss_dB_per_cm'], _format_float(row.get('loss_dB_per_cm'), '.3g')),
            ('TE', _format_float(row.get('TE_fraction'), '.4f')),
            ('TM', _format_float(row.get('TM_fraction'), '.4f')),
        ]
        if has_pref:
            stats.append((PHYS_LABEL_HTML['reference_power_fraction'], _format_float(row.get('reference_power_fraction'), '.4f')))
        else:
            stats.append((PHYS_LABEL_HTML['wavelength_nm'], _format_float(row.get('wavelength_nm'), '.1f'))) 
        stats_html = ''.join(
            '<div><span class="mode-stat-label">{label}</span><span class="mode-stat-value">{value}</span></div>'.format(
                label=label,
                value=html.escape(value),
            )
            for label, value in stats
        )
        cards.append(
            '<div class="mode-card">'
            '<div class="mode-card-head">'
            f'<span class="mode-index">Mode {mode}</span>'
            f'<span class="mode-neff"><i>n</i><sub>eff</sub> {html.escape(_format_float(row.get("n_eff"), ".6f"))}</span>'
            '</div>'
            f'<div class="mode-grid">{stats_html}</div>'
            '</div>'
        )
    cards.append('</div>')
    st.markdown(''.join(cards), unsafe_allow_html=True)
    return modes_df


def render_modes_table(results: dict, *, selected_mode: int | None = None) -> pd.DataFrame:
    modes_df = build_modes_dataframe(results)
    if modes_df.empty:
        st.caption('No mode result is available.')
        return modes_df

    has_pref = 'reference_power_fraction' in modes_df.columns
    has_group_index = 'group_index' in modes_df.columns
    headers = ['Mode', PHYS_LABEL_HTML['n_eff'], PHYS_LABEL_HTML['k_eff'], PHYS_LABEL_HTML['loss_dB_per_cm']] + ([PHYS_LABEL_HTML['group_index']] if has_group_index else []) + ['TE', 'TM'] + ([PHYS_LABEL_HTML['reference_power_fraction']] if has_pref else [])
    header_html = ''.join(f'<th>{label}</th>' for label in headers)
    rows_html: list[str] = []
    for row in modes_df.to_dict('records'):
        mode = int(row.get('mode', 0))
        cells = [
            f'Mode {mode}',
            _format_float(row.get('n_eff'), '.6f'),
            _format_float(row.get('k_eff'), '.3e'),
            _format_float(row.get('loss_dB_per_cm'), '.3g'),
        ]
        if has_group_index:
            cells.append(_format_float(row.get('group_index'), '.6f'))
        cells += [
            _format_float(row.get('TE_fraction'), '.4f'),
            _format_float(row.get('TM_fraction'), '.4f'),
        ]
        if has_pref:
            cells.append(_format_float(row.get('reference_power_fraction'), '.4f'))
        cls = ' class="active-mode"' if selected_mode is not None and mode == int(selected_mode) else ''
        rows_html.append('<tr{cls}>{cells}</tr>'.format(
            cls=cls,
            cells=''.join(f'<td>{html.escape(value)}</td>' for value in cells),
        ))
    st.markdown(
        '<div class="modes-table-wrap"><table class="modes-table"><thead><tr>'
        + header_html
        + '</tr></thead><tbody>'
        + ''.join(rows_html)
        + '</tbody></table></div>',
        unsafe_allow_html=True,
    )
    return modes_df


def _iter_geoms(shape: object):
    geoms = getattr(shape, 'geoms', None)
    return list(geoms) if geoms is not None else [shape]


def add_polygon_traces(fig: go.Figure, shapes: dict, region_materials: dict[str, str], *, opacity: float) -> None:
    for name, shape in shapes.items():
        material = region_materials.get(name, '')
        first_trace = True
        for geom in _iter_geoms(shape):
            if geom.is_empty:
                continue
            x, y = geom.exterior.xy
            hover = f'region: {name}<br>material: {material}<br>area: {float(geom.area):.6g} um^2'
            fig.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(y),
                    mode='none',
                    fill='toself',
                    name=name if first_trace else f'{name} part',
                    legendgroup=name,
                    hovertemplate=hover + '<extra></extra>',
                    opacity=opacity,
                )
            )
            first_trace = False
            for interior in geom.interiors:
                ix, iy = interior.xy
                fig.add_trace(
                    go.Scatter(
                        x=list(ix),
                        y=list(iy),
                        mode='lines',
                        name=f'{name} hole',
                        legendgroup=name,
                        showlegend=False,
                        hovertemplate=hover + '<extra></extra>',
                    )
                )


def add_domain_trace(fig: go.Figure, window: dict[str, float]) -> None:
    if not window:
        return
    x0, x1 = window['left'], window['right']
    y0, y1 = window['bottom'], window['top']
    fig.add_trace(
        go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[y0, y0, y1, y1, y0],
            mode='lines',
            name='simulation domain',
            line={'dash': 'dash', 'width': 3},
            hovertemplate='simulation domain<br>x: %{x:.4g} um<br>y: %{y:.4g} um<extra></extra>',
        )
    )


def geometry_figure(
    shapes: dict,
    region_materials: dict[str, str],
    *,
    simulation_domain: dict[str, float] | None = None,
) -> go.Figure:
    fig = go.Figure()
    add_polygon_traces(fig, shapes, region_materials, opacity=0.42)
    if simulation_domain:
        add_domain_trace(fig, simulation_domain)

    if shapes:
        bounds = get_geometry_bounds(shapes)
        left, bottom, right, top = bounds['left'], bounds['bottom'], bounds['right'], bounds['top']
        if simulation_domain:
            left = min(left, simulation_domain['left'])
            right = max(right, simulation_domain['right'])
            bottom = min(bottom, simulation_domain['bottom'])
            top = max(top, simulation_domain['top'])
        dx = max(right - left, 1e-9)
        dy = max(top - bottom, 1e-9)
        pad = 0.05 * max(dx, dy)
        fig.update_xaxes(range=[left - pad, right + pad])
        fig.update_yaxes(range=[bottom - pad, top + pad])

    fig.update_layout(
        height=500,
        margin={'l': 26, 'r': 22, 't': 16, 'b': 24},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0, 'font': {'family': 'Arial, Helvetica, sans-serif', 'size': 14, 'color': '#111111'}},
        hovermode='closest',
        dragmode='pan',
        uirevision='cross-section-model',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'family': 'Arial, Helvetica, sans-serif', 'size': 15, 'color': '#111111'},
    )
    fig.update_xaxes(title='x [um]', zeroline=False, showgrid=True, gridcolor='rgba(17,24,39,0.10)', showline=True, linecolor='rgba(17,24,39,0.72)', linewidth=1, mirror=True, ticks='inside', ticklen=6, tickwidth=1, tickcolor='rgba(17,24,39,0.72)', title_font={'family': 'Arial, Helvetica, sans-serif', 'size': 16, 'color': '#111111'}, tickfont={'family': 'Arial, Helvetica, sans-serif', 'size': 14, 'color': '#111111'})
    fig.update_yaxes(title='y [um]', scaleanchor='x', scaleratio=1, zeroline=False, showgrid=True, gridcolor='rgba(17,24,39,0.10)', showline=True, linecolor='rgba(17,24,39,0.72)', linewidth=1, mirror=True, ticks='inside', ticklen=6, tickwidth=1, tickcolor='rgba(17,24,39,0.72)', title_font={'family': 'Arial, Helvetica, sans-serif', 'size': 16, 'color': '#111111'}, tickfont={'family': 'Arial, Helvetica, sans-serif', 'size': 14, 'color': '#111111'})
    return fig


def add_region_boundary_traces(fig: go.Figure, shapes: dict, region_materials: dict[str, str] | None = None) -> None:
    for name, shape in shapes.items():
        material = (region_materials or {}).get(name, '')
        for geom in _iter_geoms(shape):
            if geom.is_empty:
                continue
            x, y = geom.exterior.xy
            fig.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(y),
                    mode='lines',
                    name=name,
                    line={'width': 1},
                    showlegend=False,
                    hovertemplate=f'region: {name}<br>material: {material}<extra></extra>',
                )
            )
            for interior in geom.interiors:
                ix, iy = interior.xy
                fig.add_trace(
                    go.Scatter(
                        x=list(ix),
                        y=list(iy),
                        mode='lines',
                        name=f'{name} hole',
                        line={'width': 1},
                        showlegend=False,
                        hovertemplate=f'region: {name}<br>material: {material}<extra></extra>',
                    )
                )




def _parse_rgb_string(color: str) -> tuple[int, int, int]:
    prefix = 'rgb('
    if not color.startswith(prefix) or not color.endswith(')'):
        return (0, 0, 0)
    parts = color[len(prefix):-1].split(',')
    return tuple(max(0, min(255, int(part.strip()))) for part in parts[:3])  # type: ignore[return-value]


def inferno_color(value: float) -> str:
    t = max(0.0, min(1.0, float(value)))
    for idx in range(len(FEMWELL_INFERNO_COLORSCALE) - 1):
        t0, c0 = FEMWELL_INFERNO_COLORSCALE[idx]
        t1, c1 = FEMWELL_INFERNO_COLORSCALE[idx + 1]
        if t <= t1:
            r0, g0, b0 = _parse_rgb_string(c0)
            r1, g1, b1 = _parse_rgb_string(c1)
            alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            r = round(r0 + alpha * (r1 - r0))
            g = round(g0 + alpha * (g1 - g0))
            b = round(b0 + alpha * (b1 - b0))
            return f'rgb({r}, {g}, {b})'
    return FEMWELL_INFERNO_COLORSCALE[-1][1]


def _add_2d_line_with_halo(
    fig: go.Figure,
    *,
    x: list[float],
    y: list[float],
    name: str,
    hovertemplate: str,
    color: str = 'white',
    width: int = 2,
    dash: str | None = None,
    showlegend: bool = False,
) -> None:
    line_base = {'width': width + 2, 'color': 'rgba(0,0,0,0.58)'}
    line_top = {'width': width, 'color': color}
    if dash:
        line_base['dash'] = dash
        line_top['dash'] = dash
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=name,
            line=line_base,
            hoverinfo='skip',
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=name,
            line=line_top,
            hovertemplate=hovertemplate,
            showlegend=showlegend,
        )
    )


def add_region_boundary_traces_for_field(fig: go.Figure, shapes: dict, region_materials: dict[str, str] | None = None) -> None:
    for name, shape in shapes.items():
        material = (region_materials or {}).get(name, '')
        hover = f'region: {name}<br>material: {material}<extra></extra>'
        for geom in _iter_geoms(shape):
            if geom.is_empty:
                continue
            x, y = geom.exterior.xy
            _add_2d_line_with_halo(
                fig,
                x=list(x),
                y=list(y),
                name=name,
                width=2,
                hovertemplate=hover,
            )
            for interior in geom.interiors:
                ix, iy = interior.xy
                _add_2d_line_with_halo(
                    fig,
                    x=list(ix),
                    y=list(iy),
                    name=f'{name} hole',
                    width=2,
                    hovertemplate=hover,
                )


def add_domain_trace_for_field(fig: go.Figure, window: dict[str, float]) -> None:
    if not window:
        return
    x0, x1 = float(window['left']), float(window['right'])
    y0, y1 = float(window['bottom']), float(window['top'])
    _add_2d_line_with_halo(
        fig,
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        name='simulation domain',
        width=2,
        dash='dash',
        hovertemplate='simulation domain<br>x: %{x:.4g} um<br>y: %{y:.4g} um<extra></extra>',
    )


def _field_ranges(field_map: dict, x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    bounds = field_map.get('bounds') or {}
    if bounds:
        x_range = [float(bounds['left']), float(bounds['right'])]
        y_range = [float(bounds['bottom']), float(bounds['top'])]
    elif x and y:
        x_range = [float(min(x)), float(max(x))]
        y_range = [float(min(y)), float(max(y))]
    else:
        x_range = [0.0, 1.0]
        y_range = [0.0, 1.0]

    dx = max(x_range[1] - x_range[0], 1e-9)
    dy = max(y_range[1] - y_range[0], 1e-9)
    pad = 0.04 * max(dx, dy)
    return [x_range[0] - pad, x_range[1] + pad], [y_range[0] - pad, y_range[1] + pad]


def triangular_field_figure(
    field_map: dict,
    *,
    shapes: dict | None = None,
    region_materials: dict[str, str] | None = None,
    height: int = 620,
    ui_revision: str = 'mode-intensity-2d',
) -> go.Figure:
    x = [float(value) for value in field_map.get('x', [])]
    y = [float(value) for value in field_map.get('y', [])]
    values = [float(value) for value in field_map.get('value', [])]
    i_values = [int(value) for value in field_map.get('i', [])]
    j_values = [int(value) for value in field_map.get('j', [])]
    k_values = [int(value) for value in field_map.get('k', [])]

    fig = go.Figure()
    if x and y and values and i_values:
        triangle_count = len(i_values)
        bin_count = 48 if triangle_count <= 1800 else 36 if triangle_count <= 4200 else 24
        grouped: dict[int, tuple[list[float], list[float]]] = {}
        for i_idx, j_idx, k_idx in zip(i_values, j_values, k_values):
            if max(i_idx, j_idx, k_idx) >= len(x) or max(i_idx, j_idx, k_idx) >= len(values):
                continue
            tri_value = (values[i_idx] + values[j_idx] + values[k_idx]) / 3.0
            if not math.isfinite(tri_value):
                continue
            tri_value = max(0.0, min(1.0, tri_value))
            bucket = max(0, min(bin_count - 1, int(round(tri_value * (bin_count - 1)))))
            xs, ys = grouped.setdefault(bucket, ([], []))
            xs.extend([x[i_idx], x[j_idx], x[k_idx], x[i_idx], None])
            ys.extend([y[i_idx], y[j_idx], y[k_idx], y[i_idx], None])

        for bucket in sorted(grouped):
            xs, ys = grouped[bucket]
            value = bucket / max(bin_count - 1, 1)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode='none',
                    fill='toself',
                    fillcolor=inferno_color(value),
                    line={'width': 0, 'color': inferno_color(value)},
                    hoverinfo='skip',
                    showlegend=False,
                    name=field_map.get('quantity', 'field'),
                )
            )

        # Invisible markers only for the colorbar. The actual field is rendered as
        # 2D filled triangles above, so the graph remains a normal 2D Plotly plot.
        fig.add_trace(
            go.Scatter(
                x=[None, None],
                y=[None, None],
                mode='markers',
                marker={
                    'color': [0.0, 1.0],
                    'colorscale': FEMWELL_INFERNO_COLORSCALE,
                    'cmin': 0.0,
                    'cmax': 1.0,
                    'size': 0,
                    'showscale': True,
                    'colorbar': {
                        'title': {'text': field_map.get('z_label', '')},
                        'len': 0.72,
                        'thickness': 16,
                        'outlinewidth': 1,
                        'tickmode': 'array',
                        'tickvals': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    },
                },
                hoverinfo='skip',
                showlegend=False,
            )
        )

    if shapes:
        add_region_boundary_traces_for_field(fig, shapes, region_materials)

    bounds = field_map.get('bounds') or {}
    if bounds:
        add_domain_trace_for_field(fig, bounds)

    x_range, y_range = _field_ranges(field_map, x, y)
    fig.update_layout(
        height=height,
        margin={'l': 24, 'r': 24, 't': 16, 'b': 18},
        paper_bgcolor='white',
        plot_bgcolor='white',
        dragmode='pan',
        hovermode=False,
        uirevision=ui_revision,
        font={'family': 'Arial, Helvetica, sans-serif', 'size': 15, 'color': '#111111'},
    )
    fig.update_xaxes(
        title='x [um]',
        range=x_range,
        zeroline=False,
        showgrid=True,
        gridcolor='rgba(17,24,39,0.10)',
        showline=True,
        linecolor='rgba(17,24,39,0.72)',
        linewidth=1,
        mirror=True,
        ticks='inside',
        ticklen=6,
        tickwidth=1,
        tickcolor='rgba(17,24,39,0.72)',
        title_font={'family': 'Arial, Helvetica, sans-serif', 'size': 16, 'color': '#111111'},
        tickfont={'family': 'Arial, Helvetica, sans-serif', 'size': 14, 'color': '#111111'},
    )
    fig.update_yaxes(
        title='y [um]',
        range=y_range,
        scaleanchor='x',
        scaleratio=1,
        zeroline=False,
        showgrid=True,
        gridcolor='rgba(17,24,39,0.10)',
        showline=True,
        linecolor='rgba(17,24,39,0.72)',
        linewidth=1,
        mirror=True,
        ticks='inside',
        ticklen=6,
        tickwidth=1,
        tickcolor='rgba(17,24,39,0.72)',
        title_font={'family': 'Arial, Helvetica, sans-serif', 'size': 16, 'color': '#111111'},
        tickfont={'family': 'Arial, Helvetica, sans-serif', 'size': 14, 'color': '#111111'},
    )
    return fig


def field_map_figure(
    field_map: dict,
    *,
    shapes: dict | None = None,
    region_materials: dict[str, str] | None = None,
    height: int = 620,
    ui_revision: str = 'mode-intensity-2d',
) -> go.Figure:
    if field_map.get('type') == 'triangular_mesh':
        return triangular_field_figure(
            field_map,
            shapes=shapes,
            region_materials=region_materials,
            height=height,
            ui_revision=ui_revision,
        )

    # Backward-compatible fallback for older result payloads.
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=field_map.get('x', []),
            y=field_map.get('y', []),
            z=field_map.get('z', []),
            colorbar={'title': field_map.get('z_label', '')},
            hovertemplate='x: %{x:.4g} um<br>y: %{y:.4g} um<br>%{z:.4g}<extra></extra>',
        )
    )
    if shapes:
        add_region_boundary_traces(fig, shapes, region_materials)
    fig.update_layout(
        height=height,
        margin={'l': 24, 'r': 24, 't': 16, 'b': 18},
        dragmode='pan',
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'family': 'Arial, Helvetica, sans-serif', 'size': 15, 'color': '#111111'},
    )
    fig.update_xaxes(title='x [um]', zeroline=False, showgrid=True, gridcolor='rgba(17,24,39,0.10)', showline=True, linecolor='rgba(17,24,39,0.72)', linewidth=1, mirror=True, ticks='inside', ticklen=6, tickwidth=1, tickcolor='rgba(17,24,39,0.72)', title_font={'family': 'Arial, Helvetica, sans-serif', 'size': 16, 'color': '#111111'}, tickfont={'family': 'Arial, Helvetica, sans-serif', 'size': 14, 'color': '#111111'})
    fig.update_yaxes(title='y [um]', scaleanchor='x', scaleratio=1, zeroline=False, showgrid=True, gridcolor='rgba(17,24,39,0.10)', showline=True, linecolor='rgba(17,24,39,0.72)', linewidth=1, mirror=True, ticks='inside', ticklen=6, tickwidth=1, tickcolor='rgba(17,24,39,0.72)', title_font={'family': 'Arial, Helvetica, sans-serif', 'size': 16, 'color': '#111111'}, tickfont={'family': 'Arial, Helvetica, sans-serif', 'size': 14, 'color': '#111111'})
    return fig


def shape_parts_for_canvas(shape: object) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for geom in _iter_geoms(shape):
        if geom.is_empty:
            continue
        exterior_x, exterior_y = geom.exterior.xy
        holes: list[list[list[float]]] = []
        for interior in geom.interiors:
            ix, iy = interior.xy
            holes.append([[float(x), float(y)] for x, y in zip(ix, iy)])
        parts.append(
            {
                'exterior': [[float(x), float(y)] for x, y in zip(exterior_x, exterior_y)],
                'holes': holes,
            }
        )
    return parts


def shapes_payload_for_canvas(shapes: dict, region_materials: dict[str, str] | None = None) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for index, (name, shape) in enumerate(shapes.items()):
        try:
            label_point = shape.representative_point()
            label_xy = [float(label_point.x), float(label_point.y)]
        except Exception:
            left, bottom, right, top = shape.bounds
            label_xy = [float((left + right) / 2), float((bottom + top) / 2)]
        payload.append(
            {
                'name': str(name),
                'material': str((region_materials or {}).get(name, '')),
                'index': index,
                'color': REGION_COLOR_PALETTE[index % len(REGION_COLOR_PALETTE)],
                'label': label_xy,
                'parts': shape_parts_for_canvas(shape),
            }
        )
    return payload


def _canvas_bounds_from_geometry(
    shapes: dict | None,
    domain: dict[str, float] | None,
    field_map: dict | None,
) -> dict[str, float]:
    bounds: dict[str, float] | None = None
    if field_map and field_map.get('bounds'):
        fb = field_map['bounds']
        bounds = {
            'left': float(fb['left']),
            'right': float(fb['right']),
            'bottom': float(fb['bottom']),
            'top': float(fb['top']),
        }
    elif domain:
        bounds = {
            'left': float(domain['left']),
            'right': float(domain['right']),
            'bottom': float(domain['bottom']),
            'top': float(domain['top']),
        }
    elif shapes:
        bounds = get_geometry_bounds(shapes)  # type: ignore[arg-type]
    else:
        bounds = {'left': 0.0, 'right': 1.0, 'bottom': 0.0, 'top': 1.0}

    if shapes:
        gb = get_geometry_bounds(shapes)  # type: ignore[arg-type]
        bounds = {
            'left': min(float(bounds['left']), float(gb['left'])),
            'right': max(float(bounds['right']), float(gb['right'])),
            'bottom': min(float(bounds['bottom']), float(gb['bottom'])),
            'top': max(float(bounds['top']), float(gb['top'])),
        }
    return bounds


def canvas_viewer_html(
    *,
    view: str,
    shapes: dict | None,
    region_materials: dict[str, str] | None,
    domain: dict[str, float] | None = None,
    field_map: dict | None = None,
    field_scale: str = 'linear',
    height: int = 650,
) -> str:
    data = {
        'view': view,
        'height': int(height),
        'shapes': shapes_payload_for_canvas(shapes or {}, region_materials),
        'domain': domain or (field_map or {}).get('bounds') or None,
        'field': field_map or None,
        'field_scale': field_scale,
        'bounds': _canvas_bounds_from_geometry(shapes, domain, field_map),
        'colorscale': FEMWELL_INFERNO_COLORSCALE,
    }
    payload = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    return f"""
<div class="ms-canvas-viewer">
  <canvas id="gl-canvas"></canvas>
  <canvas id="ui-canvas"></canvas>
  <button class="viewer-save" type="button" title="Save graph as PNG">Save PNG</button>
  <div class="viewer-legend"></div>
  <div class="viewer-help">wheel: zoom · drag: pan · double click: reset</div>
</div>
<script>
(() => {{
  const DATA = {payload};
  const root = document.currentScript.previousElementSibling;
  const glCanvas = root.querySelector('#gl-canvas');
  const uiCanvas = root.querySelector('#ui-canvas');
  const help = root.querySelector('.viewer-help');
  const legend = root.querySelector('.viewer-legend');
  const saveButton = root.querySelector('.viewer-save');
  const ctx = uiCanvas.getContext('2d');
  const gl = glCanvas.getContext('webgl', {{ antialias: true, alpha: true, preserveDrawingBuffer: true }});
  const state = {{ centerX: 0, centerY: 0, scale: 1, minScale: 1, dragging: false, lastX: 0, lastY: 0 }};
  const GRAPH_FONT = 'Arial, Helvetica, sans-serif';
  const GRAPH_TEXT = 'rgba(0,0,0,0.92)';
  const GRAPH_MUTED = 'rgba(0,0,0,0.74)';
  const GRAPH_AXIS = 'rgba(0,0,0,0.72)';
  const GRAPH_GRID = DATA.view === 'field' ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.12)';
  function graphFont(size, weight = 400) {{
    const dpr = window.devicePixelRatio || 1;
    return String(weight) + ' ' + String(size * dpr) + 'px ' + GRAPH_FONT;
  }}
  let raf = 0;
  let glProgram = null;
  let positionBuffer = null;
  let colorBuffer = null;
  let vertexCount = 0;

  function plotMargins() {{
    const dpr = window.devicePixelRatio || 1;
    return {{
      left: 78 * dpr,
      right: (DATA.view === 'field' ? 144 : 30) * dpr,
      top: 28 * dpr,
      bottom: 74 * dpr,
    }};
  }}
  function plotRect() {{
    const m = plotMargins();
    const x0 = m.left;
    const y0 = m.top;
    const x1 = uiCanvas.width - m.right;
    const y1 = uiCanvas.height - m.bottom;
    return {{ x0, y0, x1, y1, w: Math.max(20, x1 - x0), h: Math.max(20, y1 - y0) }};
  }}

  function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
  function rgbStringToArray(s) {{
    const m = /rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)/.exec(s || '');
    if (!m) return [0, 0, 0];
    return [Number(m[1]), Number(m[2]), Number(m[3])];
  }}
  const cmap = DATA.colorscale.map(([t, c]) => [Number(t), rgbStringToArray(c)]);
  function scaledValue(v) {{
    const t = clamp(Number(v) || 0, 0, 1);
    if ((DATA.field_scale || 'linear') === 'log') {{
      const floor = 1e-4;
      return clamp((Math.log10(t + floor) - Math.log10(floor)) / (0 - Math.log10(floor)), 0, 1);
    }}
    return t;
  }}
  function inferno(v) {{
    const t = scaledValue(v);
    for (let n = 0; n < cmap.length - 1; n++) {{
      const [t0, c0] = cmap[n];
      const [t1, c1] = cmap[n + 1];
      if (t <= t1) {{
        const a = (t1 === t0) ? 0 : (t - t0) / (t1 - t0);
        return [
          (c0[0] + a * (c1[0] - c0[0])) / 255,
          (c0[1] + a * (c1[1] - c0[1])) / 255,
          (c0[2] + a * (c1[2] - c0[2])) / 255,
        ];
      }}
    }}
    const c = cmap[cmap.length - 1][1];
    return [c[0] / 255, c[1] / 255, c[2] / 255];
  }}
  function infernoCss(v) {{
    const c = inferno(v).map(x => Math.round(x * 255));
    return `rgb(${{c[0]}}, ${{c[1]}}, ${{c[2]}})`;
  }}

  function compileShader(type, source) {{
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
      throw new Error(gl.getShaderInfoLog(shader) || 'shader compile failed');
    }}
    return shader;
  }}
  function initWebGL() {{
    if (!gl || DATA.view !== 'field' || !DATA.field) return;
    const vert = `
      attribute vec2 a_position;
      attribute vec3 a_color;
      uniform vec2 u_center;
      uniform vec2 u_canvas_resolution;
      uniform vec2 u_plot_origin;
      uniform vec2 u_plot_size;
      uniform float u_scale;
      varying vec3 v_color;
      void main() {{
        float screenX = u_plot_origin.x + u_plot_size.x * 0.5 + (a_position.x - u_center.x) * u_scale;
        float screenY = u_plot_origin.y + u_plot_size.y * 0.5 - (a_position.y - u_center.y) * u_scale;
        float clipX = screenX * 2.0 / u_canvas_resolution.x - 1.0;
        float clipY = 1.0 - screenY * 2.0 / u_canvas_resolution.y;
        gl_Position = vec4(clipX, clipY, 0.0, 1.0);
        v_color = a_color;
      }}
    `;
    const frag = `
      precision mediump float;
      varying vec3 v_color;
      void main() {{
        gl_FragColor = vec4(v_color, 1.0);
      }}
    `;
    glProgram = gl.createProgram();
    gl.attachShader(glProgram, compileShader(gl.VERTEX_SHADER, vert));
    gl.attachShader(glProgram, compileShader(gl.FRAGMENT_SHADER, frag));
    gl.linkProgram(glProgram);
    if (!gl.getProgramParameter(glProgram, gl.LINK_STATUS)) {{
      throw new Error(gl.getProgramInfoLog(glProgram) || 'program link failed');
    }}

    const f = DATA.field;
    const xs = f.x || [];
    const ys = f.y || [];
    const vs = f.value || [];
    const ii = f.i || [];
    const jj = f.j || [];
    const kk = f.k || [];
    const positions = [];
    const colors = [];
    function pushVertex(idx) {{
      positions.push(Number(xs[idx]) || 0, Number(ys[idx]) || 0);
      colors.push(...inferno(Number(vs[idx]) || 0));
    }}
    for (let n = 0; n < ii.length; n++) {{
      const a = ii[n], b = jj[n], c = kk[n];
      if (a < xs.length && b < xs.length && c < xs.length && a < vs.length && b < vs.length && c < vs.length) {{
        pushVertex(a); pushVertex(b); pushVertex(c);
      }}
    }}
    vertexCount = positions.length / 2;
    positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
  }}

  function fitView() {{
    const b = DATA.bounds || {{ left: 0, right: 1, bottom: 0, top: 1 }};
    const w = Math.max(1e-9, Number(b.right) - Number(b.left));
    const h = Math.max(1e-9, Number(b.top) - Number(b.bottom));
    const p = plotRect();
    const dpr = window.devicePixelRatio || 1;
    const innerPad = 12 * dpr;
    const sx = Math.max(1, (p.w - innerPad * 2) / w);
    const sy = Math.max(1, (p.h - innerPad * 2) / h);
    state.scale = Math.max(1, Math.min(sx, sy));
    state.minScale = state.scale * 0.35;
    state.centerX = (Number(b.left) + Number(b.right)) / 2;
    state.centerY = (Number(b.bottom) + Number(b.top)) / 2;
  }}
  function screenToWorld(sx, sy) {{
    const p = plotRect();
    return {{
      x: state.centerX + (sx - (p.x0 + p.w / 2)) / state.scale,
      y: state.centerY - (sy - (p.y0 + p.h / 2)) / state.scale,
    }};
  }}
  function worldToScreen(x, y) {{
    const p = plotRect();
    return {{
      x: (x - state.centerX) * state.scale + p.x0 + p.w / 2,
      y: p.y0 + p.h / 2 - (y - state.centerY) * state.scale,
    }};
  }}
  function withPlotClip(fn) {{
    const p = plotRect();
    ctx.save();
    ctx.beginPath();
    ctx.rect(p.x0, p.y0, p.w, p.h);
    ctx.clip();
    fn();
    ctx.restore();
  }}
  function requestDraw() {{
    if (!raf) raf = requestAnimationFrame(draw);
  }}
  function resize() {{
    const dpr = window.devicePixelRatio || 1;
    const rect = root.getBoundingClientRect();
    const cssW = Math.max(320, rect.width);
    const cssH = DATA.height || 650;
    for (const c of [glCanvas, uiCanvas]) {{
      c.style.width = cssW + 'px';
      c.style.height = cssH + 'px';
      c.width = Math.round(cssW * dpr);
      c.height = Math.round(cssH * dpr);
    }}
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    fitView();
    requestDraw();
  }}

  const fallbackPalette = ['#f2f2f2', '#dedede', '#c9c9c9', '#b7b7b7', '#a5a5a5', '#d7d7d7', '#e8e8e8'];
  function pathPart(part) {{
    ctx.beginPath();
    for (const ring of [part.exterior, ...(part.holes || [])]) {{
      if (!ring || !ring.length) continue;
      const p0 = worldToScreen(ring[0][0], ring[0][1]);
      ctx.moveTo(p0.x, p0.y);
      for (let m = 1; m < ring.length; m++) {{
        const p = worldToScreen(ring[m][0], ring[m][1]);
        ctx.lineTo(p.x, p.y);
      }}
      ctx.closePath();
    }}
  }}
  function initLegend() {{
    if (!legend) return;
    if (DATA.view !== 'cross_section' || !DATA.shapes.length) {{
      legend.style.display = 'none';
      return;
    }}
    legend.style.display = 'block';
    legend.innerHTML = DATA.shapes.map(shape => {{
      const label = `${{shape.name}}${{shape.material ? ' · ' + shape.material : ''}}`;
      return `<div class="legend-item"><span style="background:${{shape.color || '#ddd'}}"></span>${{label}}</div>`;
    }}).join('');
  }}
  function drawShapeLabels() {{
    if (DATA.view !== 'cross_section') return;
    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.font = graphFont(12.5, 500);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const shape of DATA.shapes || []) {{
      if (!shape.label) continue;
      const p = worldToScreen(Number(shape.label[0]), Number(shape.label[1]));
      const text = shape.name || '';
      if (!text || p.x < 20 * dpr || p.x > uiCanvas.width - 20 * dpr || p.y < 20 * dpr || p.y > uiCanvas.height - 20 * dpr) continue;
      const metrics = ctx.measureText(text);
      const bw = metrics.width + 12 * dpr;
      const bh = 18 * dpr;
      ctx.fillStyle = 'rgba(255,255,255,0.86)';
      ctx.strokeStyle = 'rgba(0,0,0,0.16)';
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      ctx.roundRect(p.x - bw / 2, p.y - bh / 2, bw, bh, 7 * dpr);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = GRAPH_TEXT;
      ctx.fillText(text, p.x, p.y + 0.5 * dpr);
    }}
    ctx.restore();
  }}
  function saveImage() {{
    const tmp = document.createElement('canvas');
    tmp.width = uiCanvas.width;
    tmp.height = uiCanvas.height;
    const tctx = tmp.getContext('2d');
    tctx.fillStyle = '#ffffff';
    tctx.fillRect(0, 0, tmp.width, tmp.height);
    tctx.drawImage(glCanvas, 0, 0);
    tctx.drawImage(uiCanvas, 0, 0);
    const link = document.createElement('a');
    link.download = DATA.view === 'field' ? 'mode_intensity.png' : 'cross_section_model.png';
    link.href = tmp.toDataURL('image/png');
    link.click();
  }}
  function niceTickStep(rawStep) {{
    const exp = Math.floor(Math.log10(Math.max(rawStep, 1e-12)));
    const base = Math.pow(10, exp);
    for (const m of [1, 2, 5, 10]) {{
      const step = m * base;
      if (rawStep <= step) return step;
    }}
    return 10 * base;
  }}
  function formatTickLabel(value) {{
    const n = Math.abs(value) < 1e-12 ? 0 : value;
    return Number(n.toPrecision(5)).toString().replace(/-/g, '−');
  }}
  function drawGrid() {{
    const dpr = window.devicePixelRatio || 1;
    const p = plotRect();
    const leftWorld = screenToWorld(p.x0, p.y1).x;
    const rightWorld = screenToWorld(p.x1, p.y0).x;
    const bottomWorld = screenToWorld(p.x0, p.y1).y;
    const topWorld = screenToWorld(p.x1, p.y0).y;
    const step = niceTickStep(105 * dpr / state.scale);

    ctx.save();
    ctx.lineWidth = 1 * dpr;
    ctx.strokeStyle = GRAPH_GRID;
    ctx.beginPath();
    ctx.rect(p.x0, p.y0, p.w, p.h);
    ctx.clip();
    const x0 = Math.floor(leftWorld / step) * step;
    for (let x = x0; x <= rightWorld + step * 0.5; x += step) {{
      const sp = worldToScreen(x, 0);
      ctx.beginPath(); ctx.moveTo(sp.x, p.y0); ctx.lineTo(sp.x, p.y1); ctx.stroke();
    }}
    const y0 = Math.floor(bottomWorld / step) * step;
    for (let y = y0; y <= topWorld + step * 0.5; y += step) {{
      const sp = worldToScreen(0, y);
      ctx.beginPath(); ctx.moveTo(p.x0, sp.y); ctx.lineTo(p.x1, sp.y); ctx.stroke();
    }}
    ctx.restore();

    ctx.save();
    ctx.lineWidth = 1 * dpr;
    ctx.strokeStyle = GRAPH_AXIS;
    ctx.strokeRect(p.x0, p.y0, p.w, p.h);
    ctx.fillStyle = GRAPH_TEXT;
    ctx.font = graphFont(14, 500);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xStart = Math.floor(leftWorld / step) * step;
    for (let x = xStart; x <= rightWorld + step * 0.5; x += step) {{
      const sp = worldToScreen(x, 0);
      if (sp.x >= p.x0 + 4 * dpr && sp.x <= p.x1 - 4 * dpr) {{
        ctx.beginPath(); ctx.moveTo(sp.x, p.y1); ctx.lineTo(sp.x, p.y1 - 6 * dpr); ctx.stroke();
        ctx.fillText(formatTickLabel(x), sp.x, p.y1 + 8 * dpr);
      }}
    }}
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yStart = Math.floor(bottomWorld / step) * step;
    for (let y = yStart; y <= topWorld + step * 0.5; y += step) {{
      const sp = worldToScreen(0, y);
      if (sp.y >= p.y0 + 4 * dpr && sp.y <= p.y1 - 4 * dpr) {{
        ctx.beginPath(); ctx.moveTo(p.x0, sp.y); ctx.lineTo(p.x0 + 6 * dpr, sp.y); ctx.stroke();
        ctx.fillText(formatTickLabel(y), p.x0 - 9 * dpr, sp.y);
      }}
    }}
    ctx.restore();
  }}
  function drawDomain() {{
    const d = DATA.domain;
    if (!d) return;
    const p0 = worldToScreen(Number(d.left), Number(d.bottom));
    const p1 = worldToScreen(Number(d.right), Number(d.top));
    ctx.save();
    ctx.setLineDash([7, 5]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(0,0,0,0.56)';
    ctx.strokeRect(p0.x, p1.y, p1.x - p0.x, p0.y - p1.y);
    ctx.setLineDash([7, 5]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#ffffff';
    ctx.strokeRect(p0.x, p1.y, p1.x - p0.x, p0.y - p1.y);
    ctx.restore();
  }}
  function drawShapes(fill) {{
    ctx.save();
    DATA.shapes.forEach((shape, idx) => {{
      for (const part of shape.parts || []) {{
        pathPart(part);
        if (fill) {{
          ctx.fillStyle = shape.color || fallbackPalette[idx % fallbackPalette.length];
          ctx.fill('evenodd');
        }}
        ctx.lineWidth = fill ? 1.7 : 2.0;
        ctx.strokeStyle = fill ? '#0f0f0f' : 'rgba(0,0,0,0.78)';
        ctx.stroke();
        if (!fill && DATA.view === 'field') {{
          ctx.lineWidth = 1.0;
          ctx.strokeStyle = 'rgba(255,255,255,0.96)';
          ctx.stroke();
        }}
      }}
    }});
    ctx.restore();
  }}
  function drawColorbar() {{
    if (DATA.view !== 'field') return;
    const dpr = window.devicePixelRatio || 1;
    const p = plotRect();
    const barW = 16 * dpr;
    const barH = Math.min(340 * dpr, p.h * 0.62);
    const x = p.x1 + 30 * dpr;
    const y = p.y0 + (p.h - barH) / 2;
    const grad = ctx.createLinearGradient(0, y + barH, 0, y);
    for (const [t, c] of cmap) grad.addColorStop(t, `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`);
    ctx.save();
    ctx.fillStyle = grad;
    ctx.fillRect(x, y, barW, barH);
    ctx.strokeStyle = GRAPH_AXIS;
    ctx.lineWidth = 1 * dpr;
    ctx.strokeRect(x, y, barW, barH);
    ctx.fillStyle = GRAPH_TEXT;
    ctx.font = graphFont(13.5, 500);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    const baseLabel = DATA.field?.z_label || 'normalized |E|^2';
    ctx.fillText((DATA.field_scale || 'linear') === 'log' ? `log ${{baseLabel}}` : baseLabel, x - 4 * dpr, y - 9 * dpr);
    ctx.textBaseline = 'middle';
    const ticks = (DATA.field_scale || 'linear') === 'log' ? [1e-4, 1e-3, 1e-2, 1e-1, 1] : [0, 0.2, 0.4, 0.6, 0.8, 1];
    for (const t of ticks) {{
      const yy = y + barH * (1 - scaledValue(t));
      ctx.beginPath(); ctx.moveTo(x + barW, yy); ctx.lineTo(x + barW + 4 * dpr, yy); ctx.stroke();
      const label = (DATA.field_scale || 'linear') === 'log' && t < 1 ? t.toExponential(0) : t.toFixed(t === 0 || t === 1 ? 0 : 1);
      ctx.fillText(label, x + barW + 8 * dpr, yy);
    }}
    ctx.restore();
  }}
  function drawWebGL() {{
    if (!gl) return;
    gl.viewport(0, 0, glCanvas.width, glCanvas.height);
    gl.clearColor(1, 1, 1, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (DATA.view !== 'field' || !glProgram || !vertexCount) return;
    const p = plotRect();
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(Math.round(p.x0), Math.round(glCanvas.height - p.y1), Math.round(p.w), Math.round(p.h));
    gl.useProgram(glProgram);
    const posLoc = gl.getAttribLocation(glProgram, 'a_position');
    const colLoc = gl.getAttribLocation(glProgram, 'a_color');
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.enableVertexAttribArray(colLoc);
    gl.vertexAttribPointer(colLoc, 3, gl.FLOAT, false, 0, 0);
    gl.uniform2f(gl.getUniformLocation(glProgram, 'u_center'), state.centerX, state.centerY);
    gl.uniform2f(gl.getUniformLocation(glProgram, 'u_canvas_resolution'), glCanvas.width, glCanvas.height);
    gl.uniform2f(gl.getUniformLocation(glProgram, 'u_plot_origin'), p.x0, p.y0);
    gl.uniform2f(gl.getUniformLocation(glProgram, 'u_plot_size'), p.w, p.h);
    gl.uniform1f(gl.getUniformLocation(glProgram, 'u_scale'), state.scale);
    gl.drawArrays(gl.TRIANGLES, 0, vertexCount);
    gl.disable(gl.SCISSOR_TEST);
  }}
  function drawAxesLabels() {{
    const dpr = window.devicePixelRatio || 1;
    const p = plotRect();
    ctx.save();
    ctx.fillStyle = GRAPH_TEXT;
    ctx.font = graphFont(15, 500);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('x [um]', p.x0 + p.w / 2, uiCanvas.height - 13 * dpr);
    ctx.translate(18 * dpr, p.y0 + p.h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('y [um]', 0, 0);
    ctx.restore();
  }}
  function draw() {{
    raf = 0;
    drawWebGL();
    ctx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
    drawGrid();
    if (DATA.view !== 'field') {{
      withPlotClip(() => {{
        drawShapes(true);
        drawShapeLabels();
        drawDomain();
      }});
    }} else {{
      withPlotClip(() => {{
        drawShapes(false);
        drawDomain();
      }});
      drawColorbar();
    }}
    drawAxesLabels();
  }}

  uiCanvas.addEventListener('wheel', (ev) => {{
    ev.preventDefault();
    const rect = uiCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const sx = (ev.clientX - rect.left) * dpr;
    const sy = (ev.clientY - rect.top) * dpr;
    const before = screenToWorld(sx, sy);
    const factor = Math.exp(-ev.deltaY * 0.0015);
    state.scale = clamp(state.scale * factor, state.minScale, state.minScale * 220);
    const p = plotRect();
    state.centerX = before.x - (sx - (p.x0 + p.w / 2)) / state.scale;
    state.centerY = before.y + (sy - (p.y0 + p.h / 2)) / state.scale;
    requestDraw();
  }}, {{ passive: false }});
  uiCanvas.addEventListener('pointerdown', (ev) => {{
    state.dragging = true;
    state.lastX = ev.clientX;
    state.lastY = ev.clientY;
    uiCanvas.setPointerCapture(ev.pointerId);
    help.style.opacity = '0';
  }});
  uiCanvas.addEventListener('pointermove', (ev) => {{
    if (!state.dragging) return;
    const dpr = window.devicePixelRatio || 1;
    const dx = (ev.clientX - state.lastX) * dpr;
    const dy = (ev.clientY - state.lastY) * dpr;
    state.lastX = ev.clientX;
    state.lastY = ev.clientY;
    state.centerX -= dx / state.scale;
    state.centerY += dy / state.scale;
    requestDraw();
  }});
  uiCanvas.addEventListener('pointerup', (ev) => {{ state.dragging = false; }});
  uiCanvas.addEventListener('pointercancel', () => {{ state.dragging = false; }});
  uiCanvas.addEventListener('dblclick', () => {{ fitView(); requestDraw(); }});
  if (saveButton) saveButton.addEventListener('click', saveImage);
  window.addEventListener('resize', resize);

  try {{ initWebGL(); }} catch (err) {{ console.error(err); }}
  initLegend();
  resize();
}})();
</script>
<style>
.ms-canvas-viewer {{
  position: relative;
  width: 100%;
  height: {int(height)}px;
  background: #ffffff;
  border: 1px solid rgba(0, 0, 0, 0.10);
  border-radius: 14px;
  box-shadow: none;
  overflow: hidden;
}}
.ms-canvas-viewer canvas {{
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: {int(height)}px;
}}
.ms-canvas-viewer #ui-canvas {{
  cursor: grab;
  touch-action: none;
}}
.ms-canvas-viewer #ui-canvas:active {{ cursor: grabbing; }}
.viewer-save {{
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 4;
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 999px;
  background: rgba(255,255,255,0.88);
  color: #111827;
  font: 750 12px system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
  letter-spacing: -0.01em;
  padding: 6px 11px;
  cursor: pointer;
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
}}
.viewer-save:hover {{ background: #ffffff; border-color: rgba(0,0,0,0.28); }}
.viewer-legend {{
  position: absolute;
  right: 12px;
  bottom: 12px;
  max-width: min(360px, 45%);
  max-height: 150px;
  overflow: auto;
  display: none;
  background: rgba(255,255,255,0.90);
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 14px;
  padding: 8px 10px;
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.07);
  font: 13px Arial, Helvetica, sans-serif;
}}
.legend-item {{ display: flex; align-items: center; gap: 7px; color: rgba(0,0,0,0.78); white-space: nowrap; }}
.legend-item span {{ width: 12px; height: 12px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.18); flex: 0 0 auto; }}
.viewer-help {{
  position: absolute;
  left: 10px;
  top: 8px;
  padding: 4px 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.90);
  border: 1px solid rgba(0,0,0,0.10);
  color: rgba(0,0,0,0.74);
  backdrop-filter: blur(12px);
  font: 13px Arial, Helvetica, sans-serif;
  pointer-events: none;
  transition: opacity 0.2s ease;
}}
</style>
"""

def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode('utf-8')


def json_bytes(data: Any) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def build_result_metadata_payload(
    *,
    csv_file: str,
    results: dict[str, Any] | None,
    section_data: dict[str, Any] | None,
    materials_data: dict[str, Any] | None,
    config: dict[str, Any] | None,
    row_count: int | None = None,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    safe_results = json_safe(results or {})
    analysis_type = 'unknown'
    if isinstance(safe_results, dict):
        analysis_type = str(safe_results.get('analysis_type') or 'single')
    return {
        'format': 'modestudio_result_metadata',
        'schema_version': 1,
        'generated_at': utc_timestamp(),
        'app_version': APP_VERSION,
        'project_name': str(st.session_state.get('project_name') or '').strip() or _fallback_project_name_from_label(st.session_state.get('section_file_label')),
        'csv_file': csv_file,
        'table': {
            'row_count': int(row_count) if row_count is not None else None,
            'columns': list(columns or []),
        },
        'analysis_type': analysis_type,
        'section': json_safe(section_data or {}),
        'materials': json_safe(materials_data or {}),
        'config': json_safe(config or {}),
        'results': safe_results,
    }


def result_metadata_json_bytes(
    *,
    csv_file: str,
    results: dict[str, Any] | None,
    section_data: dict[str, Any] | None,
    materials_data: dict[str, Any] | None,
    config: dict[str, Any] | None,
    dataframe: pd.DataFrame | None = None,
) -> bytes:
    return json_bytes(
        build_result_metadata_payload(
            csv_file=csv_file,
            results=results,
            section_data=section_data,
            materials_data=materials_data,
            config=config,
            row_count=len(dataframe) if dataframe is not None else None,
            columns=[str(column) for column in dataframe.columns] if dataframe is not None else [],
        )
    )


def field_data_json_bytes(
    *,
    field_map: dict[str, Any],
    field_key: str,
    mode_index: int,
    results: dict[str, Any] | None,
    section_data: dict[str, Any] | None,
    materials_data: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> bytes:
    payload = {
        'format': 'modestudio_mode_field_data',
        'schema_version': 1,
        'generated_at': utc_timestamp(),
        'app_version': APP_VERSION,
        'project_name': str(st.session_state.get('project_name') or '').strip() or _fallback_project_name_from_label(st.session_state.get('section_file_label')),
        'mode_index': int(mode_index),
        'field_key': str(field_key),
        'field_label': field_map.get('field_label') or field_map.get('quantity') or str(field_key),
        'wavelength_nm': field_map.get('wavelength_nm') if isinstance(field_map, dict) else None,
        'section': json_safe(section_data or {}),
        'materials': json_safe(materials_data or {}),
        'config': json_safe(config or {}),
        'results': json_safe(results or {}),
        'field_map': json_safe(field_map),
    }
    return json_bytes(payload)


def text_json_bytes(text: str) -> bytes:
    parsed = json.loads(text)
    return json.dumps(parsed, indent=2, ensure_ascii=False).encode('utf-8')


def try_text_json_bytes(text: str) -> bytes | None:
    try:
        return text_json_bytes(text)
    except Exception:
        return None


STATE_GROUPS = {
    'domain': [
        'focus_regions',
        'manual_left',
        'manual_right',
        'manual_bottom',
        'manual_top',
        'empty_area_material_choice',
    ],
    'solver': [
        'wavelength_mode',
        'single_wavelength_nm',
        'sweep_start_nm',
        'sweep_stop_nm',
        'sweep_points',
        'num_modes',
        'order',
    ],
    'mesh': [
        'mesh_preset',
    ],
    'view': [
        'active_view',
        'selected_result_mode',
        'selected_sweep_index',
        'selected_field_quantity',
        'selected_field_label',
        'selected_field_scale',
        'sweep_metric_key',
        'sweep_graph_display_modes',
    ],
}

STATE_LOAD_KEYS = sorted({key for keys in STATE_GROUPS.values() for key in keys} | {'empty_area_n_real', 'empty_area_k'})


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _pretty_json_text(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + '\n'


def _parse_json_text(text: str, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputValidationError(f'{label} JSON could not be parsed: {exc}') from exc
    if not isinstance(payload, dict):
        raise InputValidationError(f'{label} JSON must be an object at the top level.')
    return payload


def _state_value(key: str) -> Any:
    return json_safe(st.session_state.get(key))


def _values_for_keys(keys: list[str]) -> dict[str, Any]:
    return {key: _state_value(key) for key in keys if key in st.session_state}


def _domain_state_payload() -> dict[str, Any]:
    payload = _values_for_keys([
        'focus_regions',
        'manual_left',
        'manual_right',
        'manual_bottom',
        'manual_top',
        'empty_area_material_choice',
    ])
    if st.session_state.get('empty_area_material_choice') == USER_DEFINED_EMPTY_AREA_LABEL:
        payload.update(_values_for_keys(['empty_area_n_real', 'empty_area_k']))
    return payload


def _solver_state_payload() -> dict[str, Any]:
    payload = _values_for_keys(['wavelength_mode', 'num_modes', 'order'])
    mode = str(st.session_state.get('wavelength_mode', 'Single'))
    if mode == 'Sweep':
        payload.update(_values_for_keys(['sweep_start_nm', 'sweep_stop_nm', 'sweep_points']))
    else:
        payload.update(_values_for_keys(['single_wavelength_nm']))
    return payload


def build_ui_state_payload(config: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        'domain': _domain_state_payload(),
        'solver': _solver_state_payload(),
        'mesh': _values_for_keys(['mesh_preset']),
        'view': _values_for_keys([
            'active_view',
            'selected_result_mode',
            'selected_sweep_index',
            'selected_field_quantity',
            'selected_field_label',
            'selected_field_scale',
            'sweep_metric_key',
            'sweep_graph_display_modes',
        ]),
    }
    return {group: values for group, values in payload.items() if values}


def _coerce_wavelength_mode(value: Any) -> str | None:
    normalized = str(value or '').strip().lower()
    if normalized == 'sweep':
        return 'Sweep'
    if normalized == 'single':
        return 'Single'
    return None


def _apply_state_values(values: dict[str, Any]) -> None:
    for key in STATE_LOAD_KEYS:
        if key not in values:
            continue
        if key == 'wavelength_mode':
            mode = _coerce_wavelength_mode(values.get(key))
            if mode is not None:
                st.session_state[key] = mode
            continue
        st.session_state[key] = values[key]


def _flatten_current_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for group in ('domain', 'solver', 'mesh', 'view'):
        group_payload = payload.get(group, {})
        if isinstance(group_payload, dict):
            values.update(group_payload)
    return values


def _flatten_legacy_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    ui_state = payload.get('ui_state', payload.get('session_state', {}))
    if isinstance(ui_state, dict):
        values.update({key: ui_state[key] for key in STATE_LOAD_KEYS if key in ui_state})

    analysis = payload.get('analysis', payload.get('analysis_config', {}))
    if not isinstance(analysis, dict):
        analysis = {}
    if not analysis and any(key in payload for key in ('manual_left', 'wavelength_um', 'num_modes', 'order')):
        analysis = payload

    direct_keys = {
        'manual_left',
        'manual_right',
        'manual_bottom',
        'manual_top',
        'focus_regions',
        'empty_area_n_real',
        'empty_area_k',
        'wavelength_mode',
        'sweep_start_nm',
        'sweep_stop_nm',
        'sweep_points',
        'num_modes',
        'order',
        'mesh_preset',
    }
    for key in direct_keys:
        if key in analysis and key not in values:
            values[key] = analysis[key]

    if 'wavelength_um' in analysis and 'single_wavelength_nm' not in values:
        try:
            values['single_wavelength_nm'] = float(analysis['wavelength_um']) * 1000.0
        except Exception:
            pass

    empty_mode = str(analysis.get('empty_area_material_mode', '')).lower()
    background_material = str(analysis.get('background_material', '')).strip()
    if 'empty_area_material_choice' not in values:
        if empty_mode == 'user_defined':
            values['empty_area_material_choice'] = USER_DEFINED_EMPTY_AREA_LABEL
        elif background_material:
            values['empty_area_material_choice'] = background_material

    return values


def apply_ui_state_payload(payload: dict[str, Any]) -> None:
    values = _flatten_current_state_payload(payload)
    if not values:
        values = _flatten_legacy_state_payload(payload)
    _apply_state_values(values)


def _fallback_project_name_from_label(label: str | None) -> str:
    stem = Path(str(label or '').strip()).stem
    if not stem or stem == 'section_sample':
        return 'sample_project'
    return stem


def _safe_project_filename_part(name: str | None) -> str:
    cleaned = str(name or '').strip()
    cleaned = re.sub(r'(?i)\.modestudio\.zip$', '', cleaned)
    cleaned = re.sub(r'(?i)\.zip$', '', cleaned)
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', '_', cleaned)
    cleaned = re.sub(r'\s+', '_', cleaned).strip('._ ')
    if not cleaned:
        cleaned = _fallback_project_name_from_label(st.session_state.get('section_file_label'))
    return cleaned[:96] or 'sample_project'


def project_download_filename() -> str:
    return f'{_safe_project_filename_part(st.session_state.get("project_name"))}.modestudio.zip'




def build_python_script_export(
    section_data: dict[str, Any],
    materials_data: dict[str, Any],
    config: dict[str, Any],
    *,
    project_name: str | None = None,
) -> bytes:
    project_label = _safe_project_filename_part(project_name or st.session_state.get('project_name'))
    section_literal = json.dumps(json_safe(section_data), indent=2, ensure_ascii=False)
    materials_literal = json.dumps(json_safe(materials_data), indent=2, ensure_ascii=False)
    config_literal = json.dumps(json_safe(config), indent=2, ensure_ascii=False)
    script = f'''# Generated by ModeStudio.
# Default behavior: reproduce the current GUI analysis once.
# Edit SECTION_DATA, MATERIALS_DATA, or BASE_CONFIG to reuse this script in a Python/Femwell workflow.

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine import run_mode_analysis

PROJECT_NAME = {project_label!r}
OUTPUT_DIR = Path(f"{{PROJECT_NAME}}_script_results")

SECTION_DATA = {section_literal}

MATERIALS_DATA = {materials_literal}

BASE_CONFIG = {config_literal}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")


def complex_real(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("real")
    return None


def complex_imag(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("imag")
    return None


def rows_from_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in run.get("modes", []):
        row = {{
            "mode": mode.get("mode_index"),
            "wavelength_nm": run.get("wavelength_nm", mode.get("wavelength_nm")),
            "n_eff": complex_real(mode.get("n_eff")),
            "k_eff": complex_imag(mode.get("n_eff")),
            "loss_dB_per_cm": mode.get("propagation_loss_dB_per_cm"),
            "TE_fraction": mode.get("te_fraction"),
            "TM_fraction": mode.get("tm_fraction"),
            "reference_power_fraction": complex_real(mode.get("power_reference_fraction")),
        }}
        if "group_index" in mode:
            row["group_index"] = mode.get("group_index")
        rows.append(row)
    return rows


def rows_from_sweep(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs = results.get("wavelength_sweep", {{}}).get("runs", [])
    for sweep_index, run in enumerate(runs):
        for row in rows_from_run(run):
            row = {{"sweep_index": sweep_index, **row}}
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metadata(csv_file: str, rows: list[dict[str, Any]], results: dict[str, Any]) -> dict[str, Any]:
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    return {{
        "format": "modestudio_result_metadata",
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "project_name": PROJECT_NAME,
        "csv_file": csv_file,
        "table": {{
            "row_count": len(rows),
            "columns": columns,
        }},
        "analysis_type": results.get("analysis_type", "single"),
        "section": SECTION_DATA,
        "materials": MATERIALS_DATA,
        "config": BASE_CONFIG,
        "results": results,
    }}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = run_mode_analysis(SECTION_DATA, MATERIALS_DATA, BASE_CONFIG)
    results = artifacts.results

    if results.get("analysis_type") == "sweep":
        csv_name = "wavelength_sweep.csv"
        json_name = "wavelength_sweep_metadata.json"
        rows = rows_from_sweep(results)
    else:
        csv_name = "modes.csv"
        json_name = "modes_metadata.json"
        rows = rows_from_run(results)

    write_csv(OUTPUT_DIR / csv_name, rows)
    write_json(OUTPUT_DIR / json_name, build_metadata(csv_name, rows, results))
    print(f"wrote {{OUTPUT_DIR / csv_name}}")
    print(f"wrote {{OUTPUT_DIR / json_name}}")


if __name__ == "__main__":
    main()
'''
    return script.encode('utf-8')


def build_python_script_bundle_export(
    section_data: dict[str, Any],
    materials_data: dict[str, Any],
    config: dict[str, Any],
    *,
    project_name: str | None = None,
) -> bytes:
    project_label = _safe_project_filename_part(project_name or st.session_state.get('project_name'))
    script_bytes = build_python_script_export(section_data, materials_data, config, project_name=project_label)
    buffer = BytesIO()
    with ZipFile(buffer, mode='w', compression=ZIP_DEFLATED) as archive:
        archive.writestr('run_analysis.py', script_bytes)
        for source_name in ('engine.py', 'requirements.txt'):
            path = APP_DIR / source_name
            if path.exists():
                archive.writestr(source_name, path.read_bytes())
    return buffer.getvalue()


def clear_analysis_output_cache() -> None:
    for key in (
        '_analysis_output_cache_object_id',
        '_analysis_output_cache_bytes',
        '_analysis_output_cache_summary',
    ):
        st.session_state.pop(key, None)


def _analysis_output_payload() -> dict[str, Any] | None:
    analysis_output = st.session_state.get('analysis_output')
    if not isinstance(analysis_output, dict) or not isinstance(analysis_output.get('results'), dict):
        clear_analysis_output_cache()
        return None
    payload = json_safe(analysis_output)
    return payload if isinstance(payload, dict) else None


def _cache_analysis_output_payload(
    analysis_output: dict[str, Any] | None,
    *,
    encoded_payload: bytes | None = None,
    summary: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]] | None:
    if not isinstance(analysis_output, dict) or not isinstance(analysis_output.get('results'), dict):
        clear_analysis_output_cache()
        return None

    object_id = id(analysis_output)
    cached_bytes = st.session_state.get('_analysis_output_cache_bytes')
    cached_summary = st.session_state.get('_analysis_output_cache_summary')
    if (
        st.session_state.get('_analysis_output_cache_object_id') == object_id
        and isinstance(cached_bytes, bytes)
        and isinstance(cached_summary, dict)
    ):
        return cached_bytes, cached_summary

    if encoded_payload is None:
        payload = json_safe(analysis_output)
        if not isinstance(payload, dict):
            clear_analysis_output_cache()
            return None
        encoded_payload = _pretty_json_text(payload).encode('utf-8')
        summary = _analysis_results_summary(payload)
    elif summary is None:
        summary = _analysis_results_summary(analysis_output)

    st.session_state['_analysis_output_cache_object_id'] = object_id
    st.session_state['_analysis_output_cache_bytes'] = encoded_payload
    st.session_state['_analysis_output_cache_summary'] = summary
    return encoded_payload, summary


def _analysis_output_archive_entry() -> tuple[bytes, dict[str, Any]] | None:
    analysis_output = st.session_state.get('analysis_output')
    if not isinstance(analysis_output, dict) or not isinstance(analysis_output.get('results'), dict):
        clear_analysis_output_cache()
        return None
    return _cache_analysis_output_payload(analysis_output)


def _analysis_results_summary(analysis_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(analysis_payload, dict):
        return {'included': False}

    results = analysis_payload.get('results', {})
    if not isinstance(results, dict):
        return {'included': False}

    analysis_type = str(results.get('analysis_type') or 'single')
    if analysis_type == 'sweep':
        wavelengths = results.get('wavelength_sweep', {}).get('wavelengths_nm', []) if isinstance(results.get('wavelength_sweep'), dict) else []
        wavelength_count = len(wavelengths) if isinstance(wavelengths, list) else 0
    else:
        wavelength_count = 1 if results.get('wavelength_nm') is not None or results.get('wavelength_um') is not None else 0

    mode_field_maps = analysis_payload.get('mode_field_maps', [])
    sweep_field_maps = analysis_payload.get('sweep_field_maps', [])
    return {
        'included': True,
        'member': 'analysis_output.json',
        'analysis_type': analysis_type,
        'wavelength_count': int(wavelength_count),
        'mode_field_map_count': len(mode_field_maps) if isinstance(mode_field_maps, list) else 0,
        'sweep_field_map_count': len(sweep_field_maps) if isinstance(sweep_field_maps, list) else 0,
    }


def _validate_analysis_output_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload.get('results'), dict):
        raise InputValidationError('analysis_output.json must contain a results object.')
    for key in ('mode_field_maps', 'sweep_field_maps'):
        if key in payload and not isinstance(payload[key], list):
            raise InputValidationError(f'analysis_output.json field {key} must be a list.')
    payload.setdefault('mode_field_maps', [])
    payload.setdefault('sweep_field_maps', [])
    return payload


def build_project_metadata_payload(
    config: dict[str, Any] | None = None,
    *,
    analysis_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        'format': 'modestudio_project_zip',
        'schema_version': PROJECT_SCHEMA_VERSION,
        'app_version': APP_VERSION,
        'project_name': str(st.session_state.get('project_name') or '').strip() or _fallback_project_name_from_label(st.session_state.get('section_file_label')),
        'model_files': {
            'section_file_label': st.session_state.get('section_file_label', 'section.json'),
            'materials_file_label': st.session_state.get('materials_file_label', 'materials.json'),
        },
        'ui_state': build_ui_state_payload(config),
        'analysis_results': analysis_summary or {'included': False},
    }


def build_project_zip_bytes(config: dict[str, Any] | None = None) -> bytes:
    section_text = str(st.session_state.get('section_text', ''))
    materials_text = str(st.session_state.get('materials_text', ''))
    analysis_entry = _analysis_output_archive_entry()
    analysis_bytes = analysis_entry[0] if analysis_entry is not None else None
    analysis_summary = analysis_entry[1] if analysis_entry is not None else {'included': False}
    buffer = BytesIO()
    with ZipFile(buffer, mode='w', compression=ZIP_DEFLATED) as archive:
        archive.writestr('project.json', _pretty_json_text(build_project_metadata_payload(config, analysis_summary=analysis_summary)))
        archive.writestr('section.json', section_text)
        archive.writestr('materials.json', materials_text)
        if analysis_bytes is not None:
            # Field-map payloads can be very large.  They are serialized once and
            # stored without per-rerun compression so ordinary UI changes do not
            # make the app feel sluggish.
            archive.writestr('analysis_output.json', analysis_bytes, compress_type=ZIP_STORED)
    return buffer.getvalue()


def _read_required_zip_text(archive: ZipFile, name: str) -> str:
    if name not in archive.namelist():
        raise InputValidationError(f'Project ZIP must contain {name}.')
    try:
        return archive.read(name).decode('utf-8')
    except UnicodeDecodeError as exc:
        raise InputValidationError(f'{name} must be UTF-8 encoded.') from exc


def _read_optional_zip_text(archive: ZipFile, name: str) -> str | None:
    if name not in archive.namelist():
        return None
    try:
        return archive.read(name).decode('utf-8')
    except UnicodeDecodeError as exc:
        raise InputValidationError(f'{name} must be UTF-8 encoded.') from exc


def load_project_zip_bytes(raw: bytes, *, filename: str = 'project.zip') -> None:
    try:
        with ZipFile(BytesIO(raw), mode='r') as archive:
            project_member = 'project.json' if 'project.json' in archive.namelist() else 'state.json'
            project_text = _read_required_zip_text(archive, project_member)
            section_text = _read_required_zip_text(archive, 'section.json')
            materials_text = _read_required_zip_text(archive, 'materials.json')
            analysis_text = _read_optional_zip_text(archive, 'analysis_output.json')
    except BadZipFile as exc:
        raise InputValidationError('Project file must be a ZIP archive created by ModeStudio.') from exc

    payload = _parse_json_text(project_text, label='Project')
    if payload.get('format') not in (None, 'modestudio_project_zip', 'modestudio_state_zip'):
        raise InputValidationError('This ZIP does not look like a ModeStudio project file.')

    model_files = payload.get('model_files', {}) if isinstance(payload.get('model_files'), dict) else {}
    st.session_state['section_text'] = section_text
    st.session_state['materials_text'] = materials_text
    st.session_state['section_file_label'] = str(model_files.get('section_file_label') or 'section.json')
    st.session_state['materials_file_label'] = str(model_files.get('materials_file_label') or 'materials.json')
    st.session_state['section_upload_token'] = hashlib.sha256(section_text.encode('utf-8')).hexdigest()
    st.session_state['materials_upload_token'] = hashlib.sha256(materials_text.encode('utf-8')).hexdigest()

    project_name = str(payload.get('project_name') or payload.get('state_name') or '').strip()
    if not project_name:
        project_name = _safe_project_filename_part(filename)
    st.session_state['project_name'] = project_name

    ui_state = payload.get('ui_state')
    if isinstance(ui_state, dict):
        apply_ui_state_payload(ui_state)
    else:
        apply_ui_state_payload(payload)

    ignored_section_token = upload_token_from_state('section_upload')
    ignored_materials_token = upload_token_from_state('materials_upload')
    if ignored_section_token:
        st.session_state['_ignored_section_upload_token'] = ignored_section_token
    if ignored_materials_token:
        st.session_state['_ignored_materials_upload_token'] = ignored_materials_token

    restored_analysis_output: dict[str, Any] | None = None
    if analysis_text:
        restored_analysis_output = _validate_analysis_output_payload(_parse_json_text(analysis_text, label='Analysis results'))
    elif isinstance(payload.get('analysis_output'), dict):
        restored_analysis_output = _validate_analysis_output_payload(payload['analysis_output'])

    st.session_state['project_upload_token'] = hashlib.sha256(raw).hexdigest()
    st.session_state['analysis_output'] = restored_analysis_output
    if restored_analysis_output is not None:
        raw_analysis_bytes = analysis_text.encode('utf-8') if analysis_text else None
        if raw_analysis_bytes is not None and not raw_analysis_bytes.endswith(b'\n'):
            raw_analysis_bytes += b'\n'
        _cache_analysis_output_payload(restored_analysis_output, encoded_payload=raw_analysis_bytes)
    else:
        clear_analysis_output_cache()
    st.session_state['analysis_error'] = None
    if restored_analysis_output is not None:
        st.session_state['_project_status'] = ('success', f'Loaded project and analysis results: {filename}')
    else:
        st.session_state['_project_status'] = ('success', f'Loaded project: {filename}')
    st.session_state['_skip_model_upload_processing_once'] = True

def process_project_upload_if_needed() -> None:
    project_upload = st.session_state.get('project_upload')
    if project_upload is None:
        return
    raw = project_upload.getvalue()
    token = hashlib.sha256(raw).hexdigest()
    if token == st.session_state.get('project_upload_token'):
        return
    try:
        load_project_zip_bytes(raw, filename=project_upload.name)
    except Exception as exc:
        st.session_state['project_upload_token'] = token
        st.session_state['_project_status'] = ('error', f'Could not load project: {exc}')
    else:
        st.rerun()


def render_archive_dock(config: dict[str, Any] | None = None) -> None:
    fallback_name = _fallback_project_name_from_label(st.session_state.get('section_file_label'))
    if not str(st.session_state.get('project_name') or '').strip():
        st.session_state['project_name'] = fallback_name

    st.markdown('<div class="studio-archive-inline-marker"></div>', unsafe_allow_html=True)
    with st.popover('Archive', use_container_width=True):
        st.text_input(
            'Archive name',
            key='project_name',
            placeholder='sample_project',
        )
        st.download_button(
            'Save archive',
            data=build_project_zip_bytes(config),
            file_name=project_download_filename(),
            mime='application/zip',
            use_container_width=True,
        )
        st.file_uploader(
            'Open archive (.modestudio.zip)',
            type=['zip'],
            key='project_upload',
        )

    status = st.session_state.pop('_project_status', None)
    if isinstance(status, tuple) and len(status) == 2:
        level, message = status
        try:
            st.toast(str(message), icon='✅' if level == 'success' else '⚠️')
        except Exception:
            if level == 'success':
                st.success(str(message))
            else:
                st.error(str(message))


def render_raw_json_editor() -> None:
    st.caption(f"Section file: {st.session_state.get('section_file_label', 'section_sample.json')}")
    section_text = st.text_area('Section JSON', key='section_text', height=300, label_visibility='collapsed')
    section_download = try_text_json_bytes(section_text)
    if section_download is None:
        st.caption('Section JSON is not valid yet.')
    else:
        st.download_button('Download section.json', data=section_download, file_name='section.json', mime='application/json', use_container_width=True)

    st.caption(f"Materials file: {st.session_state.get('materials_file_label', 'materials_sample.json')}")
    materials_text = st.text_area('Materials JSON', key='materials_text', height=260, label_visibility='collapsed')
    materials_download = try_text_json_bytes(materials_text)
    if materials_download is None:
        st.caption('Materials JSON is not valid yet.')
    else:
        st.download_button('Download materials.json', data=materials_download, file_name='materials.json', mime='application/json', use_container_width=True)


def set_manual_domain_from_bounds(bounds: dict[str, float]) -> None:
    st.session_state['manual_left'] = float(bounds['left'])
    st.session_state['manual_right'] = float(bounds['right'])
    st.session_state['manual_bottom'] = float(bounds['bottom'])
    st.session_state['manual_top'] = float(bounds['top'])


def manual_domain_is_uninitialized() -> bool:
    keys = ('manual_left', 'manual_right', 'manual_bottom', 'manual_top')
    if any(key not in st.session_state for key in keys):
        return True
    values = [float(st.session_state.get(key, 0.0)) for key in keys]
    return all(abs(value) < 1e-12 for value in values)


def auto_fit_wavelength_um_from_state() -> float:
    mode = st.session_state.get('wavelength_mode', 'Single')
    try:
        if mode == 'Sweep':
            start_nm = float(st.session_state.get('sweep_start_nm', 1500.0))
            stop_nm = float(st.session_state.get('sweep_stop_nm', 1600.0))
            wavelength_nm = max(start_nm, stop_nm)
        else:
            wavelength_nm = float(st.session_state.get('single_wavelength_nm', 1550.0))
    except (TypeError, ValueError):
        wavelength_nm = 1550.0
    if not math.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        wavelength_nm = 1550.0
    return wavelength_nm / 1000.0


def _float_state(key: str, default: float) -> float:
    try:
        value = float(st.session_state.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    return value if math.isfinite(value) else float(default)


def _int_state(key: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        value = int(st.session_state.get(key, default))
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def configured_wavelengths_from_state() -> tuple[str, list[float]]:
    mode = str(st.session_state.get('wavelength_mode', 'Single') or 'Single')
    if mode == 'Sweep':
        start_nm = _float_state('sweep_start_nm', 1500.0)
        stop_nm = _float_state('sweep_stop_nm', 1600.0)
        points = _int_state('sweep_points', 5, minimum=2, maximum=31)
        wavelengths = [start_nm + (stop_nm - start_nm) * idx / max(points - 1, 1) for idx in range(points)]
        return 'Sweep', [float(value) for value in wavelengths]

    wavelength_nm = _float_state('single_wavelength_nm', 1550.0)
    return 'Single', [float(wavelength_nm)]


def build_analysis_config_from_state(
    *,
    focus_regions: list[str] | tuple[str, ...],
    mesh_config: dict[str, float],
    empty_choice: str,
) -> dict[str, Any]:
    wavelength_mode_value, configured_wavelengths_nm = configured_wavelengths_from_state()
    empty_area_material_mode = 'user_defined' if empty_choice == USER_DEFINED_EMPTY_AREA_LABEL else 'material'
    background_material = DEFAULT_EMPTY_AREA_MATERIAL if empty_area_material_mode == 'user_defined' else str(empty_choice).strip().lower()
    return {
        'wavelength_mode': 'sweep' if wavelength_mode_value == 'Sweep' else 'single',
        'wavelength_um': float(configured_wavelengths_nm[0]) / 1000.0,
        'sweep_start_nm': float(configured_wavelengths_nm[0]),
        'sweep_stop_nm': float(configured_wavelengths_nm[-1]),
        'sweep_points': int(len(configured_wavelengths_nm)),
        'sweep_wavelengths_um': [float(value) / 1000.0 for value in configured_wavelengths_nm],
        'num_modes': _int_state('num_modes', 2, minimum=1, maximum=20),
        'order': _int_state('order', 1, minimum=1, maximum=2),
        'window_mode': 'manual',
        'manual_left': _float_state('manual_left', 0.0),
        'manual_right': _float_state('manual_right', 1.0),
        'manual_bottom': _float_state('manual_bottom', 0.0),
        'manual_top': _float_state('manual_top', 1.0),
        'focus_regions': list(focus_regions),
        'refined_regions': list(focus_regions),
        'background_name': DEFAULT_EMPTY_AREA_MATERIAL,
        'background_material': background_material,
        'empty_area_material_mode': empty_area_material_mode,
        'empty_area_n_real': _float_state('empty_area_n_real', 1.444),
        'empty_area_k': _float_state('empty_area_k', 0.0),
        'empty_area_n_imag': _float_state('empty_area_k', 0.0),
        **{key: float(value) for key, value in mesh_config.items()},
    }


def ensure_background_material_choice(material_names: list[str]) -> str:
    material_select_options = material_names + [USER_DEFINED_EMPTY_AREA_LABEL]
    if not material_select_options:
        st.session_state['empty_area_material_choice'] = USER_DEFINED_EMPTY_AREA_LABEL
        return USER_DEFINED_EMPTY_AREA_LABEL
    if 'empty_area_material_choice' not in st.session_state or st.session_state['empty_area_material_choice'] not in material_select_options:
        preferred = 'sio2' if 'sio2' in material_names else material_names[0] if material_names else USER_DEFINED_EMPTY_AREA_LABEL
        st.session_state['empty_area_material_choice'] = preferred
    return str(st.session_state['empty_area_material_choice'])


def current_focus_regions_for_config(region_names: list[str], suggested_focus: list[str]) -> list[str]:
    if not region_names:
        return []
    current = [name for name in st.session_state.get('focus_regions', suggested_focus) if name in region_names]
    return current or suggested_focus


def mesh_preset_from_state() -> str:
    preset = str(st.session_state.get('mesh_preset', 'Normal') or 'Normal')
    if preset not in MESH_PRESETS:
        preset = 'Normal'
        st.session_state['mesh_preset'] = preset
    return preset


def process_model_uploads_from_state(*, skip_processing: bool = False) -> None:
    if skip_processing:
        return

    section_upload = st.session_state.get('section_upload')
    if section_upload is not None:
        uploaded_text, uploaded_token = decode_upload(section_upload)
        if uploaded_token == st.session_state.get('_ignored_section_upload_token'):
            pass
        elif uploaded_token != st.session_state['section_upload_token']:
            st.session_state.pop('_ignored_section_upload_token', None)
            st.session_state['section_text'] = uploaded_text
            st.session_state['section_upload_token'] = uploaded_token
            st.session_state['section_file_label'] = section_upload.name
            if str(st.session_state.get('project_name') or '').strip() in {'', 'section_sample', 'modestudio_project', 'sample_project'}:
                st.session_state['project_name'] = _fallback_project_name_from_label(section_upload.name)
            st.session_state['analysis_output'] = None
            clear_analysis_output_cache()
            reset_domain_state()

    materials_upload = st.session_state.get('materials_upload')
    if materials_upload is not None:
        uploaded_text, uploaded_token = decode_upload(materials_upload)
        if uploaded_token == st.session_state.get('_ignored_materials_upload_token'):
            pass
        elif uploaded_token != st.session_state['materials_upload_token']:
            st.session_state.pop('_ignored_materials_upload_token', None)
            st.session_state['materials_text'] = uploaded_text
            st.session_state['materials_upload_token'] = uploaded_token
            st.session_state['materials_file_label'] = materials_upload.name
            st.session_state['analysis_output'] = None
            clear_analysis_output_cache()
            reset_domain_state()


st.set_page_config(page_title='ModeStudio', layout='wide')
init_state()
process_project_upload_if_needed()

st.markdown(
    """
    <style>
    :root {
        --ms-page: #eeeeeb;
        --ms-panel: #ffffff;
        --ms-text: #0f0f0f;
        --ms-muted: #666666;
        --ms-border: #d8d8d5;
        --ms-control: #ffffff;
        --ms-control-soft: #fbfbfa;
        --ms-control-hover: #f5f5f3;
        --ms-control-border: #cfcfca;
        --ms-accent: #5f5f58;
        --ms-accent-hover: #4d4d47;
        --ms-accent-soft: #e5e5e0;
        --ms-accent-ring: rgba(95, 95, 88, 0.22);
        --ms-red: #0f0f0f;
        --ms-red-dark: #272727;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent !important;}
    .stDeployButton,
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }

    html, body,
    .stApp,
    div[data-testid="stAppViewContainer"],
    section.main {
        background: var(--ms-page) !important;
        color: var(--ms-text);
        accent-color: var(--ms-accent) !important;
        --primary-color: var(--ms-accent) !important;
    }

    .block-container {
        padding-top: 0.62rem;
        padding-bottom: 1.05rem;
        padding-left: 1.4rem;
        padding-right: 1.4rem;
        max-width: 1760px;
        background: transparent !important;
    }
    div[data-testid="stVerticalBlock"] {gap: 0.54rem;}
    div[data-testid="stHorizontalBlock"] {align-items: stretch;}

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 0 !important;
        border-radius: 0.95rem !important;
        background: var(--ms-panel) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.055) !important;
        overflow: hidden !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div,
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div {
        border: 0 !important;
        border-radius: inherit !important;
        background: var(--ms-panel) !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--ms-control-border) !important;
        border-radius: 0.72rem !important;
        background: var(--ms-control) !important;
        box-shadow: none !important;
        overflow: hidden !important;
    }
    div[data-testid="stExpander"] > details {
        border: 0 !important;
        background: var(--ms-control) !important;
        border-radius: 0.72rem !important;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 650 !important;
        color: var(--ms-text) !important;
    }

    .app-title {
        font-size: 1.04rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: var(--ms-text);
        margin: 0 0 0.20rem 0.08rem;
        font-weight: 760;
    }
    .panel-label {
        font-size: 0.78rem;
        font-weight: 760;
        color: var(--ms-muted);
        letter-spacing: 0.045em;
        text-transform: uppercase;
        margin: 0.06rem 0 0.22rem 0;
    }
    .table-label {
        font-size: 0.78rem;
        font-weight: 760;
        color: var(--ms-muted);
        letter-spacing: 0.045em;
        text-transform: uppercase;
        margin: 0.16rem 0 0.42rem 0;
    }
    .viewer-title {
        font-size: 1.02rem;
        font-weight: 760;
        color: var(--ms-text);
        margin-top: 0.15rem;
        margin-bottom: 0.05rem;
    }
    .subtle-note, .stCaptionContainer, small {
        color: var(--ms-muted) !important;
    }

    label, div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label,
    div[data-testid="stMultiSelect"] label, div[data-testid="stFileUploader"] label {
        color: var(--ms-text) !important;
        font-weight: 650 !important;
    }

    div[data-baseweb="input"],
    div[data-baseweb="select"] > div {
        border-radius: 0.62rem !important;
        background: #ffffff !important;
        border: 1px solid var(--ms-control-border) !important;
        color: var(--ms-text) !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.028) !important;
    }
    div[data-baseweb="input"]:hover,
    div[data-baseweb="select"] > div:hover {
        background: #ffffff !important;
        border-color: #b8b8b2 !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        background: #ffffff !important;
        border-color: rgba(15,15,15,0.58) !important;
        box-shadow: 0 0 0 2px rgba(15,15,15,0.08) !important;
    }
    div[data-baseweb="input"] input,
    div[data-testid="stNumberInput"] input,
    textarea,
    textarea:focus {
        color: var(--ms-text) !important;
        background: #ffffff !important;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background: #ffffff !important;
        border-color: #c8c8c2 !important;
    }
    div[data-testid="stNumberInput"] button {
        border: 0 !important;
        border-left: 1px solid #d9d9d3 !important;
        background: #ffffff !important;
        color: var(--ms-text) !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background: var(--ms-control-hover) !important;
    }

    div[data-testid="stFileUploader"] {margin: 0.00rem 0 0.25rem 0;}
    div[data-testid="stFileUploader"] section,
    div[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploaderDropzone"] > div {
        background: var(--ms-control-soft) !important;
    }
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        border: 1px solid var(--ms-control-border) !important;
        border-radius: 0.72rem !important;
        background: var(--ms-control-soft) !important;
        min-height: 3.35rem !important;
        padding: 0.38rem 0.58rem !important;
    }
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] > div {
        gap: 0.32rem !important;
        padding: 0 !important;
    }
    div[data-testid="stFileUploader"] button {
        border-radius: 999px !important;
        font-weight: 720 !important;
        background: #ffffff !important;
        color: var(--ms-text) !important;
        border: 1px solid var(--ms-control-border) !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.035) !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background: var(--ms-control-hover) !important;
        border-color: #bfbfba !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 999px !important;
        height: 2.48rem !important;
        font-weight: 720 !important;
        border: 1px solid var(--ms-control-border) !important;
        background: #ffffff !important;
        color: var(--ms-text) !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.035) !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: var(--ms-control-hover) !important;
        border-color: #bfbfba !important;
        color: var(--ms-text) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--ms-red) !important;
        color: #ffffff !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--ms-red-dark) !important;
        color: #ffffff !important;
    }

    div[role="radiogroup"] label {
        padding-top: 0.10rem;
        padding-bottom: 0.10rem;
    }
    div[role="radiogroup"] label,
    div[role="radiogroup"] span {
        color: var(--ms-text) !important;
    }
    input[type="radio"],
    input[type="checkbox"],
    div[role="radiogroup"] input[type="radio"],
    div[data-testid="stRadio"] input:checked,
    div[data-testid="stCheckbox"] input:checked {
        accent-color: var(--ms-accent) !important;
    }
    div[data-testid="stRadio"] svg,
    div[data-testid="stCheckbox"] svg {
        color: var(--ms-accent) !important;
        fill: var(--ms-accent) !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child,
    div[data-testid="stRadio"] label[data-baseweb="radio"] > span:first-child,
    div[data-testid="stCheckbox"] label > div:first-child,
    div[data-testid="stCheckbox"] label > span:first-child {
        background: #ffffff !important;
        border-color: #b9b9b2 !important;
        box-shadow: none !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > span:first-child {
        background: #ffffff !important;
        border-color: var(--ms-accent) !important;
        box-shadow: 0 0 0 2px var(--ms-accent-ring) !important;
    }
    div[data-testid="stCheckbox"] label:has(input:checked) > div:first-child,
    div[data-testid="stCheckbox"] label:has(input:checked) > span:first-child {
        background: var(--ms-accent) !important;
        border-color: var(--ms-accent) !important;
        box-shadow: 0 0 0 2px var(--ms-accent-ring) !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) svg,
    div[data-testid="stRadio"] label:has(input:checked) svg {
        color: var(--ms-accent) !important;
        fill: var(--ms-accent) !important;
    }
    div[data-testid="stCheckbox"] label:has(input:checked) svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    div[data-testid="stDataFrame"] input[type="checkbox"],
    div[data-testid="stDataEditor"] input[type="checkbox"] {
        accent-color: var(--ms-accent) !important;
        width: 16px !important;
        height: 16px !important;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background-color: var(--ms-text) !important;
        color: #ffffff !important;
        border-radius: 0.55rem !important;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] span,
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] button {
        color: #ffffff !important;
    }


    div[data-testid="stSegmentedControl"] > div {
        display: flex !important;
        flex-wrap: nowrap !important;
        width: fit-content !important;
        max-width: 100% !important;
    }
    div[data-testid="stSegmentedControl"] label,
    div[data-testid="stSegmentedControl"] button {
        white-space: nowrap !important;
        flex: 0 0 auto !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] label:has(input:checked) {
        background: var(--ms-accent-soft) !important;
        color: var(--ms-text) !important;
        border-color: var(--ms-accent) !important;
        box-shadow: inset 0 0 0 1px var(--ms-accent) !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] *,
    div[data-testid="stSegmentedControl"] label:has(input:checked) * {
        color: var(--ms-text) !important;
        fill: var(--ms-text) !important;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stDataEditor"] {
        border-radius: 0.72rem;
        overflow: hidden;
        border: 1px solid var(--ms-control-border);
        background: #ffffff !important;
        --gdg-bg-cell: #ffffff !important;
        --gdg-bg-cell-medium: #ffffff !important;
        --gdg-bg-header: #fafafa !important;
        --gdg-bg-header-hovered: #f4f4f1 !important;
        --gdg-bg-header-has-focus: #f4f4f1 !important;
        --gdg-bg-bubble: #ffffff !important;
        --gdg-bg-bubble-selected: #f4f4f1 !important;
        --gdg-accent-color: var(--ms-accent) !important;
        --gdg-accent-light: rgba(95, 95, 88, 0.14) !important;
    }
    div[data-testid="stDataFrame"] > div,
    div[data-testid="stDataEditor"] > div,
    div[data-testid="stDataFrame"] canvas,
    div[data-testid="stDataEditor"] canvas {
        background: #ffffff !important;
        background-color: #ffffff !important;
    }


    .modes-table-wrap {
        margin-top: 0.35rem;
        max-height: 18.5rem;
        overflow-y: auto;
        border: 1px solid var(--ms-control-border);
        border-radius: 0.78rem;
        background: #ffffff;
    }
    table.modes-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 0.86rem;
    }
    table.modes-table th {
        position: sticky;
        top: 0;
        z-index: 1;
        text-align: right;
        padding: 0.58rem 0.66rem;
        background: #fafafa;
        color: var(--ms-muted);
        font-weight: 760;
        border-bottom: 1px solid var(--ms-control-border);
    }
    table.modes-table th:first-child, table.modes-table td:first-child {
        text-align: left;
        width: 4.5rem;
    }
    table.modes-table td {
        text-align: right;
        padding: 0.54rem 0.66rem;
        border-bottom: 1px solid #eeeeee;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        white-space: nowrap;
        color: var(--ms-text);
        background: #ffffff;
    }
    table.modes-table tr.active-mode td {
        background: #f1f1f1;
        font-weight: 740;
    }
    table.modes-table tr:last-child td {border-bottom: 0;}

    .mode-select-table {
        margin-top: 0.35rem;
        border: 1px solid var(--ms-control-border);
        border-radius: 0.78rem 0.78rem 0 0;
        background: #ffffff;
        overflow: hidden;
    }
    .mode-select-header {
        display: grid;
        grid-template-columns: 0.72fr 0.92fr 1.18fr 1.08fr 1.05fr 0.82fr 0.82fr 0.82fr;
        gap: 0.35rem;
        padding: 0.56rem 0.66rem;
        background: #fafafa;
        color: var(--ms-muted);
        font-size: 0.82rem;
        font-weight: 760;
        border-bottom: 1px solid var(--ms-control-border);
    }
    .mode-select-row {
        margin: 0;
        padding: 0.08rem 0.66rem;
        background: #ffffff;
        border-left: 1px solid var(--ms-control-border);
        border-right: 1px solid var(--ms-control-border);
        border-bottom: 1px solid #eeeeee;
    }
    .active-mode-row {
        background: #f4f4f1;
    }
    .download-row-spacer {height: 0.15rem;}


    /* Neutral Streamlit controls. Keep MESH radio readable without red or black-filled buttons. */
    :root,
    html,
    body,
    .stApp {
        --primary-color: var(--ms-accent) !important;
        --secondary-background-color: #ffffff !important;
        accent-color: var(--ms-accent) !important;
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label * {
        color: var(--ms-text) !important;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        -webkit-appearance: none !important;
        appearance: none !important;
        width: 1.06rem !important;
        height: 1.06rem !important;
        min-width: 1.06rem !important;
        min-height: 1.06rem !important;
        border-radius: 999px !important;
        border: 1.6px solid #c7c7c1 !important;
        background: #ffffff !important;
        box-shadow: inset 0 0 0 3.5px #ffffff !important;
        margin: 0 0.48rem 0 0 !important;
        vertical-align: -0.12rem !important;
        cursor: pointer !important;
    }
    div[data-testid="stRadio"] input[type="radio"]:checked {
        border-color: #8d8d86 !important;
        background: var(--ms-accent) !important;
        box-shadow: inset 0 0 0 4px #ffffff, 0 0 0 2px rgba(95, 95, 88, 0.14) !important;
    }
    div[data-testid="stRadio"] input[type="radio"]:focus-visible {
        outline: 2px solid rgba(95, 95, 88, 0.30) !important;
        outline-offset: 2px !important;
    }
    div[data-testid="stRadio"] svg {
        color: var(--ms-accent) !important;
        fill: var(--ms-accent) !important;
    }


    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child,
    div[data-testid="stRadio"] label[data-baseweb="radio"] > span:first-child,
    div[data-testid="stRadio"] label > div:first-child,
    div[data-testid="stRadio"] label > span:first-child {
        position: relative !important;
        background: #ffffff !important;
        border-color: #c7c7c1 !important;
        box-shadow: none !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > span:first-child,
    div[data-testid="stRadio"] label:has(input:checked) > div:first-child,
    div[data-testid="stRadio"] label:has(input:checked) > span:first-child {
        background: #ffffff !important;
        border-color: #8d8d86 !important;
        box-shadow: 0 0 0 2px rgba(95, 95, 88, 0.14) !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child::after,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > span:first-child::after,
    div[data-testid="stRadio"] label:has(input:checked) > div:first-child::after,
    div[data-testid="stRadio"] label:has(input:checked) > span:first-child::after {
        content: "" !important;
        position: absolute !important;
        left: 50% !important;
        top: 50% !important;
        width: 0.48rem !important;
        height: 0.48rem !important;
        transform: translate(-50%, -50%) !important;
        border-radius: 999px !important;
        background: var(--ms-accent) !important;
        pointer-events: none !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) svg,
    div[data-testid="stRadio"] label:has(input:checked) svg * {
        color: var(--ms-accent) !important;
        fill: var(--ms-accent) !important;
        stroke: var(--ms-accent) !important;
    }

    div[data-testid="stCheckbox"] input[type="checkbox"],
    div[data-testid="stDataFrame"] input[type="checkbox"],
    div[data-testid="stDataEditor"] input[type="checkbox"] {
        accent-color: var(--ms-accent) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    /* ModeStudio redesign layer ------------------------------------------------- */
    :root {
        --ms-page: #f5f7fb;
        --ms-page-2: #eef2f7;
        --ms-panel: rgba(255, 255, 255, 0.92);
        --ms-panel-solid: #ffffff;
        --ms-panel-soft: #f8fafc;
        --ms-text: #111827;
        --ms-muted: #667085;
        --ms-muted-2: #98a2b3;
        --ms-border: #e2e8f0;
        --ms-border-strong: #cbd5e1;
        --ms-control: #ffffff;
        --ms-control-soft: #f8fafc;
        --ms-control-hover: #f1f5f9;
        --ms-control-border: #d7dee8;
        --ms-accent: #2563eb;
        --ms-accent-hover: #1d4ed8;
        --ms-accent-soft: #dbeafe;
        --ms-accent-ring: rgba(37, 99, 235, 0.18);
        --ms-red: #111827;
        --ms-red-dark: #020617;
        --ms-radius-lg: 20px;
        --ms-radius-md: 14px;
        --ms-shadow-panel: 0 18px 48px rgba(15, 23, 42, 0.075);
        --ms-shadow-soft: 0 8px 22px rgba(15, 23, 42, 0.055);
    }

    html, body,
    .stApp,
    div[data-testid="stAppViewContainer"],
    section.main {
        background: var(--ms-page) !important;
        color: var(--ms-text) !important;
        font-feature-settings: "kern" 1, "liga" 1;
    }

    .block-container {
        padding-top: 0.16rem !important;
        padding-left: 1.10rem !important;
        padding-right: 1.10rem !important;
        padding-bottom: 0.90rem !important;
        max-width: 1860px !important;
    }

    .studio-brand {display: flex; flex-direction: column; gap: 0.16rem; min-width: 0;}
    .studio-topbar-marker {display: none !important;}
    .native-topbar-brand {
        padding: 0.02rem 0 0.02rem 0.08rem;
    }
    .project-current-label {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.55rem;
        color: #475467;
        font-size: 0.78rem;
        line-height: 1.1;
        white-space: nowrap;
        min-width: 0;
    }
    .project-current-label span {
        color: #667085;
        font-weight: 760;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .project-current-label strong {
        max-width: 18rem;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #111827;
        font-weight: 680;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) {
        border-radius: 14px !important;
        background: rgba(255, 255, 255, 0.94) !important;
        border: 1px solid rgba(226, 232, 240, 0.92) !important;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05) !important;
        margin-top: 0 !important;
        margin-bottom: 0.56rem !important;
        overflow: visible !important;
        backdrop-filter: blur(18px) saturate(1.16) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) > div > div {
        background: transparent !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) div[data-testid="stVerticalBlock"] {
        gap: 0.04rem !important;
        padding: 0.04rem 0.28rem 0.04rem 0.28rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) div[data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 0.48rem !important;
        flex-wrap: nowrap !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) [data-testid="stTooltipHoverTarget"] {
        display: none !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) div[data-baseweb="input"],
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) div[data-baseweb="input"] input {
        min-height: 1.96rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) .stButton > button,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) .stDownloadButton > button {
        height: 2.04rem !important;
        min-height: 2.04rem !important;
        white-space: nowrap !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) div[data-testid="stPopover"] {
        min-width: 8.7rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) div[data-testid="stPopover"] button {
        height: 2.04rem !important;
        min-height: 2.04rem !important;
        min-width: 8.7rem !important;
        white-space: nowrap !important;
        padding-left: 0.88rem !important;
        padding-right: 0.88rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) div[data-testid="stPopover"] button *,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) div[data-testid="stPopover"] button p {
        white-space: nowrap !important;
        line-height: 1.05 !important;
    }
    .studio-title {
        width: fit-content;
        font-size: clamp(0.94rem, 0.82vw, 1.08rem);
        line-height: 1.0;
        font-weight: 790;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        background: linear-gradient(92deg, #111827 0%, #1d4ed8 46%, #475569 100%);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        -webkit-text-fill-color: transparent;
    }
    .studio-subtitle {
        font-size: 0.70rem;
        color: #64748b;
        font-weight: 560;
        letter-spacing: -0.01em;
    }

    div[data-testid="stHorizontalBlock"] {gap: 0.86rem !important;}
    div[data-testid="stVerticalBlock"] {gap: 0.46rem !important;}

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(226, 232, 240, 0.92) !important;
        border-radius: 16px !important;
        background: rgba(255, 255, 255, 0.78) !important;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04) !important;
        backdrop-filter: blur(18px) saturate(1.2) !important;
        overflow: hidden !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div,
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div {
        border-radius: inherit !important;
        background: transparent !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
        padding: 0.10rem 0.08rem 0.06rem 0.08rem;
    }

    .panel-label,
    .table-label {
        font-size: 0.67rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: #667085 !important;
        font-weight: 800 !important;
        margin: 0.06rem 0 0.20rem 0 !important;
    }
    .table-label {
        margin-top: 0.34rem !important;
        margin-bottom: 0.26rem !important;
    }

    label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stMultiSelect"] label,
    div[data-testid="stFileUploader"] label {
        font-size: 0.78rem !important;
        color: #344054 !important;
        font-weight: 670 !important;
        letter-spacing: -0.01em !important;
    }

    div[data-baseweb="input"],
    div[data-baseweb="select"] > div,
    textarea,
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        border-radius: 12px !important;
        border: 1px solid var(--ms-control-border) !important;
        background: rgba(255,255,255,0.96) !important;
        box-shadow: 0 1px 0 rgba(15,23,42,0.025) !important;
        min-height: 2.14rem !important;
    }
    div[data-baseweb="input"]:hover,
    div[data-baseweb="select"] > div:hover,
    textarea:hover {
        border-color: #b9c2cf !important;
        background: #ffffff !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    textarea:focus {
        border-color: var(--ms-accent) !important;
        box-shadow: 0 0 0 3px var(--ms-accent-ring) !important;
    }
    div[data-baseweb="input"] input,
    div[data-testid="stNumberInput"] input,
    textarea,
    textarea:focus {
        color: var(--ms-text) !important;
        font-weight: 520 !important;
    }
    div[data-testid="stNumberInput"] button {
        border-left: 1px solid #e5e7eb !important;
        color: #475467 !important;
        background: #ffffff !important;
    }
    div[data-testid="stNumberInput"] button:hover {background: #f8fafc !important;}

    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        min-height: 2.72rem !important;
        border: 1px dashed #c8d1dd !important;
        border-radius: 14px !important;
        background: #ffffff !important;
        padding: 0.36rem 0.52rem !important;
    }
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
        color: var(--ms-muted) !important;
    }

    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stFileUploader"] button {
        height: 2.12rem !important;
        border-radius: 999px !important;
        border: 1px solid var(--ms-control-border) !important;
        background: #ffffff !important;
        color: #1f2937 !important;
        font-weight: 760 !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.045) !important;
        transition: transform 120ms ease, border-color 120ms ease, background 120ms ease, box-shadow 120ms ease !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stFileUploader"] button:hover {
        transform: translateY(-1px);
        background: #f8fafc !important;
        border-color: #b8c3d0 !important;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08) !important;
    }
    .stButton > button[kind="primary"] {
        min-height: 2.30rem !important;
        border: 0 !important;
        background: #111827 !important;
        color: #ffffff !important;
        box-shadow: 0 10px 18px rgba(17, 24, 39, 0.14) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #0f172a !important;
        color: #ffffff !important;
    }

    div[data-testid="stSegmentedControl"] > div {
        padding: 0.22rem !important;
        gap: 0.16rem !important;
        border: 1px solid var(--ms-border) !important;
        border-radius: 14px !important;
        background: #f8fafc !important;
        box-shadow: inset 0 1px 2px rgba(15,23,42,0.035) !important;
    }
    div[data-testid="stSegmentedControl"] button,
    div[data-testid="stSegmentedControl"] label {
        border: 0 !important;
        border-radius: 10px !important;
        color: #475467 !important;
        font-weight: 720 !important;
        min-height: 1.80rem !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] label:has(input:checked) {
        background: #ffffff !important;
        color: #111827 !important;
        border-color: transparent !important;
        box-shadow: 0 1px 5px rgba(15,23,42,0.10) !important;
    }

    div[data-testid="stRadio"] label {gap: 0.30rem !important;}
    div[data-testid="stRadio"] input[type="radio"] {
        border-color: #cbd5e1 !important;
        box-shadow: inset 0 0 0 4px #ffffff !important;
    }
    div[data-testid="stRadio"] input[type="radio"]:checked {
        border-color: var(--ms-accent) !important;
        background: var(--ms-accent) !important;
        box-shadow: inset 0 0 0 4px #ffffff, 0 0 0 3px var(--ms-accent-ring) !important;
    }
    div[data-testid="stCheckbox"] input[type="checkbox"],
    div[data-testid="stDataFrame"] input[type="checkbox"],
    div[data-testid="stDataEditor"] input[type="checkbox"] {
        accent-color: var(--ms-accent) !important;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background: #111827 !important;
        border-radius: 9px !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(226,232,240,0.95) !important;
        border-radius: 14px !important;
        background: #f8fafc !important;
    }
    div[data-testid="stExpander"] > details,
    div[data-testid="stExpander"] summary {
        background: transparent !important;
    }

    .stCaptionContainer,
    .stCaptionContainer p,
    small,
    .subtle-note {
        color: var(--ms-muted) !important;
        font-size: 0.78rem !important;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stDataEditor"] {
        border: 1px solid var(--ms-border) !important;
        border-radius: 14px !important;
        background: #ffffff !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.9) !important;
        --gdg-bg-cell: #ffffff !important;
        --gdg-bg-cell-medium: #f8fafc !important;
        --gdg-bg-header: #f8fafc !important;
        --gdg-bg-header-hovered: #eef2f7 !important;
        --gdg-bg-header-has-focus: #eef2f7 !important;
        --gdg-text-dark: #111827 !important;
        --gdg-text-medium: #475467 !important;
        --gdg-accent-color: var(--ms-accent) !important;
        --gdg-accent-light: var(--ms-accent-ring) !important;
    }

    .modes-table-wrap,
    .mode-select-table {
        border: 1px solid var(--ms-border) !important;
        border-radius: 14px !important;
        background: #ffffff !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.88) !important;
    }
    table.modes-table th,
    .mode-select-header {
        background: #f8fafc !important;
        color: #667085 !important;
        border-bottom: 1px solid var(--ms-border) !important;
    }
    table.modes-table td {
        border-bottom: 1px solid #eef2f7 !important;
        color: #111827 !important;
    }
    table.modes-table tr.active-mode td,
    .active-mode-row {
        background: #eff6ff !important;
    }

    .element-container:has(.ms-canvas-viewer) {
        border-radius: 20px !important;
        overflow: hidden !important;
    }



    .studio-workspace-marker,
    .studio-left-scroll-marker,
    .studio-main-scroll-marker {
        display: none !important;
    }

    /* Two-pane workspace ------------------------------------------------------
       Keep Streamlit's normal page scrolling as a fallback.  Only the two
       marked pane contents get a viewport-height cap and their own scrollbars.
       Do not set overflow:hidden on the app, main section, or block-container;
       doing so can trap the page when Streamlit's DOM changes. */
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) {
        align-items: stretch !important;
        min-height: 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) > div[data-testid="column"] {
        min-width: 0 !important;
        min-height: 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-left-scroll-marker),
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-main-scroll-marker) {
        max-height: calc(100vh - 5.7rem) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 0.18rem !important;
        padding-bottom: 0.55rem !important;
        scrollbar-gutter: stable !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-left-scroll-marker) {
        gap: 0.70rem !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-left-scroll-marker)::-webkit-scrollbar,
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-main-scroll-marker)::-webkit-scrollbar {
        width: 10px;
    }
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-left-scroll-marker)::-webkit-scrollbar-track,
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-main-scroll-marker)::-webkit-scrollbar-track {
        background: transparent;
    }
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-left-scroll-marker)::-webkit-scrollbar-thumb,
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-main-scroll-marker)::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.48);
        border-radius: 999px;
        border: 3px solid transparent;
        background-clip: content-box;
    }
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-left-scroll-marker)::-webkit-scrollbar-thumb:hover,
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-main-scroll-marker)::-webkit-scrollbar-thumb:hover {
        background: rgba(100, 116, 139, 0.60);
        border: 3px solid transparent;
        background-clip: content-box;
    }
    @media (max-height: 820px) {
        div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-left-scroll-marker),
        div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-main-scroll-marker) {
            max-height: calc(100vh - 5.1rem) !important;
        }
    }
    @media (max-width: 1180px) {
        .studio-subtitle {display: none !important;}
        .project-current-label strong {max-width: 10rem !important;}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) .stButton > button,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) .stDownloadButton > button,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) div[data-testid="stPopover"] button {
            padding-left: 0.74rem !important;
            padding-right: 0.74rem !important;
        }
    }
    @media (max-width: 980px) {
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) div[data-testid="stHorizontalBlock"] {align-items: center !important;}
        .project-current-label {display: none !important;}
        .block-container {padding-left: 0.75rem !important; padding-right: 0.75rem !important;}
        div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-left-scroll-marker),
        div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-main-scroll-marker) {
            max-height: none !important;
            overflow: visible !important;
        }
    }

    /* v90 density pass: keep text readable while reducing oversized control boxes. */
    div[data-baseweb="input"],
    div[data-baseweb="select"] > div,
    textarea,
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        min-height: 2.02rem !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="input"] input,
    div[data-testid="stNumberInput"] input {
        min-height: 2.02rem !important;
        line-height: 1.15 !important;
    }
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stFileUploader"] button {
        height: 2.02rem !important;
        min-height: 2.02rem !important;
        padding-top: 0.20rem !important;
        padding-bottom: 0.20rem !important;
    }
    .stButton > button[kind="primary"] {
        min-height: 2.18rem !important;
        height: 2.18rem !important;
    }
    div[data-testid="stSegmentedControl"] > div {
        padding: 0.16rem !important;
        border-radius: 12px !important;
    }
    div[data-testid="stSegmentedControl"] button,
    div[data-testid="stSegmentedControl"] label {
        min-height: 1.72rem !important;
        border-radius: 9px !important;
    }
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        min-height: 2.48rem !important;
        padding: 0.26rem 0.46rem !important;
        border-radius: 12px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
        padding: 0.12rem 0.08rem 0.08rem 0.08rem;
    }
    div[data-testid="stVerticalBlock"] {gap: 0.56rem !important;}
    div[data-testid="stHorizontalBlock"] {gap: 0.86rem !important;}
    .panel-label,
    .table-label {
        margin-top: 0.04rem !important;
        margin-bottom: 0.22rem !important;
    }
    .table-label {
        margin-top: 0.38rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) div[data-testid="stPopover"] button {
        height: 2.04rem !important;
        min-height: 2.04rem !important;
    }


    /* v91 tool-style workspace: replace stacked cards with continuous panes. */
    .studio-settings-pane-marker,
    .studio-results-pane-marker {
        display: none !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) {
        border-radius: 16px !important;
        border: 0 !important;
        background: rgba(255, 255, 255, 0.62) !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
        overflow: visible !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) > div > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) > div > div {
        background: transparent !important;
        border-radius: inherit !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) div[data-testid="stVerticalBlock"] {
        gap: 0.28rem !important;
        padding: 0.38rem 0.44rem 0.46rem 0.44rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) div[data-testid="stVerticalBlock"] {
        gap: 0.32rem !important;
        padding: 0.42rem 0.48rem 0.50rem 0.48rem !important;
    }
    .panel-section-rule {
        height: 1px;
        background: rgba(226, 232, 240, 0.92);
        margin: 0.40rem 0 0.12rem 0;
    }
    .panel-run-space {
        height: 0.08rem;
    }
    .first-panel-label {
        margin-top: 0 !important;
    }

    .studio-page-brand-anchor {
        height: 1.58rem !important;
        min-height: 1.58rem !important;
        position: relative !important;
        z-index: 20 !important;
        pointer-events: none !important;
        margin: 0.10rem 0 0.36rem 0 !important;
    }
    .studio-page-brand {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.58rem !important;
        white-space: nowrap !important;
        line-height: 1.05 !important;
        padding-left: 0.04rem !important;
    }
    .studio-page-logo {
        font-size: 0.90rem !important;
        line-height: 1 !important;
        letter-spacing: 0.11em !important;
        font-weight: 860 !important;
        color: #111827 !important;
    }
    .studio-page-powered {
        font-size: 0.60rem !important;
        line-height: 1 !important;
        letter-spacing: 0.035em !important;
        font-weight: 620 !important;
        color: #6b7280 !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) .panel-label,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) .panel-label,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) .table-label {
        color: #475569 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) div[data-testid="stExpander"] {
        border-radius: 10px !important;
        border-color: rgba(226, 232, 240, 0.86) !important;
        background: rgba(255,255,255,0.84) !important;
        box-shadow: none !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.72) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) .ms-canvas-viewer {
        border-radius: 14px !important;
        box-shadow: none !important;
    }
    /* v93 compact refinements */
    .studio-archive-dock-marker {display: none !important;}
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-archive-dock-marker) {
        position: fixed !important;
        right: 1.05rem !important;
        bottom: 1.0rem !important;
        z-index: 1000 !important;
        width: auto !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
        overflow: visible !important;
        margin: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-archive-dock-marker) > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-archive-dock-marker) > div > div {
        background: transparent !important;
        border-radius: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-archive-dock-marker) div[data-testid="stVerticalBlock"] {
        padding: 0 !important;
        gap: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-archive-dock-marker) div[data-testid="stPopover"] button {
        height: 2.08rem !important;
        min-height: 2.08rem !important;
        min-width: 7.5rem !important;
        border-radius: 999px !important;
        border: 1px solid rgba(215, 222, 232, 0.98) !important;
        background: rgba(255,255,255,0.96) !important;
        color: #111827 !important;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.14) !important;
        backdrop-filter: blur(16px) !important;
        font-weight: 760 !important;
        padding-left: 0.95rem !important;
        padding-right: 0.95rem !important;
        white-space: nowrap !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-archive-dock-marker) div[data-testid="stPopover"] button:hover {
        background: #ffffff !important;
        border-color: rgba(191, 203, 217, 1.0) !important;
        transform: translateY(-1px);
    }

    .studio-archive-inline-marker {display: none !important;}

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) div[data-testid="stPopover"] button {
        height: 2.10rem !important;
        min-height: 2.10rem !important;
        white-space: nowrap !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) .studio-archive-inline-marker + div[data-testid="stPopover"] button,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) div[data-testid="stPopover"]:has(+ .studio-archive-inline-marker) button {
        border-radius: 999px !important;
    }
    @media (max-width: 900px) {
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-archive-dock-marker) {
            right: 0.75rem !important;
            bottom: 0.75rem !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-archive-dock-marker) div[data-testid="stPopover"] button {
            min-width: 6.8rem !important;
        }
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) {
        backdrop-filter: none !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) .stButton > button,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) .stButton > button,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) .stDownloadButton > button,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) .stDownloadButton > button {
        font-size: 0.90rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) .stButton > button,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) .stDownloadButton > button,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) div[data-testid="stPopover"] button {
        border-radius: 12px !important;
        box-shadow: none !important;
        border-color: rgba(215, 222, 232, 0.95) !important;
        background: rgba(255,255,255,0.96) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) .project-current-label {
        font-size: 0.75rem !important;
        gap: 0.45rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-topbar-marker) .project-current-label strong {
        max-width: 14rem !important;
    }

    /* v99: balanced Full-HD density. Keep spacing readable and target only bulky controls. */
    @media (max-height: 1120px) {
        .block-container {
            padding-top: 0.10rem !important;
            padding-bottom: 0.42rem !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) div[data-testid="stVerticalBlock"] {
            gap: 0.24rem !important;
            padding: 0.34rem 0.40rem 0.40rem 0.40rem !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) div[data-testid="stVerticalBlock"] {
            gap: 0.30rem !important;
            padding: 0.40rem 0.46rem 0.46rem 0.46rem !important;
        }
        .panel-section-rule {
            margin: 0.34rem 0 0.10rem 0 !important;
            height: 1px !important;
            background: rgba(226, 232, 240, 0.72) !important;
        }
        .panel-run-space {height: 0.02rem !important;}
        div[data-baseweb="input"],
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"],
        div[data-baseweb="base-input"] > div {
            min-height: 1.88rem !important;
        }
        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"] input,
        div[data-testid="stNumberInput"] input {
            min-height: 1.88rem !important;
            line-height: 1.08 !important;
        }
        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stFileUploader"] button,
        div[data-testid="stPopover"] button {
            height: 1.90rem !important;
            min-height: 1.90rem !important;
            padding-top: 0.12rem !important;
            padding-bottom: 0.12rem !important;
        }
        .stButton > button[kind="primary"] {
            height: 2.04rem !important;
            min-height: 2.04rem !important;
        }
        div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
            min-height: 2.16rem !important;
            padding: 0.20rem 0.40rem !important;
        }
        div[data-testid="stSegmentedControl"] > div {
            padding: 0.12rem !important;
        }
        div[data-testid="stSegmentedControl"] button,
        div[data-testid="stSegmentedControl"] label {
            min-height: 1.58rem !important;
        }
        div[role="radiogroup"] label {
            min-height: 1.52rem !important;
            padding-top: 0.04rem !important;
            padding-bottom: 0.04rem !important;
        }
        .stCaptionContainer,
        .stCaptionContainer p,
        div[data-testid="stCaptionContainer"],
        div[data-testid="stCaptionContainer"] p {
            line-height: 1.18 !important;
            margin-top: 0.03rem !important;
            margin-bottom: 0.03rem !important;
        }
    }



    /* v103 color refresh: neutral surfaces and pane separation without boxed borders. */
    :root {
        --ms-page: #f7f7f5;
        --ms-page-2: #efefec;
        --ms-panel: rgba(255, 255, 255, 0.78);
        --ms-panel-solid: #ffffff;
        --ms-panel-soft: #f3f3f1;
        --ms-text: #111111;
        --ms-muted: #686864;
        --ms-muted-2: #9a9a94;
        --ms-border: rgba(17, 24, 39, 0.08);
        --ms-border-strong: rgba(17, 24, 39, 0.14);
        --ms-control: #ffffff;
        --ms-control-soft: #f2f2ef;
        --ms-control-border: rgba(17, 24, 39, 0.14);
        --ms-accent: #111111;
        --ms-accent-soft: #ededeb;
        --ms-accent-ring: rgba(17, 17, 17, 0.12);
    }

    html, body,
    .stApp,
    div[data-testid="stAppViewContainer"],
    section.main {
        background: linear-gradient(180deg, #f7f7f5 0%, #f1f1ee 100%) !important;
        color: var(--ms-text) !important;
    }

    .block-container {
        background: transparent !important;
    }

    /* Large panes should read as surfaces, not framed boxes. */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) {
        border: 0 !important;
        background: rgba(255, 255, 255, 0.82) !important;
        box-shadow:
            0 18px 48px rgba(17, 24, 39, 0.070),
            0 1px 0 rgba(255, 255, 255, 0.74) inset !important;
        backdrop-filter: blur(18px) saturate(1.06) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) > div > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) > div > div {
        border: 0 !important;
        background: transparent !important;
    }

    /* Section separation: remove ruled-line feel and rely on typography/space. */
    .panel-section-rule {
        height: 0 !important;
        background: transparent !important;
        margin: 0.44rem 0 0.08rem 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) .panel-label,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) .panel-label,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) .table-label {
        color: #5f5f5b !important;
    }
    .studio-page-logo {
        color: #111111 !important;
    }
    .studio-page-powered {
        color: #8a8a84 !important;
    }

    /* Controls still need affordance, but keep them quiet and neutral. */
    div[data-baseweb="input"],
    div[data-baseweb="select"] > div,
    textarea,
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        border-color: rgba(17, 24, 39, 0.13) !important;
        background: rgba(255, 255, 255, 0.96) !important;
        box-shadow: none !important;
    }
    div[data-baseweb="input"]:hover,
    div[data-baseweb="select"] > div:hover,
    textarea:hover {
        border-color: rgba(17, 24, 39, 0.22) !important;
        background: #ffffff !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    textarea:focus {
        border-color: rgba(17, 17, 17, 0.70) !important;
        box-shadow: 0 0 0 3px rgba(17, 17, 17, 0.08) !important;
    }

    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        border: 0 !important;
        background: rgba(17, 24, 39, 0.045) !important;
        box-shadow: inset 0 0 0 1px rgba(17, 24, 39, 0.035) !important;
    }
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover {
        background: rgba(17, 24, 39, 0.060) !important;
    }

    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stFileUploader"] button,
    div[data-testid="stPopover"] button {
        border-color: rgba(17, 24, 39, 0.12) !important;
        background: rgba(255, 255, 255, 0.92) !important;
        color: #171717 !important;
        box-shadow: none !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stFileUploader"] button:hover,
    div[data-testid="stPopover"] button:hover {
        border-color: rgba(17, 24, 39, 0.18) !important;
        background: #ffffff !important;
        box-shadow: 0 6px 18px rgba(17, 24, 39, 0.060) !important;
    }
    .stButton > button[kind="primary"] {
        background: #111111 !important;
        color: #ffffff !important;
        box-shadow: 0 10px 24px rgba(17, 17, 17, 0.12) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #000000 !important;
        color: #ffffff !important;
    }

    div[data-testid="stSegmentedControl"] > div {
        border: 0 !important;
        background: rgba(17, 24, 39, 0.055) !important;
        box-shadow: none !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] label:has(input:checked) {
        background: #ffffff !important;
        color: #111111 !important;
        box-shadow: 0 5px 14px rgba(17, 24, 39, 0.080) !important;
    }

    div[data-testid="stExpander"] {
        border: 0 !important;
        background: rgba(17, 24, 39, 0.045) !important;
        box-shadow: none !important;
    }

    .ms-canvas-viewer {
        border: 0 !important;
        box-shadow: none !important;
    }
    .element-container:has(.ms-canvas-viewer) {
        box-shadow: none !important;
        background: transparent !important;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stDataEditor"],
    .modes-table-wrap,
    .mode-select-table {
        border-color: rgba(17, 24, 39, 0.09) !important;
        box-shadow: none !important;
    }
    table.modes-table th,
    .mode-select-header {
        background: #f4f4f2 !important;
        border-bottom-color: rgba(17, 24, 39, 0.08) !important;
    }
    table.modes-table td {
        border-bottom-color: rgba(17, 24, 39, 0.055) !important;
    }
    table.modes-table tr.active-mode td,
    .active-mode-row {
        background: #e5e5e5 !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-left-scroll-marker)::-webkit-scrollbar-thumb,
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-main-scroll-marker)::-webkit-scrollbar-thumb {
        background: rgba(17, 24, 39, 0.22);
        border: 3px solid transparent;
        background-clip: content-box;
    }



    /* v104: remove the visible rounded-card perimeter; separate areas by filled surfaces only. */
    :root {
        --ms-page: #f5f5f5;
        --ms-page-2: #efefef;
        --ms-panel: #efefef;
        --ms-panel-solid: #efefef;
        --ms-panel-soft: #f7f7f7;
        --ms-border: rgba(17, 24, 39, 0.055);
        --ms-border-strong: rgba(17, 24, 39, 0.10);
    }
    html, body,
    .stApp,
    div[data-testid="stAppViewContainer"],
    section.main {
        background: #f5f5f5 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) {
        border: 0 !important;
        border-radius: 0 !important;
        background: #efefef !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
        outline: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) {
        background: #efefef !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) > div > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) > div > div {
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) div[data-testid="stVerticalBlock"] {
        border-radius: 0 !important;
    }
    .panel-section-rule {
        height: 0 !important;
        background: transparent !important;
        margin: 0.42rem 0 0.06rem 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) div[data-testid="stExpander"] {
        border-radius: 10px !important;
        background: rgba(255,255,255,0.58) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) .ms-canvas-viewer {
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    /* v105: remove Streamlit border containers from the main panes.
       The left and right columns themselves become filled surfaces, so there is no rounded frame to fight. */
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) > div[data-testid="column"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-left-scroll-marker) {
        background: #efefef !important;
        border: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        outline: 0 !important;
        padding: 0.42rem 0.48rem 0.48rem 0.48rem !important;
        gap: 0.26rem !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-main-scroll-marker) {
        background: #efefef !important;
        border: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        outline: 0 !important;
        padding: 0.42rem 0.50rem 0.50rem 0.50rem !important;
        gap: 0.30rem !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stVerticalBlock"]:has(.studio-settings-pane-marker),
    div[data-testid="stVerticalBlock"]:has(.studio-results-pane-marker) {
        background: transparent !important;
        border: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        outline: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) {
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker) > div > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) > div > div {
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-left-scroll-marker)::-webkit-scrollbar-track,
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-main-scroll-marker)::-webkit-scrollbar-track {
        background: transparent !important;
    }


    
    /* v110: use the built-in wavelength select slider consistently in both views. */
    div[data-testid="stSelectSlider"] {
        margin-top: 0.10rem !important;
        margin-bottom: 0.46rem !important;
    }
    div[data-testid="stSelectSlider"] label {
        margin-bottom: 0.32rem !important;
    }
    div[data-testid="stSelectSlider"] label p {
        font-size: 0.80rem !important;
        line-height: 1.1 !important;
        letter-spacing: 0.10em !important;
        text-transform: uppercase !important;
        font-weight: 760 !important;
        color: #4b5563 !important;
    }
    div[data-testid="stSelectSlider"] [data-testid="stTickBarMin"],
    div[data-testid="stSelectSlider"] [data-testid="stTickBarMax"] {
        color: #111827 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }

    /* v112: strict neutral monochrome and canvas graph typography. */
    :root {
        --ms-page: #f5f5f5 !important;
        --ms-page-2: #efefef !important;
        --ms-panel: #efefef !important;
        --ms-panel-solid: #efefef !important;
        --ms-panel-soft: #f7f7f7 !important;
        --ms-border: rgba(0, 0, 0, 0.08) !important;
        --ms-border-strong: rgba(0, 0, 0, 0.14) !important;
    }
    html, body,
    .stApp,
    div[data-testid="stAppViewContainer"],
    section.main {
        background: #f5f5f5 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-left-scroll-marker),
    div[data-testid="stHorizontalBlock"]:has(.studio-left-scroll-marker):has(.studio-main-scroll-marker) div[data-testid="stVerticalBlock"]:has(.studio-main-scroll-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-settings-pane-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-results-pane-marker) {
        background: #efefef !important;
    }
    .ms-canvas-viewer {
        border: 1px solid rgba(0,0,0,0.10) !important;
        box-shadow: none !important;
    }
    .viewer-save,
    .viewer-help,
    .viewer-legend {
        border-color: rgba(0,0,0,0.10) !important;
        box-shadow: none !important;
    }
</style>
    """,
    unsafe_allow_html=True,
)


def handle_uploads() -> None:
    st.file_uploader('Section JSON', type=['json'], key='section_upload')
    st.file_uploader('Materials JSON', type=['json'], key='materials_upload')
    process_model_uploads_from_state(
        skip_processing=bool(st.session_state.get('_skip_model_upload_processing_this_run', False))
    )
    st.session_state.pop('_skip_model_upload_processing_this_run', None)

def parse_current_inputs() -> tuple[Any, Any, Any, Any, str | None]:
    try:
        section = loads_json(st.session_state['section_text'], label='Section')
        materials = loads_json(st.session_state['materials_text'], label='Materials')
        shapes, region_materials = build_shapes(section)
        return section, materials, shapes, region_materials, None
    except InputValidationError as exc:
        return None, None, None, None, str(exc)
    except Exception as exc:
        return None, None, None, None, str(exc)


def ensure_focus_regions(region_names: list[str], suggested_focus: list[str]) -> None:
    if 'focus_regions' not in st.session_state:
        st.session_state['focus_regions'] = suggested_focus
        return
    st.session_state['focus_regions'] = [name for name in st.session_state['focus_regions'] if name in region_names]
    if not st.session_state['focus_regions']:
        st.session_state['focus_regions'] = suggested_focus


def analysis_run_revision(results: dict) -> str:
    payload = {
        'window': results.get('analysis_window', {}),
        'mesh_elements': results.get('mesh_elements'),
        'wavelength_um': results.get('wavelength_um'),
        'num_modes': results.get('num_modes'),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:12]


def mode_field_pack(field_entry: dict) -> dict[str, Any]:
    if isinstance(field_entry, dict) and 'fields' in field_entry:
        return field_entry
    return {
        'mode_index': int(field_entry.get('mode_index', 0)) if isinstance(field_entry, dict) else 0,
        'default_field': 'intensity',
        'fields': {'intensity': field_entry},
    }


def mode_field_options(field_entry: dict) -> list[str]:
    fields = mode_field_pack(field_entry).get('fields', {})
    preferred = ['intensity', 'abs_Ex', 'abs_Ey', 'abs_Ez', 'abs_Hx', 'abs_Hy', 'abs_Hz']
    return [key for key in preferred if key in fields] + [key for key in fields.keys() if key not in preferred]


def field_option_label(field_entry: dict, key: str) -> str:
    return FIELD_QUANTITY_LABELS.get(key, key)


def unique_field_labels(field_entry: dict, keys: list[str]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    label_to_key: dict[str, str] = {}
    key_to_label: dict[str, str] = {}
    labels: list[str] = []
    for key in keys:
        base_label = field_option_label(field_entry, key)
        label = base_label
        if label in label_to_key and label_to_key[label] != key:
            label = f'{base_label} ({key})'
        label_to_key[label] = key
        key_to_label[key] = label
        labels.append(label)
    return labels, label_to_key, key_to_label


def selected_mode_from_table_state(table_key: str, mode_options: list[int], fallback: int) -> int:
    state = st.session_state.get(table_key)
    rows: list[int] = []
    try:
        if isinstance(state, dict):
            rows = list(state.get('selection', {}).get('rows', []))
        else:
            rows = list(getattr(getattr(state, 'selection', None), 'rows', []))
    except Exception:
        rows = []
    if rows:
        index = int(rows[0])
        if 0 <= index < len(mode_options):
            return int(mode_options[index])
    return int(fallback if fallback in mode_options else mode_options[0])


def render_modes_selection_table(results: dict, *, table_key: str, selected_mode: int) -> tuple[pd.DataFrame, int | None]:
    display_df = build_modes_display_dataframe(results)
    modes_df = build_modes_dataframe(results)
    if display_df.empty:
        st.caption('No mode result is available.')
        return modes_df, None

    selected_mode = int(selected_mode)
    display_df = display_df.copy()
    display_df.insert(0, 'Display', display_df['Mode'].astype(int).eq(selected_mode))

    editor_key = f'{table_key}_selected_{selected_mode}'
    styled_display_df = display_df.style.set_properties(**{'background-color': '#ffffff', 'color': '#0f0f0f'})
    edited = st.data_editor(
        styled_display_df,
        use_container_width=True,
        hide_index=True,
        height=table_height(len(display_df), max_rows=6),
        key=editor_key,
        disabled=[column for column in display_df.columns if column != 'Display'],
        column_config={
            'Display': st.column_config.CheckboxColumn('Display', width='small'),
            'Mode': st.column_config.NumberColumn('Mode', width='small', format='%d'),
            'n_eff': st.column_config.NumberColumn(PHYS_LABEL_PLAIN['n_eff'], width='small', format='%.6f'),
            'Im(n_eff)': st.column_config.NumberColumn(PHYS_LABEL_PLAIN['k_eff'], width='small', format='%.3e'),
            'Loss [dB/cm]': st.column_config.NumberColumn(PHYS_LABEL_PLAIN['loss_dB_per_cm'], width='small', format='%.3g'),
            'n_g': st.column_config.NumberColumn(PHYS_LABEL_PLAIN['group_index'], width='small', format='%.6f'),
            'TE': st.column_config.NumberColumn('TE', width='small', format='%.4f'),
            'TM': st.column_config.NumberColumn('TM', width='small', format='%.4f'),
            'P_ref': st.column_config.NumberColumn(PHYS_LABEL_PLAIN['reference_power_fraction'], width='small', format='%.4f'),
        },
    )

    try:
        checked_modes = [int(value) for value in edited.loc[edited['Display'].astype(bool), 'Mode'].tolist()]
    except Exception:
        checked_modes = []

    if not checked_modes:
        return modes_df, None

    newly_checked = [mode for mode in checked_modes if mode != selected_mode]
    return modes_df, (newly_checked[0] if newly_checked else checked_modes[0])



SWEEP_METRIC_OPTIONS = {
    'n_eff': PHYS_LABEL_PLAIN['n_eff'],
    'k_eff': PHYS_LABEL_PLAIN['k_eff'],
    'loss_dB_per_cm': PHYS_LABEL_PLAIN['loss_dB_per_cm'],
    'group_index': PHYS_LABEL_PLAIN['group_index'],
    'reference_power_fraction': PHYS_LABEL_PLAIN['reference_power_fraction'],
    'TE_fraction': 'TE',
    'TM_fraction': 'TM',
}


def is_sweep_results(results: dict | None) -> bool:
    if not isinstance(results, dict):
        return False
    sweep = results.get('wavelength_sweep')
    return results.get('analysis_type') == 'sweep' and isinstance(sweep, dict) and bool(sweep.get('runs'))


def sweep_runs(results: dict | None) -> list[dict[str, Any]]:
    if not is_sweep_results(results):
        return []
    runs = results.get('wavelength_sweep', {}).get('runs', [])
    return [run for run in runs if isinstance(run, dict)]


def sweep_wavelengths_nm(results: dict | None) -> list[float]:
    runs = sweep_runs(results)
    if runs:
        return [float(run.get('wavelength_nm', 0.0)) for run in runs]
    if is_sweep_results(results):
        return [float(value) for value in results.get('wavelength_sweep', {}).get('wavelengths_nm', [])]
    if isinstance(results, dict) and 'wavelength_nm' in results:
        return [float(results['wavelength_nm'])]
    return []


def clamp_sweep_index(results: dict | None, index: int | None = None) -> int:
    count = max(1, len(sweep_wavelengths_nm(results)))
    if index is None:
        index = int(st.session_state.get('selected_sweep_index', 0))
    return max(0, min(count - 1, int(index)))


def selected_run_results(analysis_output: dict | None, sweep_index: int | None = None) -> dict | None:
    if not analysis_output:
        return None
    results = analysis_output.get('results')
    if not is_sweep_results(results):
        return results
    runs = sweep_runs(results)
    if not runs:
        return results
    return runs[clamp_sweep_index(results, sweep_index)]


def selected_mode_field_maps(analysis_output: dict | None, sweep_index: int | None = None) -> list[dict[str, Any]]:
    if not analysis_output:
        return []
    results = analysis_output.get('results')
    if not is_sweep_results(results):
        return analysis_output.get('mode_field_maps', [])
    sweep_maps = analysis_output.get('sweep_field_maps', [])
    if not sweep_maps:
        return analysis_output.get('mode_field_maps', [])
    entry = sweep_maps[clamp_sweep_index(results, sweep_index)]
    return entry.get('mode_field_maps', []) if isinstance(entry, dict) else []


def render_wavelength_selector(results: dict | None, *, key: str = 'selected_sweep_index') -> int:
    wavelengths = sweep_wavelengths_nm(results)
    if len(wavelengths) <= 1:
        return 0
    options = list(range(len(wavelengths)))
    current = clamp_sweep_index(results)
    if key not in st.session_state or int(st.session_state.get(key, 0)) not in options:
        st.session_state[key] = current
    index = st.select_slider(
        'Wavelength',
        options=options,
        value=int(st.session_state.get(key, current)),
        key=key,
        format_func=lambda idx: f'{wavelengths[int(idx)]:.2f} nm',
    )
    return clamp_sweep_index(results, int(index))


def build_sweep_dataframe(results: dict | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sweep_index, run in enumerate(sweep_runs(results)):
        modes_df = build_modes_dataframe(run)
        if modes_df.empty:
            continue
        for row in modes_df.to_dict('records'):
            rows.append(
                {
                    'sweep_index': int(sweep_index),
                    'wavelength_nm': float(row.get('wavelength_nm', run.get('wavelength_nm', 0.0))),
                    'mode': int(row.get('mode', 0)),
                    'n_eff': float(row.get('n_eff', float('nan'))),
                    'group_index': float(row.get('group_index', float('nan'))),
                    'k_eff': float(row.get('k_eff', float('nan'))),
                    'loss_dB_per_cm': float(row.get('loss_dB_per_cm', float('nan'))),
                    'TE_fraction': float(row.get('TE_fraction', float('nan'))),
                    'TM_fraction': float(row.get('TM_fraction', float('nan'))),
                    'reference_power_fraction': float(row.get('reference_power_fraction', float('nan'))),
                }
            )
    return pd.DataFrame(rows)


def build_sweep_display_dataframe(results: dict | None, sweep_index: int) -> pd.DataFrame:
    run = None
    runs = sweep_runs(results)
    if runs:
        run = runs[clamp_sweep_index(results, sweep_index)]
    return build_modes_display_dataframe(run or {})


def sweep_metric_figure(
    sweep_df: pd.DataFrame,
    *,
    metric_key: str,
    selected_sweep_index: int,
    display_modes: list[int] | tuple[int, ...] | None = None,
) -> go.Figure:
    fig = go.Figure()
    metric_label = SWEEP_METRIC_OPTIONS.get(metric_key, metric_key)
    metric_label_html = PHYS_LABEL_HTML.get(metric_key, metric_label)
    selected_mode_set = {int(mode) for mode in (display_modes or [])}

    def apply_sweep_graph_style() -> None:
        x_values = [float(v) for v in sweep_df.get('wavelength_nm', pd.Series(dtype=float)).dropna().tolist()] if not sweep_df.empty else []
        x_range = None
        if x_values:
            x_min = min(x_values)
            x_max = max(x_values)
            span = max(x_max - x_min, 1.0)
            pad = span * 0.09
            x_range = [x_min - pad, x_max + pad]

        fig.update_layout(
            height=360,
            margin={'l': 66, 'r': 36, 't': 20, 'b': 58},
            paper_bgcolor='white',
            plot_bgcolor='white',
            title={'text': ''},
            xaxis_title=PHYS_LABEL_HTML['wavelength_nm'],
            yaxis_title=metric_label_html,
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0, 'font': {'size': 15, 'color': '#111111', 'family': 'Arial, Helvetica, sans-serif'}},
            hovermode='x unified',
            dragmode='pan',
            font={'family': 'Arial, Helvetica, sans-serif', 'size': 16, 'color': '#111111'},
        )
        fig.update_xaxes(
            range=x_range,
            showgrid=True,
            gridcolor='rgba(17,24,39,0.12)',
            zeroline=False,
            showline=True,
            linecolor='rgba(17,24,39,0.72)',
            linewidth=1,
            mirror=True,
            ticks='inside',
            ticklen=6,
            tickwidth=1,
            tickcolor='rgba(17,24,39,0.72)',
            automargin=True,
            title_font={'family': 'Arial, Helvetica, sans-serif', 'size': 17, 'color': '#111111'},
            tickfont={'family': 'Arial, Helvetica, sans-serif', 'size': 14, 'color': '#111111'},
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(17,24,39,0.12)',
            zeroline=False,
            showline=True,
            linecolor='rgba(17,24,39,0.72)',
            linewidth=1,
            mirror=True,
            ticks='inside',
            ticklen=6,
            tickwidth=1,
            tickcolor='rgba(17,24,39,0.72)',
            automargin=True,
            title_font={'family': 'Arial, Helvetica, sans-serif', 'size': 17, 'color': '#111111'},
            tickfont={'family': 'Arial, Helvetica, sans-serif', 'size': 14, 'color': '#111111'},
        )

    if sweep_df.empty or metric_key not in sweep_df.columns or not selected_mode_set:
        apply_sweep_graph_style()
        return fig

    for mode in sorted(selected_mode_set):
        mode_df = sweep_df[sweep_df['mode'].astype(int) == int(mode)].sort_values('wavelength_nm')
        if mode_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=mode_df['wavelength_nm'],
                y=mode_df[metric_key],
                mode='lines+markers',
                name=f'Mode {int(mode)}',
                line={'width': 2.4},
                marker={'size': 6.5, 'line': {'width': 0}},
                hovertemplate=PHYS_LABEL_HTML['wavelength_nm'] + ': %{x:.2f}<br>' + metric_label_html + ': %{y:.6g}<extra></extra>',
            )
        )

    wavelengths = sorted(sweep_df['wavelength_nm'].dropna().unique())
    if wavelengths:
        selected_x = float(wavelengths[clamp_sweep_index({'analysis_type': 'sweep', 'wavelength_sweep': {'wavelengths_nm': wavelengths, 'runs': [{} for _ in wavelengths]}}, selected_sweep_index)])
        fig.add_vline(x=selected_x, line_dash='dash', line_width=2.2, line_color='rgba(17,24,39,0.95)')

    apply_sweep_graph_style()
    return fig


def render_sweep_metric_selector() -> str:
    label_to_key = {label: key for key, label in SWEEP_METRIC_OPTIONS.items()}
    labels = list(label_to_key.keys())
    current_key = str(st.session_state.get('sweep_metric_key', 'n_eff'))
    current_label = SWEEP_METRIC_OPTIONS.get(current_key, SWEEP_METRIC_OPTIONS['n_eff'])
    if current_label not in labels:
        current_label = SWEEP_METRIC_OPTIONS['n_eff']
    selected_label = st.selectbox(
        'Graph quantity',
        options=labels,
        index=labels.index(current_label),
        key='sweep_metric_label',
    )
    selected_key = label_to_key[selected_label]
    st.session_state['sweep_metric_key'] = selected_key
    return selected_key


def render_sweep_modes_display_table(
    results: dict,
    *,
    table_key: str,
    display_modes_key: str = 'sweep_graph_display_modes',
) -> tuple[pd.DataFrame, list[int]]:
    display_df = build_modes_display_dataframe(results)
    modes_df = build_modes_dataframe(results)
    if display_df.empty:
        st.caption('No mode result is available.')
        return modes_df, []

    available_modes = [int(value) for value in display_df['Mode'].tolist()]
    current_modes = [int(mode) for mode in st.session_state.get(display_modes_key, []) if int(mode) in available_modes]
    if not current_modes:
        current_modes = [available_modes[0]]
        st.session_state[display_modes_key] = current_modes

    display_df = display_df.copy()
    display_df.insert(0, 'Display', display_df['Mode'].astype(int).isin(current_modes))

    edited = st.data_editor(
        display_df.style.set_properties(**{'background-color': '#ffffff', 'color': '#0f0f0f'}),
        use_container_width=True,
        hide_index=True,
        height=table_height(len(display_df), max_rows=6),
        key=table_key,
        disabled=[column for column in display_df.columns if column != 'Display'],
        column_config={
            'Display': st.column_config.CheckboxColumn('Display', width='small'),
            'Mode': st.column_config.NumberColumn('Mode', width='small', format='%d'),
            'n_eff': st.column_config.NumberColumn(PHYS_LABEL_PLAIN['n_eff'], width='small', format='%.6f'),
            'Im(n_eff)': st.column_config.NumberColumn(PHYS_LABEL_PLAIN['k_eff'], width='small', format='%.3e'),
            'Loss [dB/cm]': st.column_config.NumberColumn(PHYS_LABEL_PLAIN['loss_dB_per_cm'], width='small', format='%.3g'),
            'n_g': st.column_config.NumberColumn(PHYS_LABEL_PLAIN['group_index'], width='small', format='%.6f'),
            'TE': st.column_config.NumberColumn('TE', width='small', format='%.4f'),
            'TM': st.column_config.NumberColumn('TM', width='small', format='%.4f'),
            'P_ref': st.column_config.NumberColumn(PHYS_LABEL_PLAIN['reference_power_fraction'], width='small', format='%.4f'),
        },
    )

    try:
        checked_modes = [int(value) for value in edited.loc[edited['Display'].astype(bool), 'Mode'].tolist()]
    except Exception:
        checked_modes = []
    checked_modes = [mode for mode in checked_modes if mode in available_modes]
    st.session_state[display_modes_key] = checked_modes
    return modes_df, checked_modes

section_data = None
materials_data = None
input_shapes = None
input_region_materials = None
parse_error = None

skip_model_upload_processing = bool(st.session_state.pop('_skip_model_upload_processing_once', False))
st.session_state['_skip_model_upload_processing_this_run'] = skip_model_upload_processing
process_model_uploads_from_state(skip_processing=skip_model_upload_processing)

section_data, materials_data, input_shapes, input_region_materials, parse_error = parse_current_inputs()

region_names: list[str] = []
suggested_focus: list[str] = []
material_names: list[str] = []
if parse_error is None and input_shapes is not None and materials_data is not None:
    region_names = list(input_shapes.keys())
    material_names = list(materials_data.get('materials', {}).keys())
    suggested_focus = suggest_focus_region_names(input_shapes, input_region_materials, materials_data)
    ensure_focus_regions(region_names, suggested_focus)

focus_regions_for_config = current_focus_regions_for_config(region_names, suggested_focus)
if input_shapes is not None:
    auto_domain_for_config = auto_domain_bounds(
        input_shapes,
        focus_regions_for_config,
        wavelength_um=auto_fit_wavelength_um_from_state(),
    )
    if manual_domain_is_uninitialized():
        set_manual_domain_from_bounds(auto_domain_for_config)

empty_choice_for_config = ensure_background_material_choice(material_names)
current_domain_for_config = {
    'left': float(st.session_state.get('manual_left', 0.0)),
    'right': float(st.session_state.get('manual_right', 1.0)),
    'bottom': float(st.session_state.get('manual_bottom', 0.0)),
    'top': float(st.session_state.get('manual_top', 1.0)),
}
mesh_preset_for_config = mesh_preset_from_state()
mesh_config_for_project = build_mesh_preset_config(
    mesh_preset_for_config,
    shapes=input_shapes,
    focus_regions=focus_regions_for_config,
    domain_bounds=current_domain_for_config,
)
project_config = build_analysis_config_from_state(
    focus_regions=focus_regions_for_config,
    mesh_config=mesh_config_for_project,
    empty_choice=empty_choice_for_config,
)

st.markdown('<div class="studio-workspace-marker"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="studio-page-brand-anchor">'
    '<div class="studio-page-brand">'
    '<span class="studio-page-logo">MODESTUDIO</span>'
    '<span class="studio-page-powered">powered by FEMWELL</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)
shell_left, shell_center = st.columns([1.04, 2.36], gap='medium')

with shell_left:
    st.markdown('<div class="studio-left-scroll-marker"></div>', unsafe_allow_html=True)
    with st.container(border=False):
        st.markdown('<div class="studio-settings-pane-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-label first-panel-label">Model files</div>', unsafe_allow_html=True)
        handle_uploads()
        with st.expander('Raw JSON', expanded=False):
            render_raw_json_editor()

        st.markdown('<div class="panel-section-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Domain</div>', unsafe_allow_html=True)
        if region_names:
            focus_regions = st.multiselect('Reference regions', options=region_names, key='focus_regions')
        else:
            focus_regions = []

        auto_domain = None
        if input_shapes is not None:
            auto_domain = auto_domain_bounds(
                input_shapes,
                focus_regions,
                wavelength_um=auto_fit_wavelength_um_from_state(),
            )
            if manual_domain_is_uninitialized():
                set_manual_domain_from_bounds(auto_domain)

        if auto_domain and st.button('Fit to reference regions', use_container_width=True):
            set_manual_domain_from_bounds(auto_domain)
            st.rerun()

        domain_cols_1 = st.columns(2, gap='small')
        with domain_cols_1[0]:
            st.number_input('x min', step=0.1, format='%.4f', key='manual_left')
        with domain_cols_1[1]:
            st.number_input('x max', step=0.1, format='%.4f', key='manual_right')
        domain_cols_2 = st.columns(2, gap='small')
        with domain_cols_2[0]:
            st.number_input('y min', step=0.1, format='%.4f', key='manual_bottom')
        with domain_cols_2[1]:
            st.number_input('y max', step=0.1, format='%.4f', key='manual_top')

        material_select_options = material_names + [USER_DEFINED_EMPTY_AREA_LABEL]
        if material_select_options:
            ensure_background_material_choice(material_names)
            empty_choice = st.selectbox('Background material', options=material_select_options, key='empty_area_material_choice')
        else:
            empty_choice = USER_DEFINED_EMPTY_AREA_LABEL
            st.session_state['empty_area_material_choice'] = empty_choice

        if empty_choice == USER_DEFINED_EMPTY_AREA_LABEL:
            empty_cols = st.columns(2)
            with empty_cols[0]:
                st.number_input('n', min_value=1.0, step=0.01, format='%.6f', key='empty_area_n_real')
            with empty_cols[1]:
                st.number_input('k', min_value=0.0, step=0.0001, format='%.6f', key='empty_area_k')

        st.markdown('<div class="panel-section-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Solver</div>', unsafe_allow_html=True)
        wavelength_mode = st.segmented_control(
            'Wavelength mode',
            options=['Single', 'Sweep'],
            default='Single',
            key='wavelength_mode',
        )
        if wavelength_mode is None:
            wavelength_mode = 'Single'

        if wavelength_mode == 'Sweep':
            sweep_cols = st.columns(3)
            with sweep_cols[0]:
                st.number_input('Start [nm]', min_value=100.0, value=1500.0, step=10.0, format='%.1f', key='sweep_start_nm')
            with sweep_cols[1]:
                st.number_input('Stop [nm]', min_value=100.0, value=1600.0, step=10.0, format='%.1f', key='sweep_stop_nm')
            with sweep_cols[2]:
                st.number_input('Points', min_value=2, max_value=31, value=5, step=1, key='sweep_points')
            if float(st.session_state.get('sweep_stop_nm', 1600.0)) < float(st.session_state.get('sweep_start_nm', 1500.0)):
                st.caption('Stop is smaller than Start. The sweep will run in the entered order.')
        else:
            st.number_input('Wavelength [nm]', min_value=100.0, value=1550.0, step=10.0, format='%.1f', key='single_wavelength_nm')

        solver_cols = st.columns(2)
        with solver_cols[0]:
            st.number_input('Modes', min_value=1, max_value=20, value=2, step=1, key='num_modes')
        with solver_cols[1]:
            st.selectbox('Order', options=[1, 2], index=0, key='order')

        st.markdown('<div class="panel-section-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Mesh</div>', unsafe_allow_html=True)
        mesh_preset = st.radio('Preset', options=['Coarse', 'Normal', 'Fine', 'Ultra'], index=1, horizontal=True, label_visibility='collapsed', key='mesh_preset')
        current_domain = {
            'left': float(st.session_state.get('manual_left', 0.0)),
            'right': float(st.session_state.get('manual_right', 1.0)),
            'bottom': float(st.session_state.get('manual_bottom', 0.0)),
            'top': float(st.session_state.get('manual_top', 1.0)),
        }
        mesh_config = build_mesh_preset_config(
            mesh_preset,
            shapes=input_shapes,
            focus_regions=focus_regions,
            domain_bounds=current_domain,
        )
        st.caption(
            f"Reference regions: {mesh_config['refined_resolution']:.3g} um · "
            f"Other regions: {mesh_config['surrounding_resolution']:.3g} um · "
            f"Max: {mesh_config['default_resolution_max']:.3g} um"
        )

        current_export_config = build_analysis_config_from_state(
            focus_regions=focus_regions,
            mesh_config=mesh_config,
            empty_choice=str(empty_choice),
        )

        st.markdown('<div class="panel-run-space"></div>', unsafe_allow_html=True)
        run_clicked = st.button('Run mode analysis', type='primary', use_container_width=True)
        if parse_error is None and isinstance(section_data, dict) and isinstance(materials_data, dict):
            bundle_name = f'{_safe_project_filename_part(st.session_state.get("project_name"))}_python_export.zip'
            bundle_data = build_python_script_bundle_export(
                section_data,
                materials_data,
                current_export_config,
                project_name=st.session_state.get('project_name'),
            )
            st.download_button(
                'Export Python script',
                data=bundle_data,
                file_name=bundle_name,
                mime='application/zip',
                use_container_width=True,
            )
        else:
            st.download_button(
                'Export Python script',
                data=b'',
                file_name='modestudio_python_export.zip',
                mime='application/zip',
                use_container_width=True,
                disabled=True,
            )
        render_archive_dock(current_export_config)

config = current_export_config

analysis_shapes = None
analysis_region_materials = None
simulation_domain = {}
preview_error = None
if parse_error is None and input_shapes is not None:
    try:
        analysis_shapes, analysis_region_materials, simulation_domain = apply_analysis_window(input_shapes, input_region_materials, config)
    except Exception as exc:
        preview_error = str(exc)

if run_clicked:
    if parse_error is not None or preview_error is not None:
        st.session_state['analysis_output'] = None
        clear_analysis_output_cache()
        st.session_state['analysis_error'] = parse_error or preview_error
        st.rerun()
    else:
        with st.spinner('Running femwell...'):
            try:
                st.session_state['analysis_output'] = run_solver_subprocess(section_data, materials_data, config)
            except Exception as exc:
                st.session_state['analysis_output'] = None
                clear_analysis_output_cache()
                st.session_state['analysis_error'] = str(exc)
                st.rerun()
            else:
                _cache_analysis_output_payload(st.session_state.get('analysis_output'))
                st.session_state['analysis_error'] = None
                st.session_state['active_view'] = MODE_FIELD_VIEW_LABEL
                st.session_state['selected_sweep_index'] = 0
                st.session_state['selected_result_mode'] = 0
                st.session_state['selected_field_quantity'] = 'intensity'
                st.session_state['selected_field_label'] = FIELD_QUANTITY_LABELS['intensity']
                st.session_state['selected_field_scale'] = 'linear'
                st.rerun()

analysis_output = st.session_state.get('analysis_output')
root_results = analysis_output['results'] if analysis_output else None
selected_sweep_index = clamp_sweep_index(root_results) if root_results else 0
results = selected_run_results(analysis_output, selected_sweep_index) if analysis_output else None
field_maps = selected_mode_field_maps(analysis_output, selected_sweep_index) if analysis_output else []
sweep_df = build_sweep_dataframe(root_results) if root_results else pd.DataFrame()
run_revision = analysis_run_revision(results) if results else 'no-result'

if 'active_view' not in st.session_state:
    st.session_state['active_view'] = CROSS_SECTION_VIEW_LABEL
if not field_maps and st.session_state['active_view'] == MODE_FIELD_VIEW_LABEL:
    st.session_state['active_view'] = CROSS_SECTION_VIEW_LABEL
if not is_sweep_results(root_results) and st.session_state.get('active_view') == WAVELENGTH_SWEEP_VIEW_LABEL:
    st.session_state['active_view'] = MODE_FIELD_VIEW_LABEL if field_maps else CROSS_SECTION_VIEW_LABEL

with shell_center:
    st.markdown('<div class="studio-main-scroll-marker"></div>', unsafe_allow_html=True)
    with st.container(border=False):
        st.markdown('<div class="studio-results-pane-marker"></div>', unsafe_allow_html=True)
        available_views = [CROSS_SECTION_VIEW_LABEL] + ([MODE_FIELD_VIEW_LABEL] if field_maps else []) + ([WAVELENGTH_SWEEP_VIEW_LABEL] if is_sweep_results(root_results) else [])
        if st.session_state.get('active_view') not in available_views:
            st.session_state['active_view'] = CROSS_SECTION_VIEW_LABEL
        active_view = st.segmented_control(
            'View',
            options=available_views,
            key='active_view',
            label_visibility='collapsed',
        )
        if active_view is None:
            active_view = available_views[0]
            st.session_state['active_view'] = active_view

        if parse_error:
            st.error(parse_error)
        elif preview_error:
            st.error(preview_error)
        elif active_view == MODE_FIELD_VIEW_LABEL and field_maps:
            mode_options = list(range(len(field_maps)))
            table_key = f'modes_table_{run_revision}'
            fallback_mode = int(st.session_state.get('selected_result_mode', mode_options[0]))
            selected_mode = fallback_mode if fallback_mode in mode_options else mode_options[0]
            st.session_state['selected_result_mode'] = selected_mode

            selected_pack = mode_field_pack(field_maps[selected_mode])
            fields = selected_pack.get('fields', {})
            if not all(key in fields for key in FIELD_QUANTITY_ORDER):
                st.error('Field map data is incomplete. Re-run the mode analysis with the current app version.')
                st.stop()

            current_quantity = st.session_state.get('selected_field_quantity', 'intensity')
            current_label = FIELD_QUANTITY_LABELS.get(current_quantity, FIELD_QUANTITY_LABELS['intensity'])
            if st.session_state.get('selected_field_label') not in FIELD_QUANTITY_OPTIONS:
                st.session_state['selected_field_label'] = current_label if current_label in FIELD_QUANTITY_OPTIONS else FIELD_QUANTITY_LABELS['intensity']

            control_cols = st.columns([1.2, 0.55])
            with control_cols[0]:
                selected_label = st.selectbox(
                    'Field quantity',
                    options=FIELD_QUANTITY_OPTIONS,
                    key='selected_field_label',
                )
            selected_quantity = FIELD_LABEL_TO_KEY[selected_label]
            st.session_state['selected_field_quantity'] = selected_quantity
            with control_cols[1]:
                field_scale = st.selectbox('Scale', options=['linear', 'log'], key='selected_field_scale')

            if is_sweep_results(root_results):
                render_wavelength_selector(root_results)

            selected_map = fields[selected_quantity]
            components.html(
                canvas_viewer_html(
                    view='field',
                    shapes=analysis_shapes if analysis_shapes is not None else None,
                    region_materials=analysis_region_materials if analysis_region_materials is not None else None,
                    domain=selected_map.get('bounds'),
                    field_map=selected_map,
                    field_scale=field_scale,
                    height=500,
                ),
                height=516,
                scrolling=False,
            )
            st.download_button(
                'Download field data.json',
                data=field_data_json_bytes(
                    field_map=selected_map,
                    field_key=selected_quantity,
                    mode_index=selected_mode,
                    results=results,
                    section_data=section_data,
                    materials_data=materials_data,
                    config=config,
                ),
                file_name=f'mode_{int(selected_mode)}_{selected_quantity}_field_data.json',
                mime='application/json',
                use_container_width=True,
            )
            if results:
                st.markdown('<div class="table-label">Modes</div>', unsafe_allow_html=True)
                modes_df, clicked_mode = render_modes_selection_table(results, table_key=table_key, selected_mode=selected_mode)
                if clicked_mode is not None and clicked_mode in mode_options and clicked_mode != selected_mode:
                    st.session_state['selected_result_mode'] = int(clicked_mode)
                    st.rerun()
                st.caption('Check a row to display that mode.')
                st.markdown('<div class="download-row-spacer"></div>', unsafe_allow_html=True)
                d1, d2 = st.columns([1, 1])
                d1.download_button('Download modes.csv', data=dataframe_csv_bytes(modes_df), file_name='modes.csv', mime='text/csv', use_container_width=True)
                d2.download_button(
                    'Download modes metadata.json',
                    data=result_metadata_json_bytes(
                        csv_file='modes.csv',
                        results=results,
                        section_data=section_data,
                        materials_data=materials_data,
                        config=config,
                        dataframe=modes_df,
                    ),
                    file_name='modes_metadata.json',
                    mime='application/json',
                    use_container_width=True,
                )
        elif active_view == WAVELENGTH_SWEEP_VIEW_LABEL and is_sweep_results(root_results):
            selected_metric_key = render_sweep_metric_selector()

            sweep_index = render_wavelength_selector(root_results)
            if sweep_index != selected_sweep_index:
                st.rerun()

            current_run = selected_run_results(analysis_output, sweep_index)
            graph_slot = st.container()
            table_slot = st.container()

            display_modes: list[int] = []
            modes_df = pd.DataFrame()
            with table_slot:
                if current_run:
                    st.markdown('<div class="table-label">Modes at selected wavelength</div>', unsafe_allow_html=True)
                    modes_df, display_modes = render_sweep_modes_display_table(
                        current_run,
                        table_key=f'sweep_modes_table_{run_revision}_{sweep_index}',
                    )
                    st.caption('Checked Display rows are drawn in the sweep graph.')

            with graph_slot:
                if display_modes:
                    st.plotly_chart(
                        sweep_metric_figure(
                            sweep_df,
                            metric_key=selected_metric_key,
                            selected_sweep_index=sweep_index,
                            display_modes=display_modes,
                        ),
                        use_container_width=True,
                        config={'displaylogo': False, 'scrollZoom': True},
                    )
                else:
                    st.info('Check one or more Display rows to draw the sweep graph.')

            if current_run:
                st.markdown('<div class="download-row-spacer"></div>', unsafe_allow_html=True)
                d1, d2 = st.columns([1, 1])
                d1.download_button('Download sweep.csv', data=dataframe_csv_bytes(sweep_df), file_name='wavelength_sweep.csv', mime='text/csv', use_container_width=True)
                d2.download_button(
                    'Download sweep metadata.json',
                    data=result_metadata_json_bytes(
                        csv_file='wavelength_sweep.csv',
                        results=root_results,
                        section_data=section_data,
                        materials_data=materials_data,
                        config=config,
                        dataframe=sweep_df,
                    ),
                    file_name='wavelength_sweep_metadata.json',
                    mime='application/json',
                    use_container_width=True,
                )
        else:
            components.html(
                canvas_viewer_html(
                    view='cross_section',
                    shapes=input_shapes,
                    region_materials=input_region_materials,
                    domain=simulation_domain,
                    field_map=None,
                    height=500,
                ),
                height=516,
                scrolling=False,
            )
            if parse_error is None and materials_data is not None:
                materials_df = build_materials_dataframe(
                    materials_data,
                    str(empty_choice),
                    float(st.session_state['empty_area_n_real']),
                    float(st.session_state['empty_area_k']),
                )
                if input_shapes is not None and input_region_materials is not None:
                    regions_df = build_regions_dataframe(input_shapes, input_region_materials)
                    info_col_1, info_col_2 = st.columns(2, gap='medium')
                    with info_col_1:
                        st.markdown('<div class="table-label">Materials</div>', unsafe_allow_html=True)
                        st.dataframe(
                            materials_df.style.set_properties(**{'background-color': '#ffffff', 'color': '#0f0f0f'}),
                            use_container_width=True,
                            hide_index=True,
                            height=table_height(len(materials_df), max_rows=4),
                        )
                    with info_col_2:
                        st.markdown('<div class="table-label">Regions</div>', unsafe_allow_html=True)
                        st.dataframe(
                            regions_df.style.set_properties(**{'background-color': '#ffffff', 'color': '#0f0f0f'}),
                            use_container_width=True,
                            hide_index=True,
                            height=table_height(len(regions_df), max_rows=4),
                        )
                else:
                    st.markdown('<div class="table-label">Materials</div>', unsafe_allow_html=True)
                    st.dataframe(
                        materials_df.style.set_properties(**{'background-color': '#ffffff', 'color': '#0f0f0f'}),
                        use_container_width=True,
                        hide_index=True,
                        height=table_height(len(materials_df), max_rows=4),
                    )


    if st.session_state.get('analysis_error'):
        st.error(st.session_state['analysis_error'])