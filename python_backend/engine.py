from __future__ import annotations

import json
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

DEFAULT_AUTO_DOMAIN_PADDING = {
    'x': 1.50,
    'y': 1.50,
}
DEFAULT_EMPTY_AREA_MATERIAL = 'empty_area'


@dataclass
class AnalysisArtifacts:
    results: dict[str, Any]
    mode_field_maps: list[dict[str, Any]]
    sweep_field_maps: list[dict[str, Any]] | None = None


class AnalysisImportError(RuntimeError):
    pass


class InputValidationError(ValueError):
    pass


def loads_json(text: str, *, label: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputValidationError(f'{label} JSON could not be parsed: {exc}') from exc
    if not isinstance(data, dict):
        raise InputValidationError(f'{label} JSON must be an object at the top level.')
    return data


def validate_section_data(section_data: dict[str, Any]) -> None:
    unit = section_data.get('unit', 'um')
    if unit != 'um':
        raise InputValidationError(f"Section JSON currently expects unit='um', got {unit!r}.")
    regions = section_data.get('regions')
    if not isinstance(regions, list) or not regions:
        raise InputValidationError("Section JSON must contain a non-empty 'regions' list.")


def validate_materials_data(materials_data: dict[str, Any]) -> None:
    materials = materials_data.get('materials')
    if not isinstance(materials, dict) or not materials:
        raise InputValidationError("Materials JSON must contain a non-empty 'materials' object.")
    for name, entry in materials.items():
        if not isinstance(entry, dict):
            raise InputValidationError(f'Material {name!r} must be an object.')
        n_value = entry.get('n')
        if not isinstance(n_value, dict) or 'real' not in n_value or 'imag' not in n_value:
            raise InputValidationError(f'Material {name!r} must contain n.real and n.imag.')


def polygon_from_dict(poly_data: dict[str, Any]) -> Polygon:
    if 'hull' not in poly_data:
        raise InputValidationError('Every polygon entry must contain a hull.')
    polygon = Polygon(poly_data['hull'], holes=poly_data.get('holes', []))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise InputValidationError('Encountered an empty polygon after cleanup.')
    return polygon


def _region_mesh_order(region: dict[str, Any]) -> float:
    try:
        return float(region.get('mesh_order', 2))
    except (TypeError, ValueError) as exc:
        name = str(region.get('name', '<unnamed>'))
        raise InputValidationError(f'Region {name!r} has an invalid mesh_order.') from exc


def _clean_resolved_shape(shape: object) -> object:
    if getattr(shape, 'is_empty', True):
        return shape
    if not getattr(shape, 'is_valid', True):
        shape = shape.buffer(0)
    return shape


def build_shapes(section_data: dict[str, Any]) -> tuple[OrderedDict[str, object], dict[str, str]]:
    validate_section_data(section_data)
    region_entries: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for index, region in enumerate(section_data['regions']):
        name = str(region['name'])
        if name in seen_names:
            raise InputValidationError(f'Duplicate region name: {name!r}')
        seen_names.add(name)

        material = str(region['material']).strip().lower()
        polygons = [polygon_from_dict(poly) for poly in region.get('polygons', [])]
        if not polygons:
            continue

        shape = unary_union(polygons)
        shape = _clean_resolved_shape(shape)
        if shape.is_empty or float(shape.area) <= 0.0:
            continue

        region_entries.append(
            {
                'name': name,
                'material': material,
                'mesh_order': _region_mesh_order(region),
                'index': index,
                'shape': shape,
            }
        )

    if not region_entries:
        raise InputValidationError('No usable polygons were found in the section JSON.')

    # Resolve overlapping regions before meshing. Lower mesh_order has higher
    # priority. For the same mesh_order, later entries in Section JSON have
    # higher priority, matching the usual "draw/define larger background first,
    # then put smaller objects on top" workflow.
    resolved_by_name: dict[str, object] = {}
    covered = None
    for entry in sorted(region_entries, key=lambda item: (item['mesh_order'], -item['index'])):
        shape = entry['shape']
        if covered is not None and not covered.is_empty:
            shape = shape.difference(covered)
        shape = _clean_resolved_shape(shape)
        if shape.is_empty or float(shape.area) <= 1e-12:
            continue
        resolved_by_name[entry['name']] = shape
        covered = shape if covered is None else unary_union([covered, shape])

    shapes: OrderedDict[str, object] = OrderedDict()
    materials: dict[str, str] = {}
    for entry in region_entries:
        name = entry['name']
        if name not in resolved_by_name:
            continue
        shapes[name] = resolved_by_name[name]
        materials[name] = entry['material']

    if not shapes:
        raise InputValidationError('No usable polygons were found after resolving overlapping regions.')
    return shapes, materials


def bounds_to_dict(bounds: tuple[float, float, float, float]) -> dict[str, float]:
    left, bottom, right, top = bounds
    return {
        'left': float(left),
        'bottom': float(bottom),
        'right': float(right),
        'top': float(top),
    }


def get_geometry_bounds(shapes: OrderedDict[str, object]) -> dict[str, float]:
    return bounds_to_dict(unary_union(list(shapes.values())).bounds)


def build_material_lookup(materials_data: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, complex]:
    validate_materials_data(materials_data)
    lookup: dict[str, complex] = {}
    for name, entry in materials_data['materials'].items():
        n_value = entry['n']
        lookup[str(name).strip().lower()] = complex(float(n_value['real']), float(n_value['imag']))

    if config is not None and str(config.get('empty_area_material_mode', '')).lower() == 'user_defined':
        empty_material = str(config.get('background_material') or DEFAULT_EMPTY_AREA_MATERIAL).strip().lower()
        k_value = config.get('empty_area_k', config.get('empty_area_n_imag', 0.0))
        lookup[empty_material] = complex(float(config['empty_area_n_real']), float(k_value))

    return lookup


def suggest_focus_region_names(
    shapes: OrderedDict[str, object],
    region_materials: dict[str, str],
    materials_data: dict[str, Any],
) -> list[str]:
    if not shapes:
        return []

    try:
        material_lookup = build_material_lookup(materials_data)
    except Exception:
        material_lookup = {}

    areas = {name: float(shape.area) for name, shape in shapes.items()}
    max_area = max(areas.values()) if areas else 0.0
    global_bounds = get_geometry_bounds(shapes)
    global_bottom = global_bounds['bottom']
    n_real_by_region: dict[str, float] = {}
    for name, material in region_materials.items():
        n_real_by_region[name] = float(np.real(material_lookup.get(material, complex(1.0, 0.0))))

    if n_real_by_region:
        n_max = max(n_real_by_region.values())
    else:
        n_max = 0.0

    candidates: list[tuple[float, str]] = []
    for name, shape in shapes.items():
        area = areas[name]
        _, bottom, _, top = shape.bounds
        is_high_index = n_real_by_region.get(name, 0.0) >= n_max - 0.05
        is_not_large_layer = max_area <= 0.0 or area <= 0.20 * max_area
        is_not_bottom_filling_layer = not (abs(bottom - global_bottom) < 1e-9 and area > 0.10 * max_area)
        if is_high_index and is_not_large_layer and is_not_bottom_filling_layer:
            vertical_score = float(top)
            area_score = -area
            candidates.append((vertical_score + area_score * 1e-3, name))

    if candidates:
        candidates.sort(reverse=True)
        selected = [name for _, name in candidates]
        return selected[: min(6, len(selected))]

    sorted_by_area = sorted(areas, key=areas.get)
    return sorted_by_area[: min(2, len(sorted_by_area))]


def focus_shape(shapes: OrderedDict[str, object], focus_regions: list[str] | tuple[str, ...] | None) -> object:
    selected = [shapes[name] for name in (focus_regions or []) if name in shapes]
    if selected:
        return unary_union(selected)
    return unary_union(list(shapes.values()))


def auto_domain_bounds(
    shapes: OrderedDict[str, object],
    focus_regions: list[str] | tuple[str, ...] | None,
    *,
    wavelength_um: float | None = None,
) -> dict[str, float]:
    anchor = focus_shape(shapes, focus_regions)
    left, bottom, right, top = anchor.bounds
    width = max(float(right - left), 1e-9)
    height = max(float(top - bottom), 1e-9)

    # Auto fit is meant to be a safe first guess, not a fully manual domain
    # replacement.  Keep the selected reference bbox itself, but do not let the
    # margin scale with the bbox long side.  Long slab/rib regions otherwise
    # create unnecessarily large windows.  The symmetric margin is based on the
    # short-side feature size and a moderate wavelength allowance.  In SOI
    # structures, an overly large wavelength-based margin can include too much
    # substrate and make substrate-guided modes dominate, so the wavelength term
    # is intentionally 1.0 * lambda, not a multi-wavelength padding.
    feature_size = max(min(width, height), 1e-9)
    try:
        wavelength_value = float(wavelength_um) if wavelength_um is not None else 1.55
    except (TypeError, ValueError):
        wavelength_value = 1.55
    if not np.isfinite(wavelength_value) or wavelength_value <= 0.0:
        wavelength_value = 1.55

    margin_candidate = max(6.0 * feature_size, 1.0 * wavelength_value, 0.5)
    margin_cap = max(2.0, 1.25 * wavelength_value)
    margin = min(margin_candidate, margin_cap)

    # Keep manual-domain inputs readable.  Round the final bounds outward to a
    # 50 nm grid so the reference bbox is still fully contained while avoiding
    # values like 0.337499999999.
    step = 0.05
    rounded_left = np.floor((left - margin) / step) * step
    rounded_right = np.ceil((right + margin) / step) * step
    rounded_bottom = np.floor((bottom - margin) / step) * step
    rounded_top = np.ceil((top + margin) / step) * step

    return {
        'left': float(round(rounded_left, 6)),
        'right': float(round(rounded_right, 6)),
        'bottom': float(round(rounded_bottom, 6)),
        'top': float(round(rounded_top, 6)),
    }


def _unique_region_name(existing: OrderedDict[str, object], preferred: str) -> str:
    base = (preferred or DEFAULT_EMPTY_AREA_MATERIAL).strip() or DEFAULT_EMPTY_AREA_MATERIAL
    if base not in existing:
        return base
    index = 1
    while f'{base}_{index}' in existing:
        index += 1
    return f'{base}_{index}'


def clip_to_window(
    shapes: OrderedDict[str, object],
    materials: dict[str, str],
    *,
    window,
    background_name: str,
    background_material: str,
) -> tuple[OrderedDict[str, object], dict[str, str], dict[str, float]]:
    clipped_shapes: OrderedDict[str, object] = OrderedDict()
    clipped_materials: dict[str, str] = {}
    for name, shape in shapes.items():
        clipped = shape.intersection(window)
        if clipped.is_empty or float(clipped.area) <= 1e-12:
            continue
        clipped_shapes[name] = clipped
        clipped_materials[name] = materials[name]

    if not clipped_shapes:
        raise InputValidationError('The analysis window does not overlap any section polygons.')

    covered = unary_union(list(clipped_shapes.values()))
    empty_area = window.difference(covered)
    if not empty_area.is_empty and float(empty_area.area) > 1e-12:
        name = _unique_region_name(clipped_shapes, background_name)
        clipped_shapes[name] = empty_area
        clipped_materials[name] = background_material.strip().lower()

    return clipped_shapes, clipped_materials, bounds_to_dict(window.bounds)


def clip_to_auto_window(
    shapes: OrderedDict[str, object],
    materials: dict[str, str],
    *,
    focus_regions: list[str] | tuple[str, ...] | None,
    background_name: str,
    background_material: str,
    wavelength_um: float | None = None,
) -> tuple[OrderedDict[str, object], dict[str, str], dict[str, float]]:
    bounds = auto_domain_bounds(shapes, focus_regions, wavelength_um=wavelength_um)
    return clip_to_window(
        shapes,
        materials,
        window=box(bounds['left'], bounds['bottom'], bounds['right'], bounds['top']),
        background_name=background_name,
        background_material=background_material,
    )


def clip_to_manual_window(
    shapes: OrderedDict[str, object],
    materials: dict[str, str],
    *,
    left: float,
    bottom: float,
    right: float,
    top: float,
    background_name: str,
    background_material: str,
) -> tuple[OrderedDict[str, object], dict[str, str], dict[str, float]]:
    if not left < right:
        raise InputValidationError('Manual analysis window requires left < right.')
    if not bottom < top:
        raise InputValidationError('Manual analysis window requires bottom < top.')
    return clip_to_window(
        shapes,
        materials,
        window=box(left, bottom, right, top),
        background_name=background_name,
        background_material=background_material,
    )


def apply_analysis_window(
    shapes: OrderedDict[str, object],
    materials: dict[str, str],
    config: dict[str, Any],
) -> tuple[OrderedDict[str, object], dict[str, str], dict[str, float]]:
    window_mode = str(config.get('window_mode') or 'auto').lower()
    background_name = str(config.get('background_name', DEFAULT_EMPTY_AREA_MATERIAL))
    background_material = str(config.get('background_material', DEFAULT_EMPTY_AREA_MATERIAL))
    focus_regions = list(config.get('focus_regions') or [])

    if window_mode == 'auto':
        return clip_to_auto_window(
            shapes,
            materials,
            focus_regions=focus_regions,
            background_name=background_name,
            background_material=background_material,
            wavelength_um=float(config.get('wavelength_um', 1.55)),
        )

    if window_mode == 'manual':
        return clip_to_manual_window(
            shapes,
            materials,
            left=float(config['manual_left']),
            bottom=float(config['manual_bottom']),
            right=float(config['manual_right']),
            top=float(config['manual_top']),
            background_name=background_name,
            background_material=background_material,
        )

    if window_mode in {'full', 'section'}:
        return shapes, materials, {}

    raise InputValidationError(f'Unknown analysis window mode: {window_mode!r}.')


def preview_analysis_geometry(
    section_data: dict[str, Any],
    config: dict[str, Any],
) -> tuple[OrderedDict[str, object], dict[str, str], dict[str, float]]:
    shapes, materials = build_shapes(section_data)
    return apply_analysis_window(shapes, materials, config)


def build_resolutions(region_names: list[str], config: dict[str, Any]) -> dict[str, dict[str, float]]:
    if 'refined_resolution' in config:
        refined_regions = set(str(name) for name in config.get('refined_regions', []))
        if not refined_regions:
            refined_regions = set(str(name) for name in config.get('focus_regions', []))

        refined_resolution = float(config['refined_resolution'])
        refined_distance = float(config.get('refined_distance', refined_resolution))
        surrounding_resolution = float(config['surrounding_resolution'])
        surrounding_distance = float(config.get('surrounding_distance', surrounding_resolution))
        size_max = float(config['default_resolution_max'])

        resolutions: dict[str, dict[str, float]] = {}
        for name in region_names:
            if name in refined_regions:
                resolutions[name] = {'resolution': refined_resolution, 'distance': refined_distance}
            else:
                resolutions[name] = {
                    'resolution': surrounding_resolution,
                    'distance': surrounding_distance,
                    'SizeMax': size_max,
                }
        return resolutions

    if 'mesh_resolution' in config:
        resolution = float(config['mesh_resolution'])
        distance = float(config.get('mesh_distance', resolution))
        size_max = float(config['default_resolution_max'])
        return {
            name: {'resolution': resolution, 'distance': distance, 'SizeMax': size_max}
            for name in region_names
        }

    resolutions: dict[str, dict[str, float]] = {}
    for name in region_names:
        lname = name.lower()
        if lname == 'core':
            resolutions[name] = {'resolution': float(config['core_resolution']), 'distance': float(config['core_distance'])}
        elif lname == 'slab':
            resolutions[name] = {'resolution': float(config['slab_resolution']), 'distance': float(config['slab_distance'])}
        else:
            resolutions[name] = {
                'resolution': float(config['clad_resolution']),
                'distance': float(config['clad_distance']),
                'SizeMax': float(config['default_resolution_max']),
            }
    return resolutions



def _finite_float_list(values: np.ndarray) -> list[float | None]:
    output: list[float | None] = []
    for value in np.asarray(values, dtype=float).reshape(-1):
        output.append(None if not np.isfinite(value) else float(value))
    return output


def _coordinate_key(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), 12), round(float(y), 12))


def _values_at_mesh_vertices(basis, values) -> np.ndarray | None:
    mesh = basis.mesh
    points = np.asarray(mesh.p, dtype=float)
    doflocs = np.asarray(getattr(basis, 'doflocs', points), dtype=float)
    values_array = np.real(np.asarray(values)).reshape(-1)

    vertex_count = points.shape[1]
    if values_array.size == vertex_count:
        return values_array.astype(float)

    if doflocs.ndim == 2 and doflocs.shape[0] >= 2 and values_array.size == doflocs.shape[1]:
        by_location: dict[tuple[float, float], float] = {}
        for index in range(doflocs.shape[1]):
            by_location[_coordinate_key(doflocs[0, index], doflocs[1, index])] = float(values_array[index])

        vertex_values = np.empty(vertex_count, dtype=float)
        missing: list[int] = []
        for index in range(vertex_count):
            key = _coordinate_key(points[0, index], points[1, index])
            if key in by_location:
                vertex_values[index] = by_location[key]
            else:
                missing.append(index)

        if not missing:
            return vertex_values

        # Last-resort fallback for bases whose dof locations are numerically close,
        # but not bit-identical, to mesh vertices.  This path is rarely used and is
        # intentionally kept local to result serialization.
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return None
        tree = cKDTree(doflocs[:2].T)
        _, nearest = tree.query(points[:2, missing].T, k=1)
        for vertex_index, dof_index in zip(missing, np.asarray(nearest).reshape(-1)):
            vertex_values[vertex_index] = float(values_array[int(dof_index)])
        return vertex_values

    return None



def scalar_field_to_triangular_mesh(
    basis,
    values,
    *,
    quantity: str,
    z_label: str,
    normalize: bool = False,
    bounds: dict[str, float] | None = None,
) -> dict[str, Any]:
    mesh = basis.mesh
    points = np.asarray(mesh.p, dtype=float)
    triangles = np.asarray(mesh.t, dtype=int)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] == 0:
        raise AnalysisImportError(f'Could not prepare {quantity}: mesh point coordinates are unavailable.')
    if triangles.ndim != 2 or triangles.shape[0] < 3 or triangles.shape[1] == 0:
        raise AnalysisImportError(f'Could not prepare {quantity}: triangular mesh connectivity is unavailable.')

    triangles = triangles[:3, :]
    values_array = np.real(np.asarray(values)).reshape(-1).astype(float)
    element_dofs = np.asarray(getattr(basis, 'element_dofs', np.empty((0, 0))), dtype=int)
    doflocs = np.asarray(getattr(basis, 'doflocs', np.empty((0, 0))), dtype=float)

    if (
        element_dofs.ndim == 2
        and element_dofs.shape[0] >= 3
        and element_dofs.shape[1] == triangles.shape[1]
        and doflocs.ndim == 2
        and doflocs.shape[0] >= 2
        and values_array.size > int(np.max(element_dofs[:3, :]))
    ):
        # Femwell's intensity is typically represented on ElementDG(ElementTriP1()).
        # Preserve the element-local degrees of freedom instead of merging equal
        # coordinates across neighboring elements.  This keeps the Femwell-style
        # linear interpolation inside each triangle while allowing discontinuities
        # across triangle boundaries.
        local_dofs = element_dofs[:3, :].T
        x = doflocs[0, local_dofs].reshape(-1)
        y = doflocs[1, local_dofs].reshape(-1)
        z = values_array[local_dofs].reshape(-1)
        i = np.arange(0, 3 * local_dofs.shape[0], 3, dtype=int)
        j = i + 1
        k = i + 2
        intensity_mode = 'vertex'
        render_basis = 'element_local_p1'
    else:
        vertex_values = _values_at_mesh_vertices(basis, values_array)

        if vertex_values is None and values_array.size == triangles.shape[1]:
            # Elementwise values: duplicate vertices per element so Plotly displays
            # a flat-colored triangle mesh without interpolating across elements.
            tri = triangles.T
            x = points[0, tri].reshape(-1)
            y = points[1, tri].reshape(-1)
            z = np.repeat(values_array, 3)
            i = np.arange(0, 3 * tri.shape[0], 3, dtype=int)
            j = i + 1
            k = i + 2
            intensity_mode = 'vertex'
            render_basis = 'element_p0'
        elif vertex_values is not None:
            x = points[0]
            y = points[1]
            z = vertex_values
            i = triangles[0]
            j = triangles[1]
            k = triangles[2]
            intensity_mode = 'vertex'
            render_basis = 'mesh_vertex'
        else:
            raise AnalysisImportError(
                f'Could not prepare {quantity}: field values do not match mesh vertices or elements '
                f'({values_array.size} values, {points.shape[1]} vertices, {triangles.shape[1]} elements).'
            )

    if normalize:
        finite = z[np.isfinite(z)]
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 0.0
        if vmax > 0.0:
            z = z / vmax
            z = np.clip(z, 0.0, 1.0)

    return {
        'type': 'triangular_mesh',
        'quantity': quantity,
        'z_label': z_label,
        'x': [float(value) for value in x],
        'y': [float(value) for value in y],
        'i': [int(value) for value in np.asarray(i).reshape(-1)],
        'j': [int(value) for value in np.asarray(j).reshape(-1)],
        'k': [int(value) for value in np.asarray(k).reshape(-1)],
        'value': _finite_float_list(z),
        'intensity_mode': intensity_mode,
        'render_basis': render_basis,
        'mesh_vertices': int(points.shape[1]),
        'mesh_triangles': int(triangles.shape[1]),
        'bounds': bounds if bounds is not None else bounds_to_dict((float(np.min(points[0])), float(np.min(points[1])), float(np.max(points[0])), float(np.max(points[1])))),
    }


def mode_intensity_map(mode, bounds: dict[str, float] | None = None) -> dict[str, Any]:
    intensity_basis, intensity = mode.calculate_intensity()
    field_map = scalar_field_to_triangular_mesh(
        intensity_basis,
        intensity,
        quantity='Mode intensity',
        z_label='normalized |E|^2',
        normalize=True,
        bounds=bounds,
    )
    field_map['field_key'] = 'intensity'
    field_map['field_label'] = '|E|^2'
    field_map['value_kind'] = 'nonnegative'
    return field_map


def _component_part(values, part: str):
    if part == 'real':
        return np.real(values)
    if part == 'imag':
        return np.imag(values)
    if part == 'abs':
        return np.abs(values)
    raise ValueError(f'Unsupported field part: {part!r}')


def mode_component_map(
    mode,
    *,
    field: str,
    component: str,
    part: str = 'abs',
    bounds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Serialize one E/H field component on a plot-ready triangular basis.

    This follows the same basic projection path as Femwell's plot_component:
    transverse components are projected onto a discontinuous vector P1 basis,
    while the longitudinal component is projected onto a discontinuous scalar
    P1 basis.  The result can be drawn by the Canvas/WebGL viewer without
    falling back to matplotlib or PNGs.
    """
    try:
        from skfem import ElementDG, ElementTriP1, ElementVector
    except Exception as exc:  # pragma: no cover - only hit when dependencies are missing
        raise AnalysisImportError('Could not import skfem elements needed for field-component export.') from exc

    field_name = field.upper()
    component_name = component.lower()
    if field_name == 'E':
        raw_field = mode.E
    elif field_name == 'H':
        raw_field = mode.H
    else:
        raise ValueError("field must be 'E' or 'H'.")

    (field_t, field_t_basis), (field_z, field_z_basis) = mode.basis.split(raw_field)
    label_part = '|' if part == 'abs' else f'{part}('

    if component_name in ('x', 'y'):
        plot_basis = field_t_basis.with_element(ElementVector(ElementDG(ElementTriP1())))
        projected_xy = plot_basis.project(_component_part(field_t_basis.interpolate(field_t), part))
        (field_x, field_x_basis), (field_y, field_y_basis) = plot_basis.split(projected_xy)
        if component_name == 'x':
            plot_values = field_x
            output_basis = field_x_basis
        else:
            plot_values = field_y
            output_basis = field_y_basis
    elif component_name in ('z', 'n'):
        output_basis = field_z_basis.with_element(ElementDG(ElementTriP1()))
        plot_values = output_basis.project(_component_part(field_z_basis.interpolate(field_z), part))
        component_name = 'z'
    else:
        raise ValueError("component must be 'x', 'y', or 'z'.")

    component_label = f'{field_name}{component_name}'
    if part == 'abs':
        quantity = f'|{component_label}|'
        z_label = f'normalized |{component_label}|'
        value_kind = 'nonnegative'
    else:
        quantity = f'{part}({component_label})'
        z_label = f'normalized {part}({component_label})'
        value_kind = 'signed'

    field_map = scalar_field_to_triangular_mesh(
        output_basis,
        plot_values,
        quantity=quantity,
        z_label=z_label,
        normalize=True,
        bounds=bounds,
    )
    field_map['field_key'] = f'{part}_{field_name}{component_name}'
    field_map['field_label'] = quantity
    field_map['value_kind'] = value_kind
    return field_map


def build_mode_field_maps(mode, bounds: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {'intensity': mode_intensity_map(mode, bounds)}
    for field_name in ('E', 'H'):
        for component_name in ('x', 'y', 'z'):
            key = f'abs_{field_name}{component_name}'
            try:
                fields[key] = mode_component_map(
                    mode,
                    field=field_name,
                    component=component_name,
                    part='abs',
                    bounds=bounds,
                )
            except Exception as exc:
                # Keep intensity usable even if a particular component cannot be
                # projected by an older femwell/skfem combination.  The UI simply
                # omits failed component maps.
                fields[key] = {
                    'type': 'error',
                    'field_key': key,
                    'field_label': f'|{field_name}{component_name}|',
                    'error': str(exc),
                }
    return {key: value for key, value in fields.items() if value.get('type') != 'error'}


def _import_solver_dependencies():
    try:
        from femwell.mesh import mesh_from_OrderedDict
        from femwell.maxwell.waveguide import compute_modes
        from skfem import Basis, ElementTriP0
        from skfem.io import from_meshio
    except Exception as exc:
        raise AnalysisImportError(
            'femwell or one of its meshing dependencies could not be imported. Install the packages from requirements.txt and retry.'
        ) from exc
    return mesh_from_OrderedDict, compute_modes, Basis, ElementTriP0, from_meshio


def _complex_dict(value: complex) -> dict[str, float]:
    return {'real': float(np.real(value)), 'imag': float(np.imag(value))}


def _mode_results_from_modes(
    modes,
    *,
    wavelength_um: float,
    focus_regions: list[str],
) -> list[dict[str, Any]]:
    mode_results: list[dict[str, Any]] = []
    for idx, mode in enumerate(modes):
        n_eff_value = complex(mode.n_eff)
        propagation_loss_dB_per_cm = float(
            20.0 * np.log10(np.e) * (2.0 * np.pi / wavelength_um) * abs(np.imag(n_eff_value)) * 1.0e4
        )
        entry: dict[str, Any] = {
            'mode_index': idx,
            'wavelength_um': float(wavelength_um),
            'wavelength_nm': float(wavelength_um) * 1000.0,
            'n_eff': _complex_dict(n_eff_value),
            'propagation_loss_dB_per_cm': propagation_loss_dB_per_cm,
            'te_fraction': float(np.real(mode.te_fraction)),
            'tm_fraction': float(np.real(mode.tm_fraction)),
        }

        reference_power = 0.0 + 0.0j
        reference_power_by_region: dict[str, dict[str, float]] = {}
        total_power = complex(mode.calculate_power())
        for region_name in focus_regions:
            value = complex(mode.calculate_power(elements=region_name))
            reference_power += value
            reference_power_by_region[region_name] = _complex_dict(value)
        if focus_regions:
            if abs(total_power) > 1e-30:
                reference_power_fraction = reference_power / total_power
            else:
                reference_power_fraction = 0.0 + 0.0j
            entry['power_reference_fraction'] = _complex_dict(reference_power_fraction)
            entry['power_reference'] = _complex_dict(reference_power)
            entry['power_total'] = _complex_dict(total_power)
            entry['power_reference_by_region'] = reference_power_by_region

        mode_results.append(entry)
    return mode_results


def _field_maps_from_modes(
    modes,
    *,
    wavelength_um: float,
    field_bounds: dict[str, float],
) -> list[dict[str, Any]]:
    mode_field_maps: list[dict[str, Any]] = []
    for idx, mode in enumerate(modes):
        fields = build_mode_field_maps(mode, field_bounds)
        for field_map in fields.values():
            field_map['mode_index'] = idx
            field_map['wavelength_um'] = float(wavelength_um)
            field_map['wavelength_nm'] = float(wavelength_um) * 1000.0
        mode_field_maps.append(
            {
                'mode_index': idx,
                'wavelength_um': float(wavelength_um),
                'wavelength_nm': float(wavelength_um) * 1000.0,
                'default_field': 'intensity',
                'fields': fields,
            }
        )
    return mode_field_maps


def _build_base_results(
    *,
    section_data: dict[str, Any],
    config: dict[str, Any],
    material_lookup: dict[str, complex],
    region_materials: dict[str, str],
    shapes: OrderedDict[str, object],
    analysis_window: dict[str, float],
    focus_regions: list[str],
    mesh_elements: int,
) -> dict[str, Any]:
    return {
        'unit': section_data.get('unit', 'um'),
        'num_modes': int(config['num_modes']),
        'order': int(config['order']),
        'analysis_window': analysis_window,
        'focus_regions': focus_regions,
        'regions': [{'name': name, 'material': region_materials[name]} for name in shapes.keys()],
        'materials': {
            name: {'n': _complex_dict(value)}
            for name, value in sorted(material_lookup.items())
        },
        'mesh_elements': int(mesh_elements),
    }



def _attach_group_index_to_sweep_runs(wavelength_runs: list[dict[str, Any]], wavelengths_um: list[float]) -> None:
    """Attach real group index n_g = n_eff - lambda * dn_eff/dlambda to sweep mode entries."""
    if len(wavelength_runs) < 2 or len(wavelength_runs) != len(wavelengths_um):
        return

    wavelengths = np.asarray(wavelengths_um, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size < 2 or not np.all(np.isfinite(wavelengths)):
        return

    mode_counts = [len(run.get('modes', [])) for run in wavelength_runs]
    if not mode_counts:
        return
    max_modes = max(mode_counts)
    edge_order = 2 if wavelengths.size >= 3 else 1

    for mode_index in range(max_modes):
        valid_run_indices: list[int] = []
        n_eff_values: list[float] = []
        wavelength_values: list[float] = []
        for run_index, run in enumerate(wavelength_runs):
            modes = run.get('modes', [])
            if mode_index >= len(modes):
                continue
            mode = modes[mode_index]
            try:
                n_eff_real = float(mode['n_eff']['real'])
                wavelength = float(wavelengths[run_index])
            except Exception:
                continue
            if not (np.isfinite(n_eff_real) and np.isfinite(wavelength)):
                continue
            valid_run_indices.append(run_index)
            n_eff_values.append(n_eff_real)
            wavelength_values.append(wavelength)

        if len(valid_run_indices) < 2:
            continue

        wl = np.asarray(wavelength_values, dtype=float)
        neff = np.asarray(n_eff_values, dtype=float)
        try:
            dneff_dlambda = np.gradient(neff, wl, edge_order=edge_order if len(wl) > edge_order else 1)
        except Exception:
            continue
        group_index = neff - wl * dneff_dlambda

        for local_index, run_index in enumerate(valid_run_indices):
            value = float(group_index[local_index])
            if np.isfinite(value):
                wavelength_runs[run_index]['modes'][mode_index]['group_index'] = value
                wavelength_runs[run_index]['modes'][mode_index]['group_index_method'] = 'finite_difference_sweep'

def _wavelengths_from_config(config: dict[str, Any]) -> list[float]:
    mode = str(config.get('wavelength_mode', config.get('analysis_type', 'single'))).strip().lower()
    if mode == 'sweep':
        explicit_values = config.get('sweep_wavelengths_um')
        if explicit_values is not None:
            wavelengths = [float(value) for value in explicit_values]
        else:
            start_nm = float(config.get('sweep_start_nm', float(config['wavelength_um']) * 1000.0))
            stop_nm = float(config.get('sweep_stop_nm', start_nm))
            points = int(config.get('sweep_points', 2))
            if points < 2:
                raise InputValidationError('Wavelength sweep requires at least two points.')
            wavelengths = [float(value) / 1000.0 for value in np.linspace(start_nm, stop_nm, points)]
    else:
        wavelengths = [float(config['wavelength_um'])]

    cleaned: list[float] = []
    for wavelength in wavelengths:
        if not np.isfinite(wavelength) or wavelength <= 0.0:
            raise InputValidationError('All wavelengths must be positive finite values.')
        cleaned.append(float(wavelength))
    if not cleaned:
        raise InputValidationError('At least one wavelength is required.')
    return cleaned


def run_mode_analysis(section_data: dict[str, Any], materials_data: dict[str, Any], config: dict[str, Any]) -> AnalysisArtifacts:
    shapes, region_materials = build_shapes(section_data)
    material_lookup = build_material_lookup(materials_data, config)
    shapes, region_materials, analysis_window = apply_analysis_window(shapes, region_materials, config)

    mesh_from_OrderedDict, compute_modes, Basis, ElementTriP0, from_meshio = _import_solver_dependencies()

    with tempfile.TemporaryDirectory(prefix='femwell_gui_') as tmpdir:
        mesh_path = Path(tmpdir) / 'mesh.msh'
        mesh = from_meshio(
            mesh_from_OrderedDict(
                shapes,
                build_resolutions(list(shapes.keys()), config),
                default_resolution_max=float(config['default_resolution_max']),
                filename=str(mesh_path),
            )
        )

    basis0 = Basis(mesh, ElementTriP0())
    epsilon = basis0.zeros().astype(complex)
    missing_materials: list[str] = []
    for region_name, material_name in region_materials.items():
        if material_name not in material_lookup:
            missing_materials.append(material_name)
            continue
        n_complex = material_lookup[material_name]
        dofs = basis0.get_dofs(elements=region_name)
        epsilon[dofs] = n_complex ** 2
    if missing_materials:
        missing = ', '.join(sorted(set(missing_materials)))
        raise InputValidationError(f'No refractive-index definition was found for: {missing}. Add the missing material(s) to the materials JSON.')

    focus_regions = [name for name in config.get('focus_regions', []) if name in shapes]
    field_bounds = analysis_window or get_geometry_bounds(shapes)
    wavelengths_um = _wavelengths_from_config(config)
    base_results = _build_base_results(
        section_data=section_data,
        config=config,
        material_lookup=material_lookup,
        region_materials=region_materials,
        shapes=shapes,
        analysis_window=analysis_window,
        focus_regions=focus_regions,
        mesh_elements=int(mesh.nelements),
    )

    wavelength_runs: list[dict[str, Any]] = []
    sweep_field_maps: list[dict[str, Any]] = []
    for sweep_index, wavelength_um in enumerate(wavelengths_um):
        modes = compute_modes(
            basis0,
            epsilon,
            wavelength=float(wavelength_um),
            num_modes=int(config['num_modes']),
            order=int(config['order']),
        )
        mode_results = _mode_results_from_modes(modes, wavelength_um=float(wavelength_um), focus_regions=focus_regions)
        mode_field_maps = _field_maps_from_modes(modes, wavelength_um=float(wavelength_um), field_bounds=field_bounds)
        run_results = {
            **base_results,
            'analysis_type': 'single',
            'sweep_index': int(sweep_index),
            'wavelength_um': float(wavelength_um),
            'wavelength_nm': float(wavelength_um) * 1000.0,
            'modes': mode_results,
        }
        wavelength_runs.append(run_results)
        sweep_field_maps.append(
            {
                'sweep_index': int(sweep_index),
                'wavelength_um': float(wavelength_um),
                'wavelength_nm': float(wavelength_um) * 1000.0,
                'mode_field_maps': mode_field_maps,
            }
        )

    _attach_group_index_to_sweep_runs(wavelength_runs, wavelengths_um)

    if len(wavelength_runs) == 1:
        single_results = wavelength_runs[0]
        single_results['analysis_type'] = 'single'
        return AnalysisArtifacts(
            results=single_results,
            mode_field_maps=sweep_field_maps[0]['mode_field_maps'],
            sweep_field_maps=None,
        )

    wavelengths_nm = [float(value) * 1000.0 for value in wavelengths_um]
    results: dict[str, Any] = {
        **base_results,
        'analysis_type': 'sweep',
        'wavelength_um': float(wavelengths_um[0]),
        'wavelength_nm': float(wavelengths_nm[0]),
        'wavelength_sweep': {
            'wavelengths_um': [float(value) for value in wavelengths_um],
            'wavelengths_nm': wavelengths_nm,
            'start_nm': float(wavelengths_nm[0]),
            'stop_nm': float(wavelengths_nm[-1]),
            'points': int(len(wavelengths_nm)),
            'runs': wavelength_runs,
        },
        # Keep the first sweep point in the legacy location so older UI code and
        # JSON consumers still have a valid representative mode table.
        'modes': wavelength_runs[0]['modes'],
    }

    return AnalysisArtifacts(
        results=results,
        mode_field_maps=sweep_field_maps[0]['mode_field_maps'],
        sweep_field_maps=sweep_field_maps,
    )
