from __future__ import annotations

import json
import sys
from pathlib import Path

from engine import AnalysisImportError, InputValidationError, run_mode_analysis


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit('Usage: solver_runner.py section.json materials.json config.json output.json')

    section_path = Path(sys.argv[1])
    materials_path = Path(sys.argv[2])
    config_path = Path(sys.argv[3])
    output_path = Path(sys.argv[4])

    section_data = json.loads(section_path.read_text(encoding='utf-8'))
    materials_data = json.loads(materials_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))

    try:
        artifacts = run_mode_analysis(section_data, materials_data, config)
        payload = {
            'ok': True,
            'results': artifacts.results,
            'mode_field_maps': artifacts.mode_field_maps,
            'sweep_field_maps': artifacts.sweep_field_maps or [],
        }
    except (InputValidationError, AnalysisImportError) as exc:
        payload = {'ok': False, 'error': str(exc)}
    except Exception as exc:
        payload = {'ok': False, 'error': f'Unexpected solver error: {exc}'}

    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
