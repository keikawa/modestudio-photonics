from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

from streamlit.web import cli as stcli

SOLVER_CLI_FLAG = '--modestudio-solver-runner'


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
    return base / relative


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def run_solver_runner_from_bundle() -> int:
    if len(sys.argv) != 6 or sys.argv[1] != SOLVER_CLI_FLAG:
        raise SystemExit(
            'Usage: modestudio-backend.exe '
            f'{SOLVER_CLI_FLAG} section.json materials.json config.json output.json'
        )

    bundle_dir = resource_path('.')
    os.chdir(str(bundle_dir))
    sys.path.insert(0, str(bundle_dir))

    from solver_runner import main as solver_main

    sys.argv = [str(resource_path('solver_runner.py')), *sys.argv[2:]]
    return int(solver_main())


def main() -> None:
    app_path = resource_path('app.py')
    os.chdir(str(app_path.parent))

    port = find_free_port()
    url = f'http://127.0.0.1:{port}'
    print(f'MODESTUDIO_URL={url}', flush=True)

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        f"--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--browser.gatherUsageStats=false",
        "--client.toolbarMode=viewer",
        "--global.developmentMode=false",
    ]

    stcli.main()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == SOLVER_CLI_FLAG:
        raise SystemExit(run_solver_runner_from_bundle())
    main()
