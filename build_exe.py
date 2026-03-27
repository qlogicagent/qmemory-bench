"""PyInstaller build script for QMemory Benchmark UI.

Usage:
    python build_exe.py

Produces: dist/qmemory-bench.exe (single-file Windows executable)
"""

import PyInstaller.__main__

PyInstaller.__main__.run([
    "src/qmemory_bench/ui/app.py",
    "--name=qmemory-bench",
    "--onefile",
    "--windowed",
    "--icon=NONE",
    "--add-data=data;data",
    "--hidden-import=nicegui",
    "--hidden-import=qmemory_bench",
    "--hidden-import=qmemory_bench.runner",
    "--hidden-import=qmemory_bench.judge",
    "--hidden-import=qmemory_bench.providers",
    "--hidden-import=qmemory_bench.dataset",
    "--collect-all=nicegui",
    f"--paths=src",
    "--noconfirm",
])
