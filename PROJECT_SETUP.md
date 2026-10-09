# PROJECT_SETUP.md

## System Overview
- **OS**: Microsoft Windows 11 Pro (Build `10.0.26200`)
- **Default Shell**: PowerShell 5.1 (`5.1.26100.9444`) / Windows PowerShell
- **Host Architecture**: x64 (AMD64)
- **Primary Workspace Root**: `D:\RAG`

---

## Python Installations and Exact Executable Paths

### Python 3.12 (Primary Global Installation)
- **Executable**: `C:\Users\mohamed\AppData\Local\Programs\Python\Python312\python.exe`
- **Version**: `Python 3.12.3` (tags/v3.12.3:f6650f9, Apr 9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)]
- **Prefix / Home**: `C:\Users\mohamed\AppData\Local\Programs\Python\Python312`
- **Scripts Directory**: `C:\Users\mohamed\AppData\Local\Programs\Python\Python312\Scripts`
- **Notes**: Windows App Execution Alias (`python.exe` under `WindowsApps`) is present but points to the Microsoft Store installer stub. Always invoke the full path or ensure Python 3.12 is in `PATH`.

---

## Project Virtual Environments
- **Active / Project Virtual Environment**: None found in workspace root (`D:\RAG`).
- **Mode**: Global Python installation (`C:\Users\mohamed\AppData\Local\Programs\Python\Python312\python.exe`) or create a new `.venv` using:
  ```powershell
  & "C:\Users\mohamed\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

---

## pip and Package Locations
- **pip Executable**: `C:\Users\mohamed\AppData\Local\Programs\Python\Python312\Scripts\pip.exe` (or invoke via `python -m pip`)
- **pip Version**: `24.0`
- **Site-Packages Directory**: `C:\Users\mohamed\AppData\Local\Programs\Python\Python312\Lib\site-packages`
- **Key Installed Libraries**:
  - Web & API: `fastapi` (0.141.1), `uvicorn` (0.53.0), `Flask` (3.1.0), `httpx` (0.28.1), `requests` (2.32.5), `streamlit` (1.52.1)
  - Data Science & ML: `numpy` (2.2.0), `pandas` (2.3.3), `scipy` (1.14.1), `scikit-learn` (1.6.0), `xgboost` (3.2.0), `lightgbm` (4.6.0), `catboost` (1.2.10), `optuna` (4.8.0), `shap` (0.51.0)
  - Inference / Runtime: `onnx` (1.21.0), `onnxruntime` (1.25.1), `pyspark` (4.1.0)
  - Documents & Parsing: `PyMuPDF` (1.27.2.3), `beautifulsoup4` (4.14.2), `pydantic` (2.13.5)
  - Tooling & Linting: `ruff` (0.16.7), `pytest` / `ipykernel` (7.1.0), `jupyterlab` (4.5.0)

---

## PyTorch and CUDA
- **PyTorch**: Not installed in global Python 3.12.
- **CUDA Toolkit (Installed Metadata)**: CUDA SDK `13.0.1` located at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0` (Metadata file: `version.json`).
- **CUDA Binaries in PATH**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin` and `bin\x64`.

---

## GPU and Development Tools
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU
- **VRAM**: 6144 MiB (6 GB)
- **NVIDIA Driver Version**: `546.30`
- **`nvidia-smi` Location**: `C:\Windows\system32\nvidia-smi.exe`
- **Git**: `C:\Program Files\Git\cmd\git.exe` (`git version 2.47.0.windows.2`)
- **CMake**: `C:\Program Files\CMake\bin\cmake.exe` (`cmake version 4.0.0-rc2`)
- **C/C++ Compilers (`cl.exe`, `gcc.exe`)**: Not found in default system `PATH`.

---

## Important Environment Variables
- `PATH` contains:
  - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64`
  - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`
  - `C:\Program Files\Git\cmd`
  - `C:\Program Files\CMake\bin`
- `CUDA_PATH`: Not explicitly set in environment (defaults to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0` if configured manually).
- `CUDA_HOME`: Not set.

---

## Project Structure
- **Root Directory**: `D:\RAG`
- **Initial Contents**: Empty project directory.
- **Dependency Specification Files**: None currently present.

---

## Common Run and Test Commands

### Run Python scripts using exact executable:
```powershell
& "C:\Users\mohamed\AppData\Local\Programs\Python\Python312\python.exe" script.py
```

### Install packages:
```powershell
& "C:\Users\mohamed\AppData\Local\Programs\Python\Python312\python.exe" -m pip install <package-name>
```

### Format / Lint with Ruff:
```powershell
& "C:\Users\mohamed\AppData\Local\Programs\Python\Python312\python.exe" -m ruff check .
```

---

## Unknown or Unverified Information
- **C/C++ Compiler Build Tools**: MSVC Build Tools / MinGW are not detected in standard system `PATH`.
- **Conda Environment**: Conda is not installed or not present in `PATH`.
