"""Check the local Python environment required by the ML skills."""
import importlib
import sys

REQUIRED = (("numpy", "NumPy"), ("matplotlib", "Matplotlib"))

print(f"Python {sys.version.split()[0]}: OK")
failed = False
for module, label in REQUIRED:
    try:
        imported = importlib.import_module(module)
        print(f"{label} {getattr(imported, '__version__', 'installed')}: OK")
    except ImportError:
        failed = True
        print(f"{label}: MISSING")

if failed:
    print("\n请在当前 Python 环境执行：python -m pip install -r requirements.txt")
    sys.exit(1)
print("\n环境检查通过，可以运行技能生成的代码。")
