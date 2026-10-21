import sys
import importlib.util
import subprocess

def check_package(package_name, required_version=None):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"❌ [MISSING] {package_name} is not installed.")
        return False
    
    mod = __import__(package_name)
    version = getattr(mod, "__version__", "unknown")
    if required_version and version != required_version:
        print(f"⚠️ [VERSION MISMATCH] {package_name}: found {version}, required {required_version}")
    else:
        print(f"✅ [READY] {package_name} version {version} is correctly installed.")
    return True

print("=== SYSTEM DIAGNOSTIC REPORT ===")
print(f"Python Version: {sys.version.split()[0]} (Target: 3.14.7)")

check_package("pandas", "3.0.5")
check_package("sklearn", "1.9.1")
check_package("joblib", "1.6.0")

git_check = subprocess.run(["git", "--version"], capture_output=True, text=True)
if git_check.returncode == 0:
    print(f"✅ [READY] Git is installed: {git_check.stdout.strip()}")
else:
    print("❌ [MISSING] Git is not found in system PATH.")
print("================================")