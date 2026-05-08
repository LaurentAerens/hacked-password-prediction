"""Quick validation of syntax and imports."""
import ast
import sys
from pathlib import Path

# Try to parse ui_app.py with proper encoding
try:
    with open(Path(__file__).parent.parent / 'ai-resources' / 'ui_app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print("✓ ui_app.py: Valid Python syntax")
except SyntaxError as e:
    print(f"✗ ui_app.py: Syntax error at line {e.lineno}: {e.msg}")
    exit(1)

# Check imports work
try:
    ai_path = str(Path(__file__).parent.parent / 'ai-resources')
    sys.path.insert(0, ai_path)
    from shared_lib.realtime_dashboard import RealtimeDashboardState
    print("✓ RealtimeDashboardState: Import successful")
except Exception as e:
    print(f"✗ RealtimeDashboardState: Import failed: {e}")
    exit(1)

print("\n✓ All syntax and import checks passed")
