try:
    import pytz
    print("✅ pytz installed")
except ImportError:
    print("❌ pytz NOT installed")

try:
    from query_data import query_rag
    print("✅ query_data.py importable")
except ImportError as e:
    print(f"❌ query_data.py import failed: {e}")

try:
    from create_database import generate_data_store
    print("✅ create_database.py importable")
except ImportError as e:
    print(f"❌ create_database.py import failed: {e}")

import json
import os
if not os.path.exists("chat_history.json"):
    with open("chat_history.json", "w") as f:
        json.dump([], f)
    print("✅ chat_history.json created")
else:
    print("✅ chat_history.json exists")
