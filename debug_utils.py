import json
import os
from typing import Any, Dict


DEBUG_ENABLED = os.getenv("DEBUG_AGENT", "1").lower() not in {"0", "false", "no"}


def _shorten(value: Any, limit: int = 240) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    if isinstance(value, dict):
        return {k: _shorten(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_shorten(v, limit) for v in value]
    return value


def debug_log(node: str, label: str, payload: Dict[str, Any]) -> None:
    if not DEBUG_ENABLED:
        return
    safe_payload = {k: _shorten(v) for k, v in payload.items()}
    print(f"\n[DEBUG][{node}] {label}:")
    print(json.dumps(safe_payload, indent=2))

