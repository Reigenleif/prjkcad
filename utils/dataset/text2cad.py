from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


class Text2CADLoader:
    """Load Text2CAD metadata from the CSV and attach parsed JSON payloads."""

    def __init__(self, data_root: str | Path, csv_path: str | Path = "text2cad_v1.1.csv"):
        self.data_root = Path(data_root).expanduser().resolve()
        self.csv_path = self.data_root / Path(csv_path)
        self.failed_uids: list[str] = []

    def load_csv(self) -> pd.DataFrame:
        """Load the text2cad CSV from the configured data root."""
        return pd.read_csv(self.csv_path)

    def load_jsons(
        self,
        df: pd.DataFrame | None = None,
        uid_column: str = "uid",
        json_column: str = "json_target",
    ) -> pd.DataFrame:
        """Attach parsed JSON content to `json_column` and return only valid rows."""
        if df is None:
            df = self.load_csv()

        if uid_column not in df.columns:
            raise KeyError(f"Missing required column: {uid_column}")

        valid_rows: list[dict[str, Any]] = []
        self.failed_uids = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Text2CAD JSONs"):
            uid = self._normalize_uid(row[uid_column])
            json_path = self._resolve_json_path(uid)

            if json_path is None:
                self.failed_uids.append(uid)
                continue

            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                self.failed_uids.append(uid)
                continue

            row_data = row.to_dict()
            row_data[uid_column] = uid
            row_data[json_column] = payload
            valid_rows.append(row_data)

        result = pd.DataFrame(valid_rows).reset_index(drop=True)
        return result

    def load(self, uid_column: str = "uid", json_column: str = "json_target") -> pd.DataFrame:
        """Convenience method that loads the CSV and filters to valid JSON rows."""
        return self.load_jsons(uid_column=uid_column, json_column=json_column)

    def _resolve_json_path(self, uid: str) -> Path | None:
        folder, item_id = self._split_uid(uid)
        base_dir = self.data_root / folder / item_id / "minimal_json"

        candidates = [
            base_dir / item_id,
            base_dir / f"{item_id}.json",
            base_dir / f"{item_id}_merged_vlm.json",
            base_dir / f"{item_id}_merged.json",
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        return None

    def _split_uid(self, uid: str) -> tuple[str, str]:
        normalized = uid.strip().replace("\\", "/")
        if "/" in normalized:
            folder, item_id = normalized.split("/", 1)
            return folder.zfill(4), item_id.zfill(8)

        item_id = normalized.zfill(8)
        return item_id[:4], item_id

    def _normalize_uid(self, uid: Any) -> str:
        uid_text = str(uid).strip()
        folder, item_id = self._split_uid(uid_text)
        return f"{folder}/{item_id}"
