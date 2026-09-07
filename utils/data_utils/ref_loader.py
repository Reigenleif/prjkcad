import gc
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
from utils.representations.dual_seq.dual_seq import DualSeq


class RefLoader:
    """Load Text2CAD metadata from the CSV and attach parsed JSON payloads.
    
    Supported Source Data Types :
    - "text2cad": The original Text2CAD dataset.
    - "text2caddeepcad": The DeepCAD JSON with Text2CAD anotation
    """
    
    SOURCE_DATA_TYPES = {"text2cad", "text2caddeepcad"}

    def __init__(self, 
        data_root: str | Path, 
        csv_path: str | Path = "text2cad_v1.1.csv", 
        max_samples: int | None = None, 
        source_data_type: str = "text2cad"
    ):

        if source_data_type not in self.SOURCE_DATA_TYPES:
            raise ValueError(f"Unsupported source data type: {source_data_type}. Must be one of {self.SOURCE_DATA_TYPES}")
        
        self.data_root = Path(data_root).expanduser().resolve()
        csv_path_obj = Path(csv_path)
        if csv_path_obj.is_absolute() or csv_path_obj.exists():
            self.csv_path = csv_path_obj.expanduser().resolve()
        else:
            self.csv_path = (self.data_root / csv_path_obj).expanduser().resolve()
        self.failed_uids: list[str] = []
        self.max_samples = max_samples
        self.source_data_type = source_data_type

    def load_csv(self) -> pd.DataFrame:
        """Load the text2cad CSV from the configured data root."""
        print(f"Loading CSV from: {self.csv_path}")
        if self.max_samples is not None:
            df = pd.read_csv(self.csv_path, nrows=self.max_samples)
        else:
            df = pd.read_csv(self.csv_path)
        return df

    def load_jsons(
        self,
        df: pd.DataFrame | None = None,
        uid_column: str = "uid",
        json_column: str = "json_target",
    ) -> pd.DataFrame:
        """Attach parsed JSON content to `json_column` and return only valid rows."""
        if df is None:
            df = self.load_csv()

        print(f"Processing {len(df)} rows to load JSON payloads...")
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
        if self.source_data_type == "deepcad":
            candidate = self.data_root / folder / f"{item_id}.json"
            if candidate.exists() and candidate.is_file():
                return candidate
            return None
        elif self.source_data_type == "text2cad":
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


def load_split_data(
    data_folder: str,
    metadata_csv: str,
    source_data_type: str = "text2cad",
    split_json: str | None = None,
    max_samples: int | None = None,
    sample_ratio: float | None = None,
    val_max_sample: int | None = None,
    val_sample_ratio: float | None = None,
    random_train: bool = False,
    seed: int = 42,
) -> tuple[list[DualSeq], list[DualSeq] | None]:
    if split_json and os.path.exists(split_json):
        print(f"Loading split from JSON: {split_json}")
        with open(split_json, "r") as f:
            splits = json.load(f)
        train_uids = set(splits.get("train", []))
        val_uids = set(splits.get("validation", []))
        
        loader = RefLoader(
            data_folder,
            csv_path=metadata_csv,
            max_samples=None,
            source_data_type=source_data_type
        )
        df_all = loader.load_csv()
        df_all["normalized_uid"] = df_all["uid"].apply(loader._normalize_uid)
        
        df_train = df_all[df_all["normalized_uid"].isin(train_uids)]
        df_val = df_all[df_all["normalized_uid"].isin(val_uids)]
        
        if sample_ratio and sample_ratio < 1.0:
            sample_size = int(len(df_train) * sample_ratio)
            df_train = df_train.sample(n=sample_size, random_state=seed)
            
        v_ratio = val_sample_ratio if val_sample_ratio is not None else (sample_ratio if sample_ratio and sample_ratio < 1.0 else None)
        if v_ratio is not None and v_ratio == 0.0:
            df_val = df_val.head(0)
        elif v_ratio and v_ratio < 1.0:
            val_sample_size = int(len(df_val) * v_ratio)
            df_val = df_val.sample(n=val_sample_size, random_state=seed)
            
        if random_train and max_samples is not None and len(df_train) > max_samples:
            df_train = df_train.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        elif max_samples is not None:
            df_train = df_train.head(max_samples)
            
        v_max = val_max_sample 
        if v_max is not None:
            df_val = df_val.head(v_max)
            
        df_train_loaded = loader.load_jsons(df=df_train)
        df_val_loaded = loader.load_jsons(df=df_val) if not df_val.empty else pd.DataFrame()
        
        dual_seqs = DualSeq.from_text2cad_df(df_train_loaded)
        val_dual_seqs = DualSeq.from_text2cad_df(df_val_loaded) if not df_val_loaded.empty else None
            
        del df_all, df_train, df_val, df_train_loaded, df_val_loaded
        gc.collect()
        return dual_seqs, val_dual_seqs
    else:
        loader = RefLoader(
            data_folder,
            csv_path=metadata_csv,
            max_samples=max_samples,
            source_data_type=source_data_type
        )
        df = loader.load()
        dual_seqs = DualSeq.from_text2cad_df(df)
        
        if sample_ratio and sample_ratio < 1.0:
            sample_size = int(len(dual_seqs) * sample_ratio)
            dual_seqs = random.sample(dual_seqs, sample_size)
            
        return dual_seqs, None


def load_val_data(
    data_folder: str,
    metadata_csv: str,
    source_data_type: str = "text2cad",
    split_json: str | None = None,
    val_max_sample: int | None = None,
) -> list[DualSeq]:
    """Load only validation split sequence data."""
    if split_json and os.path.exists(split_json):
        with open(split_json, "r") as f:
            splits = json.load(f)
        val_uids = set(splits.get("validation", []))

        loader = RefLoader(
            data_folder,
            csv_path=metadata_csv,
            max_samples=None,
            source_data_type=source_data_type
        )
        df_all = loader.load_csv()
        df_all["normalized_uid"] = df_all["uid"].apply(loader._normalize_uid)
        df_val = df_all[df_all["normalized_uid"].isin(val_uids)]

        if val_max_sample is not None:
            df_val = df_val.head(val_max_sample)

        df_val_loaded = loader.load_jsons(df=df_val)
        return DualSeq.from_text2cad_df(df_val_loaded)
    return []


def load_test_data(
    data_folder: str,
    metadata_csv: str,
    source_data_type: str = "text2cad",
    split_json: str | None = None,
    test_max_sample: int | None = None,
    test_sample_ratio: float | None = None,
) -> list[DualSeq]:
    """Load only test split sequence data."""
    if split_json and os.path.exists(split_json):
        with open(split_json, "r") as f:
            splits = json.load(f)
        test_uids = set(splits.get("test", []))

        loader = RefLoader(
            data_folder,
            csv_path=metadata_csv,
            max_samples=None,
            source_data_type=source_data_type
        )
        df_all = loader.load_csv()
        df_all["normalized_uid"] = df_all["uid"].apply(loader._normalize_uid)
        df_test = df_all[df_all["normalized_uid"].isin(test_uids)]

        if test_sample_ratio and test_sample_ratio < 1.0:
            test_sample_size = int(len(df_test) * test_sample_ratio)
            df_test = df_test.sample(n=test_sample_size, random_state=42)

        if test_max_sample is not None:
            df_test = df_test.head(test_max_sample)

        df_test_loaded = loader.load_jsons(df=df_test)
        return DualSeq.from_text2cad_df(df_test_loaded)
    return []
