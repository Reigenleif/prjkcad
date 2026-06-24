from __future__ import annotations

import os
import json
import re
import math
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from ..dual_seq import DualSeq


class CoresetCreator:
    """Class to create a coreset from DualSeq sequences by clustering their descriptions."""

    def __init__(
        self,
        dual_seqs: list[DualSeq] | None = None,
        data_root: str | None = None,
        source_data_type: str = "text2cad",
        description_level: str = "abstract",
        p: float = 0.01,
        k: int = 100,
        model_name: str = "bert-base-uncased",
        batch_size: int = 32,
        device: str | None = None,
        random_seed: int = 42,
    ):
        """
        Args:
            dual_seqs: List of DualSeq instances. If None, loaded using data_root and source_data_type.
            data_root: Root directory of the dataset (required if dual_seqs is None).
            source_data_type: The source data type to load (e.g. 'text2cad', 'text2caddeepcad').
            description_level: The description level to use for embedding (e.g. 'abstract', 'beginner', 'intermediate', 'expert').
            p: The cluster portion to pick from each cluster.
            k: The number of clusters for KMeans.
            model_name: The name/path of the BERT-based Hugging Face model for embeddings.
            batch_size: Batch size used for embedding extraction.
            device: Device to run the embedding model on (e.g. 'cuda', 'cpu'). If None, automatically determined.
            random_seed: Random seed for KMeans clustering.
        """
        if dual_seqs is None:
            if data_root is None:
                raise ValueError("Either dual_seqs or data_root must be provided.")
            from .ref_loader import RefLoader
            loader = RefLoader(data_root=data_root, source_data_type=source_data_type)
            self.ref_loader = loader


        

        self.dual_seqs = dual_seqs or None
        self.description_level = description_level
        self.p = p
        self.k = k
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed

    def get_embeddings(self) -> np.ndarray:
        """Embed all dual_seq descriptions for the configured description level using BERT."""
        descriptions = []
        for i, ds in enumerate(self.dual_seqs):
            if self.description_level not in ds.descriptions:
                raise ValueError(
                    f"DualSeq at index {i} (uid: {ds.uid}) is missing the description level '{self.description_level}'"
                )
            descriptions.append(ds.descriptions[self.description_level])

        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertModel.from_pretrained(self.model_name).to(self.device)
        model.eval()

        embeddings = []
        with torch.no_grad():
            pbar    = tqdm(range(0, len(descriptions), self.batch_size), desc="Extracting embeddings")
            for i in pbar:
                batch_texts = descriptions[i : i + self.batch_size]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                outputs = model(**encoded)
                # Compute mean pooling
                attention_mask = encoded["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask

                embeddings.append(mean_embeddings.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def create_coreset(
        self,
        selection_method: str = "closest",
        save_path: str | None = None,
        force_recreate: bool = False
    ) -> list[DualSeq]:
        """
        Embed all descriptions, cluster them using KMeans with k clusters,
        and select p portion of samples from each cluster. Saves the coreset
        to out/coreset, or loads it from there if the path is found.

        Args:
            selection_method: How to select samples from each cluster:
                - 'closest': Select samples closest to the cluster center.
                - 'random': Select samples randomly.
                - 'farthest': Select samples farthest from the cluster center.
            save_path: The file path to save/load the coreset. If None,
                it defaults to a path under 'out/coreset' constructed from parameters.
            force_recreate: If True, recalculate the coreset even if save_path exists.

        Returns:
            A list of selected DualSeq instances forming the coreset.
        """
        if selection_method not in {"closest", "random", "farthest"}:
            raise ValueError(
                f"Unsupported selection_method: {selection_method}. Must be 'closest', 'random', or 'farthest'."
            )

        if save_path is None:
            # Construct a safe default path under out/coreset
            safe_model_name = re.sub(r"[^a-zA-Z0-9_-]", "_", self.model_name)
            filename = (
                f"coreset_{self.description_level}_p{self.p}_k{self.k}_"
                f"{safe_model_name}_{selection_method}_seed{self.random_seed}.json"
            )
            save_path = os.path.join("out", "coreset", filename)
        
        if not force_recreate and os.path.exists(save_path):
            print(f"Loading coreset from cached path: {save_path}")
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            coreset = []
            for item in data:
                ds = DualSeq(
                    json_object=item["json_object"],
                    uid=item["uid"],
                    format=item.get("format", "text2cad"),
                    descriptions=item["descriptions"]
                )
                coreset.append(ds)
            return coreset

        if self.dual_seqs is None:
            df = self.ref_loader.load()
            self.dual_seqs = DualSeq.from_text2cad_df(df)

        if len(self.dual_seqs) == 0:
            raise ValueError("The dual_seqs list cannot be empty, check the data root and source data type.")

        embeddings = self.get_embeddings()
        num_samples = len(self.dual_seqs)

        # Adjust n_clusters if k exceeds number of samples
        n_clusters = min(self.k, num_samples)
        if n_clusters < self.k:
            print(
                f"Warning: k ({self.k}) is larger than the number of samples ({num_samples}). "
                f"Adjusting clusters to {n_clusters}."
            )

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings)
        centers = kmeans.cluster_centers_

        selected_indices = []
        rng = np.random.default_rng(self.random_seed)

        pbar = tqdm(range(n_clusters), desc="Selecting coreset samples")
        for cluster_idx in pbar:
            # Find indices of samples belonging to this cluster
            indices_in_cluster = np.where(cluster_labels == cluster_idx)[0]
            if len(indices_in_cluster) == 0:
                continue

            # Calculate how many samples to pick
            num_to_pick = int(np.round(len(indices_in_cluster) * self.p))
            # Ensure at least 1 sample is picked from each cluster if p > 0
            if num_to_pick == 0 and self.p > 0:
                num_to_pick = 1

            if num_to_pick >= len(indices_in_cluster):
                # Pick all samples in the cluster
                selected_indices.extend(indices_in_cluster.tolist())
                continue

            if selection_method == "random":
                picked = rng.choice(indices_in_cluster, size=num_to_pick, replace=False)
                selected_indices.extend(picked.tolist())
            elif selection_method in {"closest", "farthest"}:
                cluster_embeddings = embeddings[indices_in_cluster]
                center = centers[cluster_idx]
                # Compute Euclidean distances to cluster center
                distances = np.linalg.norm(cluster_embeddings - center, axis=1)

                # Sort indices based on distance
                if selection_method == "closest":
                    sorted_order = np.argsort(distances)
                else:  # farthest
                    sorted_order = np.argsort(distances)[::-1]

                picked_indices_in_cluster = sorted_order[:num_to_pick]
                picked = indices_in_cluster[picked_indices_in_cluster]
                selected_indices.extend(picked.tolist())

        # Sort the final selected indices to maintain the relative ordering of the original list
        selected_indices.sort()

        coreset = [self.dual_seqs[idx] for idx in selected_indices]

        # Save to save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        serializable_data = []
        for ds in coreset:
            fmt = "text2caddeepcad" if "entities" in ds.json_object else "text2cad"
            serializable_data.append({
                "uid": ds.uid,
                "json_object": ds.json_object,
                "descriptions": ds.descriptions,
                "format": fmt
            })
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2)
        print(f"Saved coreset to: {save_path}")

        return coreset
