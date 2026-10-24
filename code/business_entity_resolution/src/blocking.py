import re
import pandas as pd
from rapidfuzz import process, fuzz
from unidecode import unidecode


def normalize_block_text(value):
    """
    Normalize text for candidate generation.

    Converts non-Latin scripts into a Latin representation where possible.
    """
    if pd.isna(value):
        return ""

    value = unidecode(str(value)).lower()

    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def make_block_keys(df):
    """
    Create blocking representations.
    """
    out = df.copy()

    out["name_norm"] = out["business_name"].map(
        normalize_block_text
    )

    out["address_norm"] = out["business_address"].map(
        normalize_block_text
    )

    out["country_norm"] = (
        out["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    out["block_name_exact"] = out["name_norm"]

    out["block_name_first"] = (
        out["name_norm"]
        .str.split()
        .str[0]
        .fillna("")
    )

    out["block_name_prefix"] = (
        out["name_norm"]
        .str[:4]
    )

    out["block_address_first"] = (
        out["address_norm"]
        .str.split()
        .str[0]
        .fillna("")
    )

    def address_tokens(value):
        tokens = [
            token
            for token in value.split()
            if len(token) >= 3
        ]

        return " ".join(
            sorted(set(tokens))[:8]
        )

    out["block_address_tokens"] = (
        out["address_norm"].map(address_tokens)
    )

    return out


class Blocker:
    """
    Scalable candidate generator.

    Uses deterministic blocking keys:
      1. Exact normalized business name
      2. First business-name token
      3. Four-character name prefix
      4. First address token
      5. Normalized address-token signature

    Fuzzy matching is optional and disabled by default for full-scale runs.
    """

    def __init__(
        self,
        max_candidates_per_key=20,
        fuzzy_name_limit=5,
        fuzzy_address_limit=5,
        fuzzy_name_score=70,
        fuzzy_address_score=60,
        use_fuzzy=False,
    ):
        self.max_candidates_per_key = max_candidates_per_key

        self.fuzzy_name_limit = fuzzy_name_limit
        self.fuzzy_address_limit = fuzzy_address_limit
        self.fuzzy_name_score = fuzzy_name_score
        self.fuzzy_address_score = fuzzy_address_score

        self.use_fuzzy = use_fuzzy

        self.target = None

        self.indices = {}

        self.name_choices = {}
        self.address_choices = {}

    def fit(self, target):
        """
        Build compact lookup indices for the target source.
        """

        self.target = make_block_keys(
            target
        ).reset_index(drop=True)

        key_columns = [
            "block_name_exact",
            "block_name_first",
            "block_name_prefix",
            "block_address_first",
            "block_address_tokens",
        ]

        self.indices = {}

        for key_name in key_columns:

            grouped = (
                self.target[
                    [key_name]
                ]
                .reset_index()
                .rename(
                    columns={
                        "index": "target_idx"
                    }
                )
            )

            grouped = grouped[
                grouped[key_name].notna()
                & (grouped[key_name] != "")
            ]

            # Keep only the first N target records
            # per blocking key.
            grouped = (
                grouped
                .groupby(
                    key_name,
                    sort=False
                )["target_idx"]
                .apply(
                    lambda x:
                    x.iloc[
                        :self.max_candidates_per_key
                    ].tolist()
                )
                .to_dict()
            )

            self.indices[key_name] = grouped

        if self.use_fuzzy:

            self.name_choices = {
                idx: value
                for idx, value in enumerate(
                    self.target["name_norm"]
                )
                if value
            }

            self.address_choices = {
                idx: value
                for idx, value in enumerate(
                    self.target["address_norm"]
                )
                if value
            }

        return self

    def _deterministic_candidates(self, source1):
        """
        Vectorized-ish deterministic candidate generation.

        The actual joins happen through pandas merge operations
        rather than iterating over every Source 1 row.
        """

        source1 = make_block_keys(
            source1
        ).reset_index(drop=True)

        candidate_frames = []

        key_columns = [
            "block_name_exact",
            "block_name_first",
            "block_name_prefix",
            "block_address_first",
            "block_address_tokens",
        ]

        # Build lookup tables once.
        for key_name in key_columns:

            lookup_rows = []

            for key, target_indices in self.indices[
                key_name
            ].items():

                for target_idx in target_indices:

                    lookup_rows.append(
                        (
                            key,
                            target_idx,
                        )
                    )

            if not lookup_rows:
                continue

            lookup = pd.DataFrame(
                lookup_rows,
                columns=[
                    key_name,
                    "_target_idx",
                ],
            )

            left = source1[
                [
                    "entity_id",
                    key_name,
                ]
            ]

            merged = left.merge(
                lookup,
                on=key_name,
                how="inner",
                sort=False,
            )

            if len(merged) == 0:
                continue

            target_ids = self.target[
                "entity_id"
            ].to_numpy()

            merged["candidate_entity_id"] = (
                merged["_target_idx"]
                .map(
                    lambda idx:
                    target_ids[idx]
                )
            )

            merged = merged[
                [
                    "entity_id",
                    "candidate_entity_id",
                ]
            ]

            merged = merged.rename(
                columns={
                    "entity_id":
                    "source1_entity_id"
                }
            )

            candidate_frames.append(
                merged
            )

        if not candidate_frames:

            return pd.DataFrame(
                columns=[
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            )

        result = pd.concat(
            candidate_frames,
            ignore_index=True,
        )

        result = result.drop_duplicates(
            subset=[
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )

        return result.reset_index(
            drop=True
        )

    def _fuzzy_candidates(self, source1):
        """
        Optional fuzzy candidate generation.

        This is intentionally separate from the deterministic
        path because it is expensive at large scale.
        """

        if not self.use_fuzzy:
            return pd.DataFrame(
                columns=[
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            )

        source1 = make_block_keys(
            source1
        ).reset_index(drop=True)

        rows = []

        for _, s1 in source1.iterrows():

            candidate_indices = set()

            name = s1["name_norm"]

            if name and self.name_choices:

                matches = process.extract(
                    name,
                    self.name_choices,
                    scorer=fuzz.ratio,
                    limit=self.fuzzy_name_limit,
                    score_cutoff=self.fuzzy_name_score,
                )

                for _, score, target_idx in matches:
                    candidate_indices.add(
                        target_idx
                    )

            address = s1["address_norm"]

            if address and self.address_choices:

                matches = process.extract(
                    address,
                    self.address_choices,
                    scorer=fuzz.ratio,
                    limit=self.fuzzy_address_limit,
                    score_cutoff=self.fuzzy_address_score,
                )

                for _, score, target_idx in matches:
                    candidate_indices.add(
                        target_idx
                    )

            for target_idx in candidate_indices:

                rows.append(
                    {
                        "source1_entity_id":
                            s1["entity_id"],

                        "candidate_entity_id":
                            self.target.iloc[
                                target_idx
                            ]["entity_id"],
                    }
                )

        return pd.DataFrame(
            rows,
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
        )

    def generate(self, source1):
        """
        Generate candidate pairs.
        """

        deterministic = (
            self._deterministic_candidates(
                source1
            )
        )

        if not self.use_fuzzy:
            return deterministic

        fuzzy = (
            self._fuzzy_candidates(
                source1
            )
        )

        if len(fuzzy) == 0:
            return deterministic

        result = pd.concat(
            [
                deterministic,
                fuzzy,
            ],
            ignore_index=True,
        )

        return result.drop_duplicates(
            subset=[
                "source1_entity_id",
                "candidate_entity_id",
            ]
        ).reset_index(
            drop=True
        )