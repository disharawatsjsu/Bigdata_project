"""Colab helper: save regression artifacts to HDFS via HttpFS.

Paste/run this after the regression notebook has trained models and computed
SHAP. It expects these notebook variables to exist:

  NGROK_URL, HDFS_USER, LOCAL_ARTIFACT_ROOT
  COMMODITIES, TARGETS, FEATURE_COLS
  results, shap_results, metrics_df, attribution_df
  VERDICT, avg_test_r2, models_improving

This uses HttpFS/WebHDFS CREATE redirects directly so parquet uploads preserve
their bytes over ngrok.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode, urlparse, urlunparse

import numpy as np
import pandas as pd
import requests
from hdfs import InsecureClient


class HdfsBridge:
    """Small HttpFS client for Colab -> local HDFS writes."""

    def __init__(self, webhdfs_url: str, user: str = "root"):
        self._client = InsecureClient(webhdfs_url.rstrip("/"), user=user)
        self.url = webhdfs_url.rstrip("/")
        self.user = user

    def mkdirs(self, hdfs_dir: str) -> None:
        self._client.makedirs(hdfs_dir)

    def write_bytes(self, hdfs_path: str, content: bytes, overwrite: bool = True) -> None:
        params = {"op": "CREATE", "user.name": self.user, "overwrite": str(overwrite).lower()}
        create_url = f"{self.url}/webhdfs/v1{quote(hdfs_path, safe='/')}?{urlencode(params)}"
        headers = {"ngrok-skip-browser-warning": "true"}
        first = requests.put(create_url, headers=headers, timeout=60, allow_redirects=False)
        first.raise_for_status()
        location = first.headers.get("Location")
        if not location:
            raise RuntimeError(f"HttpFS CREATE did not return a redirect for {hdfs_path}")

        target = self._rewrite_redirect_url(location)
        upload_headers = {
            "Content-Type": "application/octet-stream",
            "ngrok-skip-browser-warning": "true",
        }
        second = requests.put(target, data=content, headers=upload_headers, timeout=300)
        second.raise_for_status()

    def write_text(self, hdfs_path: str, text: str, overwrite: bool = True) -> None:
        self.write_bytes(hdfs_path, text.encode("utf-8"), overwrite=overwrite)

    def _rewrite_redirect_url(self, redirect_url: str) -> str:
        redirect = urlparse(redirect_url)
        base = urlparse(self.url)
        return urlunparse(redirect._replace(scheme=base.scheme, netloc=base.netloc))


hdfs = HdfsBridge(NGROK_URL, user=HDFS_USER)
written_artifacts: list[dict[str, object]] = []


def hdfs_upload_v2(local_path: str | Path, hdfs_path: str, overwrite: bool = True) -> int:
    local_path = Path(local_path)
    parent = "/" + "/".join(hdfs_path.strip("/").split("/")[:-1])
    hdfs.mkdirs(parent)
    content = local_path.read_bytes()
    hdfs.write_bytes(hdfs_path, content, overwrite=overwrite)
    size = len(content)
    print(f"UPLOAD {local_path} -> {hdfs_path} ({size:,} bytes) user={hdfs.user}")
    return size


def write_parquet_and_upload(df: pd.DataFrame, local_path: Path, hdfs_path: str) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(local_path, index=False)
    size = hdfs_upload_v2(local_path, hdfs_path, overwrite=True)
    written_artifacts.append({"hdfs_path": hdfs_path, "bytes": size})


for key, result in results.items():
    commodity, target = key
    pred_df = result["predictions"].copy()
    pred_hdfs = f"/supply-chain/analytics/predictions_v2/{commodity}_{target}/{commodity}_{target}.parquet"
    pred_local = LOCAL_ARTIFACT_ROOT / "predictions_v2" / f"{commodity}_{target}.parquet"
    write_parquet_and_upload(pred_df, pred_local, pred_hdfs)

    shap_info = shap_results[key]
    x_test = shap_info["X_test"].reset_index(drop=True)
    values = shap_info["values"]
    dates = shap_info["dates_test"].reset_index(drop=True)
    shap_frames = []
    for feature_idx, feature_name in enumerate(FEATURE_COLS):
        shap_frames.append(
            pd.DataFrame(
                {
                    "event_date": dates,
                    "feature_name": feature_name,
                    "shap_value": values[:, feature_idx],
                    "feature_value": x_test[feature_name].to_numpy(),
                }
            )
        )
    shap_local_df = pd.concat(shap_frames, ignore_index=True)
    shap_hdfs = f"/supply-chain/analytics/shap_local_v2/{commodity}_{target}/{commodity}_{target}.parquet"
    shap_local = LOCAL_ARTIFACT_ROOT / "shap_local_v2" / f"{commodity}_{target}.parquet"
    write_parquet_and_upload(shap_local_df, shap_local, shap_hdfs)


metrics_out = metrics_df[
    [
        "commodity",
        "target",
        "test_r2",
        "test_mae",
        "naive_mae",
        "improvement_pct",
        "spearman",
        "decision_flag",
    ]
].copy()
write_parquet_and_upload(
    metrics_out,
    LOCAL_ARTIFACT_ROOT / "regression_metrics" / "metrics.parquet",
    "/supply-chain/analytics/regression_metrics/metrics.parquet",
)

write_parquet_and_upload(
    attribution_df,
    LOCAL_ARTIFACT_ROOT / "shap_attribution_v2" / "attribution.parquet",
    "/supply-chain/analytics/shap_attribution_v2/attribution.parquet",
)

metadata = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model_decision_verdict": VERDICT,
    "average_test_r2": float(avg_test_r2),
    "models_with_improvement_gt_5pct": int(models_improving),
    "feature_contract_version": "rolling_gdelt_96_macro_vol_targets_v2",
    "feature_columns": FEATURE_COLS,
    "feature_column_count": len(FEATURE_COLS),
    "target_definitions": {
        "abs_return_5d_fwd": "abs((price[t+5] / price[t]) - 1)",
        "abs_return_20d_fwd": "abs((price[t+20] / price[t]) - 1)",
    },
    "split_logic": "Sort by event_date; first 70% train, next 15% validation, final 15% test; no shuffle.",
    "decision_criteria": {
        "average_test_r2_gt": 0.10,
        "min_models_with_improvement_pct_gt_5": 6,
    },
}
metadata_local = LOCAL_ARTIFACT_ROOT / "regression_metadata.json"
metadata_local.write_text(json.dumps(metadata, indent=2))
metadata_size = hdfs_upload_v2(
    metadata_local,
    "/supply-chain/analytics/regression_metadata.json",
    overwrite=True,
)
written_artifacts.append(
    {"hdfs_path": "/supply-chain/analytics/regression_metadata.json", "bytes": metadata_size}
)

print("Artifacts written to HDFS:")
for artifact in written_artifacts:
    print(f"{artifact['bytes']:>10,} bytes  {artifact['hdfs_path']}")
