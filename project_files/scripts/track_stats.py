"""Print track quality stats for one or more trajectory parquets side by side.

Run: uv run --no-sync python scripts/track_stats.py a.parquet b.parquet
"""

import sys

import pandas as pd

from trafficflow.stats import track_stats

tables = {p: track_stats(pd.read_parquet(p)) for p in sys.argv[1:]}
df = pd.DataFrame(tables)
df.columns = [c.split("/")[-1] for c in df.columns]
print(df.to_string())
