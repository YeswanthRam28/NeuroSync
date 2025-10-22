# prepare_ml_data.py
import os
import pandas as pd

# locate input file relative to this script
input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "focus_log.csv"))
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found: {input_path}")

# expected header columns in the raw input
expected_cols = ["timestamp", "focus_percent", "blink_per_min", "gaze_x", "gaze_y", "yaw", "pitch"]

# read first two lines to detect header / data column-count mismatch
with open(input_path, "r", newline="") as f:
    lines = f.readlines()

if len(lines) < 2:
    raise ValueError("Input CSV must contain a header and at least one data row")

raw_header = lines[0].rstrip("\n").split(",")
first_data = lines[1].rstrip("\n").split(",")

if len(first_data) > len(raw_header):
    extra = len(first_data) - len(raw_header)
    # build column names: use header names + extra names (prefer 'activity' if single extra)
    if extra == 1:
        names = raw_header + ["activity"]
    else:
        names = raw_header + [f"extra_{i}" for i in range(extra)]
    # read the file skipping the header row (header row already read), assign names to data columns
    df = pd.read_csv(input_path, header=None, names=names, skiprows=1)
    # keep the header semantics if header matched expected names partly
    print(f"⚠️ Header/data column count mismatch detected — added {extra} extra column(s): {names[-extra:]}")
else:
    # normal case: header matches data columns count
    df = pd.read_csv(input_path, header=0)

# sanitize column names that pandas may have created (e.g., 'Unnamed: ...')
df.columns = [c.strip() for c in df.columns]

# Ensure required columns exist, add zeros if missing
for col in expected_cols:
    if col not in df.columns:
        df[col] = 0

# Convert focus_percent and timestamp to numeric (coerce bad values)
df["focus_percent"] = pd.to_numeric(df["focus_percent"], errors="coerce").fillna(0)
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

# If timestamps were parsed wrongly (e.g. NaN) attempt a recovery check:
# if timestamp column is mostly small (<=100) and focus_percent contains epoch-like values,
# it likely means the file already was shifted — do not auto-fix here, just warn.
if df["timestamp"].isna().mean() > 0.5:
    print("⚠️ Many timestamp values are non-numeric after parsing. Verify input CSV header/rows.")
elif df["timestamp"].max() is not None and df["timestamp"].max() < 1e6:
    # timestamps that are not epoch-like are suspicious (epoch seconds ~1e9)
    print("⚠️ Parsed timestamp values look small (not epoch). Confirm the first column is the epoch timestamp.")

# Initialize drift_count (set to 0 for now, can compute later)
df["drift_count"] = 0

# --- Step 1: Label Cognitive State ---
def label_cognitive_state(fp):
    if fp >= 75:
        return "Fresh"
    elif fp >= 50:
        return "Mild Fatigue"
    else:
        return "Fatigued"

df["cognitive_state"] = df["focus_percent"].apply(label_cognitive_state)

# --- Step 2: Save cleaned data for ML ---
# enforce output column order to avoid accidental shifts
out_cols = expected_cols.copy()
# preserve activity/extra trailing columns if present
for c in ["activity"] + [col for col in df.columns if col.startswith("extra_")]:
    if c in df.columns and c not in out_cols:
        out_cols.append(c)
out_cols += ["drift_count", "cognitive_state"]

# write out only the columns we want (missing ones already added above)
out_path = os.path.join(os.path.dirname(__file__), "ml_ready_data.csv")
df.to_csv(out_path, columns=out_cols, index=False)
print(f"✅ Data prepared for ML. Saved as {out_path}")
