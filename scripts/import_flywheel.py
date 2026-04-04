"""
Import training data from training-flywheel's SQLite database.

Usage:
    python scripts/import_flywheel.py \
        --flywheel-db ~/.config/training-flywheel/flywheel.db \
        --output data/flywheel_export.jsonl \
        --min-quality 0.6 \
        --format chatml
"""

import argparse
import json
import os
import sqlite3
import sys


def export_flywheel(db_path, output_path, min_quality=0.6, fmt="chatml"):
    """Read flywheel DB and export high-quality sessions as training JSONL."""
    if not os.path.exists(db_path):
        print(f"Error: flywheel database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Discover table structure
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row["name"] for row in cursor.fetchall()]

    # Try common flywheel table names
    sessions_table = None
    for candidate in ["sessions", "conversations", "examples", "training_data"]:
        if candidate in tables:
            sessions_table = candidate
            break

    if sessions_table is None:
        print(f"Available tables: {tables}")
        print("Error: could not find a sessions/conversations table")
        conn.close()
        sys.exit(1)

    # Get column names
    cursor = conn.execute(f"PRAGMA table_info({sessions_table})")
    columns = [row["name"] for row in cursor.fetchall()]

    # Build query based on available columns
    quality_col = None
    for candidate in ["quality", "score", "rating", "quality_score"]:
        if candidate in columns:
            quality_col = candidate
            break

    messages_col = None
    for candidate in ["messages", "conversation", "data", "content"]:
        if candidate in columns:
            messages_col = candidate
            break

    if messages_col is None:
        print(f"Available columns: {columns}")
        print("Error: could not find a messages/conversation column")
        conn.close()
        sys.exit(1)

    query = f"SELECT * FROM {sessions_table}"
    if quality_col:
        query += f" WHERE {quality_col} >= ?"
        cursor = conn.execute(query, (min_quality,))
    else:
        print(f"Warning: no quality column found, exporting all rows")
        cursor = conn.execute(query)

    rows = cursor.fetchall()
    conn.close()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    exported = 0
    with open(output_path, "w") as f:
        for row in rows:
            raw = row[messages_col]
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue

            # Normalize to {"messages": [...]} format
            if isinstance(data, list):
                messages = data
            elif isinstance(data, dict) and "messages" in data:
                messages = data["messages"]
            else:
                continue

            if not messages or len(messages) < 2:
                continue

            # Ensure all messages have role and content
            valid = True
            for msg in messages:
                if not isinstance(msg, dict) or "role" not in msg:
                    valid = False
                    break
            if not valid:
                continue

            record = {"messages": messages}
            f.write(json.dumps(record) + "\n")
            exported += 1

    print(f"Exported {exported} conversations to {output_path}")
    print(f"Format: {fmt}")
    if exported > 0:
        print(f"Use with: python -m scripts.sft --data {output_path} --format {fmt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import training data from training-flywheel")
    parser.add_argument("--flywheel-db", type=str,
                        default=os.path.expanduser("~/.config/training-flywheel/flywheel.db"),
                        help="Path to flywheel SQLite database")
    parser.add_argument("--output", type=str, default="data/flywheel_export.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--min-quality", type=float, default=0.6,
                        help="Minimum quality score to include")
    parser.add_argument("--format", type=str, default="chatml", choices=["chatml", "smoltalk"],
                        help="Output format")
    args = parser.parse_args()
    export_flywheel(args.flywheel_db, args.output, args.min_quality, args.format)
