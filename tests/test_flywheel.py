"""Tests for flywheel import script."""

import json
import os
import sqlite3
import tempfile

from scripts.import_flywheel import export_flywheel


def test_export_flywheel_basic():
    """Test flywheel export with a simple test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        output_path = os.path.join(tmpdir, "output.jsonl")

        # Create test database
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY,
                messages TEXT,
                quality REAL
            )
        """)

        good_conv = json.dumps([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ])
        bad_conv = json.dumps([
            {"role": "user", "content": "Bad"},
            {"role": "assistant", "content": "Low quality"},
        ])

        conn.execute("INSERT INTO sessions (messages, quality) VALUES (?, ?)", (good_conv, 0.9))
        conn.execute("INSERT INTO sessions (messages, quality) VALUES (?, ?)", (bad_conv, 0.3))
        conn.commit()
        conn.close()

        export_flywheel(db_path, output_path, min_quality=0.6)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["content"] == "Hi there!"


def test_export_flywheel_chatml_format():
    """Test flywheel export with ChatML tool-calling data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        output_path = os.path.join(tmpdir, "output.jsonl")

        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY,
                messages TEXT,
                quality REAL
            )
        """)

        conv = json.dumps([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "List files"},
            {"role": "assistant", "content": "<tool_call>{\"name\": \"bash\"}</tool_call>"},
            {"role": "tool", "content": "file1.txt\nfile2.txt"},
            {"role": "assistant", "content": "Here are the files."},
        ])

        conn.execute("INSERT INTO sessions (messages, quality) VALUES (?, ?)", (conv, 0.8))
        conn.commit()
        conn.close()

        export_flywheel(db_path, output_path, min_quality=0.5, fmt="chatml")

        with open(output_path) as f:
            data = json.loads(f.readline())

        assert len(data["messages"]) == 5
        assert data["messages"][2]["role"] == "assistant"
        assert "<tool_call>" in data["messages"][2]["content"]
        assert data["messages"][3]["role"] == "tool"
