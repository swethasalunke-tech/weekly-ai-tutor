import json
from pathlib import Path

import pytest

from weekly_ai_tutor.ingest import load_transcript_file, load_transcripts_from_dir

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_transcript_file_valid():
    t = load_transcript_file(FIXTURES / "sample_transcript.json")
    assert t.session_id == "sess-001"
    assert t.title == "Fix race condition in worker queue"
    assert len(t.messages) == 3


def test_load_transcript_file_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_transcript_file(FIXTURES / "does_not_exist.json")


def test_load_transcript_file_bad_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_transcript_file(bad)


def test_load_transcript_file_invalid_schema_includes_path_in_error(tmp_path):
    bad = tmp_path / "invalid.json"
    bad.write_text(json.dumps({"title": "no session id", "messages": []}))
    with pytest.raises(Exception) as exc_info:
        load_transcript_file(bad)
    assert "invalid.json" in str(exc_info.value)


def test_load_transcripts_from_dir_skips_bad_files(tmp_path, capsys):
    good = {
        "session_id": "good-1",
        "title": "ok",
        "messages": [{"role": "user", "content": "hi"}],
    }
    (tmp_path / "a_good.json").write_text(json.dumps(good))
    (tmp_path / "b_bad.json").write_text("{broken")
    (tmp_path / "c_invalid_schema.json").write_text(json.dumps({"title": "no id"}))

    transcripts = load_transcripts_from_dir(tmp_path)

    assert len(transcripts) == 1
    assert transcripts[0].session_id == "good-1"
    err = capsys.readouterr().err
    assert "b_bad.json" in err
    assert "c_invalid_schema.json" in err


def test_load_transcripts_from_dir_not_a_directory_raises(tmp_path):
    f = tmp_path / "file.json"
    f.write_text("{}")
    with pytest.raises(NotADirectoryError):
        load_transcripts_from_dir(f)


def test_load_transcripts_from_dir_empty_dir_returns_empty_list(tmp_path):
    assert load_transcripts_from_dir(tmp_path) == []
