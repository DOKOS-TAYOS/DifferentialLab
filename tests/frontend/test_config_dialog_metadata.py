"""Tests for human-readable Settings metadata."""

from __future__ import annotations

from config import ENV_SCHEMA
from frontend.ui_dialogs.config_dialog import _FIELD_LABELS, _SECTION_ORDER


def test_settings_sections_cover_each_schema_key_once() -> None:
    section_keys = [key for _section_id, _title, keys in _SECTION_ORDER for key in keys]
    schema_keys = [item["key"] for item in ENV_SCHEMA]

    assert len(section_keys) == len(set(section_keys))
    assert set(section_keys) == set(schema_keys)


def test_settings_sections_use_the_v1_user_facing_groups() -> None:
    assert [title for _section_id, title, _keys in _SECTION_ORDER] == [
        "Appearance",
        "Plots",
        "Solver Defaults",
        "Advanced",
    ]


def test_every_setting_has_a_human_readable_label() -> None:
    schema_keys = {item["key"] for item in ENV_SCHEMA}

    assert set(_FIELD_LABELS) == schema_keys
    assert all(label.strip() and label != key for key, label in _FIELD_LABELS.items())
