"""Tests for human-readable Settings metadata."""

from __future__ import annotations

from typing import Any

from config import ENV_SCHEMA
from frontend.ui_dialogs import config_dialog
from frontend.ui_dialogs.config_dialog import (
    _COLOR_KEYS,
    _FIELD_LABELS,
    _SECTION_ORDER,
    _SUBGROUPS,
    _default_form_values,
)


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


def test_subgroups_cover_each_schema_key_once() -> None:
    subgroup_keys = [
        key for groups in _SUBGROUPS.values() for _subgroup_title, keys in groups for key in keys
    ]
    schema_keys = [item["key"] for item in ENV_SCHEMA]

    assert len(subgroup_keys) == len(set(subgroup_keys))
    assert set(subgroup_keys) == set(schema_keys)


def test_each_sections_subgroups_match_its_top_level_keys() -> None:
    for section_id, _title, section_keys in _SECTION_ORDER:
        subgroup_keys = [key for _title, keys in _SUBGROUPS[section_id] for key in keys]
        assert subgroup_keys == section_keys


def test_every_setting_has_a_human_readable_label() -> None:
    schema_keys = {item["key"] for item in ENV_SCHEMA}

    assert set(_FIELD_LABELS) == schema_keys
    assert all(label.strip() and label != key for key, label in _FIELD_LABELS.items())


def test_default_form_values_cover_schema_with_expected_types() -> None:
    defaults = _default_form_values()

    assert set(defaults) == {item["key"] for item in ENV_SCHEMA}
    for item in ENV_SCHEMA:
        value = defaults[item["key"]]
        if item["cast_type"] is bool:
            assert isinstance(value, bool)
            assert value is item["default"]
        else:
            assert isinstance(value, str)
            assert value == str(item["default"])


def test_restore_defaults_only_updates_in_memory_form(monkeypatch: Any) -> None:
    class FakeVariable:
        def __init__(self) -> None:
            self.value: str | bool | None = None

        def set(self, value: str | bool) -> None:
            self.value = value

    dialog = object.__new__(config_dialog.ConfigDialog)
    dialog.win = object()
    dialog._vars = {item["key"]: FakeVariable() for item in ENV_SCHEMA}
    previews_updated = False

    def mark_previews_updated() -> None:
        nonlocal previews_updated
        previews_updated = True

    dialog._update_color_swatches = mark_previews_updated
    monkeypatch.setattr(config_dialog.messagebox, "askyesno", lambda *args, **kwargs: True)

    def fail_write(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Restore Defaults must not write .env")

    monkeypatch.setattr(config_dialog, "write_env_file", fail_write)

    dialog._on_restore_defaults()

    defaults = _default_form_values()
    assert {key: var.value for key, var in dialog._vars.items()} == defaults
    assert previews_updated


def test_color_metadata_contains_only_individual_color_settings() -> None:
    assert _COLOR_KEYS == {
        "UI_BACKGROUND",
        "UI_FOREGROUND",
        "UI_BUTTON_BG",
        "UI_BUTTON_FG",
        "UI_BUTTON_FG_CANCEL",
        "UI_BUTTON_FG_ACCENT2",
        "PLOT_LINE_COLOR",
        "PLOT_MARKER_FACE_COLOR",
        "PLOT_MARKER_EDGE_COLOR",
        "PLOT_PHASE_START_COLOR",
        "PLOT_PHASE_END_COLOR",
    }
    assert _COLOR_KEYS <= {item["key"] for item in ENV_SCHEMA}
