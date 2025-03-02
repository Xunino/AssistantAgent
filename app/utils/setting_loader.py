# Description: This file is used to load the settings from the settings.yaml file
# and merge them with the environment variables. The settings are loaded based on the
# active profiles and merged together. The settings are then returned as a dictionary.
import os
import sys
import logging
import functools
from typing import Any
from pathlib import Path
from collections.abc import Iterable
from pydantic.v1.utils import deep_update, unique_list

# Custom import
from utils.yaml import load_yaml_with_envvars

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

_settings_folder = os.environ.get("PGPT_SETTINGS_FOLDER", PROJECT_ROOT_PATH)

# if running in unittest, use the test profile
_test_profile = ["test"] if "tests.fixtures" in sys.modules else []

active_profiles: list[str] = unique_list(
    ["default"]
    + [
        item.strip()
        for item in os.environ.get("PGPT_PROFILES", "azopenai").split(",")
        if item.strip()
    ]
    + _test_profile
)


def merge_settings(settings: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return functools.reduce(deep_update, settings, {})


def load_settings_from_profile(profile: str) -> dict[str, Any]:
    if profile == "default":
        profile_file_name = "settings.yaml"
    else:
        profile_file_name = f"settings-{profile}.yaml"

    path = Path(_settings_folder) / profile_file_name
    with Path(path).open("r") as f:
        config = load_yaml_with_envvars(f)
    if not isinstance(config, dict):
        raise TypeError(f"Config file has no top-level mapping: {path}")
    return config


def load_active_settings() -> dict[str, Any]:
    """Load active profiles and merge them."""
    logger.info("Starting application with profiles=%s", active_profiles)
    loaded_profiles = [
        load_settings_from_profile(profile) for profile in active_profiles
    ]
    merged: dict[str, Any] = merge_settings(loaded_profiles)
    return merged
