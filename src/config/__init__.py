"""
Configuration loading for FinRisk.

Everything that could vary between environments or experiments (database
credentials, risk assumptions, ML thresholds, transaction costs, etc.) lives
in config/config.yaml or .env — never hard-coded inside business logic.
"""
from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
LOGGING_CONFIG_PATH = PROJECT_ROOT / "config" / "logging.yaml"

load_dotenv(PROJECT_ROOT / ".env")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the project's YAML configuration file.

    Raises FileNotFoundError with a clear message if the config is missing,
    instead of failing later with a confusing KeyError somewhere deep in a
    risk calculation.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Did you forget to create "
            "config/config.yaml from the template in the repo?"
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(path: Path = LOGGING_CONFIG_PATH) -> None:
    """Configure Python logging from config/logging.yaml.

    Falls back to a sane default if the file is missing, so the app never
    crashes purely because logging config is absent.
    """
    if path.exists():
        with open(path, "r") as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )


class MySQLSettings:
    """MySQL connection settings, read only from environment variables.

    Never put real credentials in code or in config.yaml — that file is
    meant to be committed to git, .env is not.
    """

    def __init__(self, database_override: str | None = None) -> None:
        self.host = os.getenv("MYSQL_HOST", "localhost")
        self.port = int(os.getenv("MYSQL_PORT", "3306"))
        self.user = os.getenv("MYSQL_USER", "root")
        self.password = os.getenv("MYSQL_PASSWORD", "")
        self.database = database_override or os.getenv("MYSQL_DATABASE", "finrisk")

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


def get_test_mysql_settings() -> "MySQLSettings":
    """MySQL settings pointed at an ISOLATED test database.

    WHY THIS EXISTS (see docs/SECOND_FORENSIC_AUDIT.md, Issue C3): the
    integration test suite (tests/test_database_integration.py) runs
    destructive `DELETE FROM` statements to get a clean slate before each
    test. Before this function existed, those tests connected via the
    plain `MYSQL` settings above -- the SAME database a developer's
    dashboard/notebooks use for real ingested data. Running `pytest
    tests/ -v` (this project's own documented first command) would
    silently wipe that real data.

    `MYSQL_TEST_DATABASE` is read from the environment; if unset, this
    derives `f"{MYSQL_DATABASE}_test"` as a sane default (e.g. `finrisk`
    -> `finrisk_test`) RATHER than falling back to the plain database
    name -- there is deliberately no code path here that can silently
    resolve to the same database the app uses.
    """
    explicit_test_db = os.getenv("MYSQL_TEST_DATABASE")
    if explicit_test_db:
        return MySQLSettings(database_override=explicit_test_db)

    base_database = os.getenv("MYSQL_DATABASE", "finrisk")
    return MySQLSettings(database_override=f"{base_database}_test")


CONFIG = load_config()
MYSQL = MySQLSettings()
