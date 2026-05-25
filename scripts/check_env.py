"""Validate inference-service runtime environment before startup."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

SERVICE_ROOT = Path(__file__).resolve().parent.parent

PLACEHOLDER_VALUES = {
    "INTERNAL_SECRET": {"", "replace-with-a-shared-random-secret"},
}


@dataclass(slots=True)
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_dotenv_file() -> None:
    if load_dotenv is None:
        return
    env_file = SERVICE_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


def _get_env(environ: Mapping[str, str], key: str) -> str:
    return environ.get(key, "").strip()


def validate_environment(environ: Mapping[str, str] | None = None) -> ValidationResult:
    env = environ or os.environ
    result = ValidationResult()

    internal = _get_env(env, "INTERNAL_SECRET")
    if internal in PLACEHOLDER_VALUES["INTERNAL_SECRET"]:
        result.errors.append("Missing or placeholder value for INTERNAL_SECRET")

    inference_secret = _get_env(env, "INFERENCE_SERVICE_SECRET")
    allow_insecure = _get_env(env, "INFERENCE_ALLOW_INSECURE_DEV")
    if not inference_secret:
        if allow_insecure == "1":
            result.warnings.append(
                "INFERENCE_ALLOW_INSECURE_DEV=1 — running without auth"
            )
        else:
            result.errors.append(
                "INFERENCE_SERVICE_SECRET is required "
                "(set INFERENCE_ALLOW_INSECURE_DEV=1 to bypass for local development)"
            )

    return result


def format_validation_result(result: ValidationResult) -> str:
    lines: list[str] = []
    if result.errors:
        lines.append("Environment validation failed:")
        lines.extend(f"- {item}" for item in result.errors)
    if result.warnings:
        if lines:
            lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in result.warnings)
    if not lines:
        lines.append("Environment validation passed")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate inference-service environment")
    parser.add_argument("--no-dotenv", action="store_true", help="Skip loading .env file")
    args = parser.parse_args()

    if not args.no_dotenv:
        load_dotenv_file()

    result = validate_environment()
    print(format_validation_result(result))
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
