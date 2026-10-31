"""Static checks of the Helm chart that run without a helm binary (CI also runs helm lint)."""

from pathlib import Path

import yaml

CHART = Path("charts/llm-inference")


def test_values_files_parse_and_engines_are_valid():
    for f in CHART.glob("values*.yaml"):
        values = yaml.safe_load(f.read_text())
        assert values["engine"] in ("nim", "vllm"), f


def test_templates_reference_existing_values():
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    text = "".join(p.read_text() for p in (CHART / "templates").glob("*"))
    for key in ("autoscaling", "monitoring", "modelCache", "images", "model", "resources"):
        assert f".Values.{key}" in text
        assert key in values


def test_dashboard_is_valid_json():
    import json

    dash = json.loads(Path("observability/grafana-dashboard.json").read_text())
    assert len(dash["panels"]) >= 6
