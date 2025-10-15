import pytest
from click.testing import CliRunner
from pathlib import Path
from pyprecip.cli import main
import json
import pandas as pd

@pytest.mark.integration
def test_organize_tr_cmd():
    """Run `pyprecip organizeTR` using the real YAML config and verify expected output."""

    # relative to this script's root
    here = Path(__file__).resolve().parent
    config_path = Path(here / "../examples/configs/organizer_example_4TRstate.yaml")
    output_dir = Path(here / "../examples/outputs/organized")
    expected_file = output_dir / "station_18186.json"

    # --- preconditions ---
    assert config_path.exists(), f"Config not found: {config_path}"
    runner = CliRunner()

    # --- run CLI command ---
    result = runner.invoke(main, ["organize-tr", "-c", str(config_path)])

    # --- assertions ---
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Organization completed." in result.output

    assert output_dir.exists(), f"Output directory missing: {output_dir}"
    json_files = list(output_dir.glob("station_*.json"))
    assert json_files, "No station_*.json files created"

    # --- expected output checks ---
    assert expected_file.exists(), f"Expected JSON file not found: {expected_file}"

    # --- check contents of JSON file ---
    df = pd.read_json(expected_file)

    assert not df.empty, f"DataFrame loaded from {expected_file} is empty"
    assert "precip" in df.columns, "Missing column: 'precip'"
    assert df.precip.sum() == 107.2
