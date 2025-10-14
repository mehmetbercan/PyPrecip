import pytest
from click.testing import CliRunner
from pathlib import Path
from pyprecip.cli import main

@pytest.mark.integration
def test_organizeTR():
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
    result = runner.invoke(main, ["organizeTR", "-c", str(config_path)])

    # --- assertions ---
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Organization completed." in result.output

    assert output_dir.exists(), f"Output directory missing: {output_dir}"
    json_files = list(output_dir.glob("station_*.json"))
    assert json_files, "No station_*.json files created"

    # if specific file exists, check it
    if expected_file.exists():
        print(f"SUCCESS: CLI succeeded and created: {expected_file}")
    else:
        print(f"WARNING: CLI ran successfully but did not find {expected_file.name}")
