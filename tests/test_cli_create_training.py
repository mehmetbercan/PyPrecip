import pytest
from click.testing import CliRunner
from pathlib import Path
from pyprecip.cli import main
import joblib

@pytest.mark.integration
def test_create_training_cmd():
    """Run `pyprecip create-training` using the real YAML config and verify expected output."""

    # relative to this script's root
    here = Path(__file__).resolve().parent
    config_path = Path(here / "../examples/configs/create_training_example.yaml")
    output_dir = Path(here / "../examples/outputs/training_inputs/CumEvnt_Nstn2_pcp-instm")
    expected_file = output_dir / "18186_training_data.joblib"

    # --- preconditions ---
    assert config_path.exists(), f"Config not found: {config_path}"
    runner = CliRunner()

    # --- run CLI command ---
    result = runner.invoke(main, ["create-training", "-c", str(config_path)])

    # --- assertions ---
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Training data created." in result.output

    assert output_dir.exists(), f"Output directory missing: {output_dir}"
    joblib_files = list(output_dir.glob("*_training_data.joblib"))
    assert joblib_files, "No *_training_data.joblib files created"

    # --- expected output checks ---
    assert expected_file.exists(), f"Expected JSON file not found: {expected_file}"

    # --- check contents of joblib file ---
    df = joblib.load(expected_file)

    assert not df.empty, f"DataFrame loaded from {expected_file} is empty"
    assert "cum_pcp" in df.columns, "Missing column: 'cum_pcp'"
    assert round(df.cum_pcp.sum()) == 311.0
