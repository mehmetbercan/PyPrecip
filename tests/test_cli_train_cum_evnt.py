import pytest
from click.testing import CliRunner
from pathlib import Path
from pyprecip.cli import main
import joblib

@pytest.mark.integration
def test_train_cum_evnt_cmd():
    """Run `pyprecip train-cum-evnt` using the real YAML config and verify expected output."""

    # relative to this script's root
    here = Path(__file__).resolve().parent
    config_path = Path(here / "../examples/configs/train_cum_evnt_example.yaml")
    output_dir = Path(here / "../examples/outputs/models/cnn/v0")
    expected_file = output_dir / "NowcastMdl_st18186_timestp1.keras"

    output_dir2 = Path(output_dir, "histories")
    expected_file2 = output_dir2 / "NowcastMdl_st18186_timestp1.pckl"

    # --- preconditions ---
    assert config_path.exists(), f"Config not found: {config_path}"
    runner = CliRunner()

    # --- run CLI command ---
    result = runner.invoke(main, ["train-cum-evnt", "-c", str(config_path)])

    # --- assertions ---
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Metrics:" in result.output

    # --- expected output checks ---
    assert output_dir.exists(), f"Output directory missing: {output_dir}"
    assert expected_file.exists(), f"Expected Keras file not found: {expected_file}"
    assert expected_file2.exists(), f"Expected Pickle file not found: {expected_file2}"


