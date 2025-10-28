import click
from .version import __version__
from .config import load_yaml, OrganizerConfig, CreateTrainingConfig, TrainConfig
from .data.organizer import StationOrganizerTR
from .data.events import TrainingDataCreator
from .modeling.cnn import train_cnn

from .visualize.visualizer import (interactive_config_builder_4_create_training,
                                   interactive_training_data_visualizer)

@click.group()
@click.version_option(__version__, prog_name="pyprecip")
def main():
    """PyPrecip CLI"""

@main.command("organize-tr")
@click.option("-c", "--config", type=click.Path(exists=True), required=True, help="YAML config for organizer.")
def organizetr_cmd(config):
    """Organize raw MGM inputs into per-station JSON + normalized JSON."""
    cfg = load_yaml(config, OrganizerConfig)
    StationOrganizerTR(cfg).run()
    click.echo("Organization completed.")

@main.command("create-training")
@click.option("-c", "--config", type=click.Path(exists=True), required=True, help="YAML config for create-training.")
def create_training_cmd(config):
    """Create event-based training datasets."""
    cfg = load_yaml(config, CreateTrainingConfig)
    TrainingDataCreator(cfg).run()
    click.echo("Training data created.")

@main.command("train-cum-evnt")
@click.option("-c", "--config", type=click.Path(exists=True), required=True, help="YAML config for training.")
def train_cmd(config):
    """Train CNN classification model for 1-hour nowcast."""
    cfg = load_yaml(config, TrainConfig)
    model_path, metrics = train_cnn(cfg)
    click.echo(f"Model saved to: {model_path}")
    click.echo("Metrics:")
    for k, v in metrics.items():
        if k != "ConfusionMatrix":
            click.echo(f"- {k}: {v:.3f}")

# ----------------------------------------------------------------------
# Interactive Helpers (visualizer and configuration yaml file creator)
@main.command("config-builder-4-create-training")
@click.option("-i", "--input", type=click.Path(exists=True), required=True, help="Path to organized input directory.")
def interactive_config_builder_1_cmd(input):
    """Interactive builder and  data visualizer for create-training cli YAML configuration file."""
    interactive_config_builder_4_create_training(input)

@main.command("training-data-visualizer")
@click.option("-c", "--config", type=click.Path(exists=True), required=True, help="YAML config for training data visualizer.")
def interactive_training_data_visualizer_cmd(config):
    """Interactive data visualizer for training data."""
    cfg = load_yaml(config, TrainConfig)
    interactive_training_data_visualizer(cfg)



if __name__ == "__main__":
    main()
