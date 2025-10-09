import click
from .version import __version__
from .config import load_yaml, OrganizerConfig, CreateTrainingConfig, TrainConfig
from .data.organizer import StationOrganizerTR
from .data.events import TrainingDataCreator
from .modeling.cnn import train_cnn

@click.group()
@click.version_option(__version__, prog_name="pyprecip")
def main():
    """PyPrecip CLI"""

@main.command("organizeTR")
@click.option("-c", "--config", type=click.Path(exists=True), required=True, help="YAML config for organizer.")
def organize_cmd(config):
    """Organize raw MGM inputs into per-station JSON + normalized JSON."""
    cfg = load_yaml(config, OrganizerConfig)
    StationOrganizerTR(cfg).run()
    click.echo("Organization completed.")

@main.command("create-training")
@click.option("-c", "--config", type=click.Path(exists=True), required=True, help="YAML config for create-training.")
def create_training_cmd(config):
    """Create event-based training datasets (Pirone-like)."""
    cfg = load_yaml(config, CreateTrainingConfig)
    TrainingDataCreator(cfg).run()
    click.echo("Training data created.")

@main.command("train")
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

if __name__ == "__main__":
    main()
