import click
import yaml


@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run_pipeline(config) -> None:
    with open(config, 'r') as file:
        config_data = yaml.safe_load(file)

        train_images_dir = config_data.get('images_train')
        test_images_dir = config_data.get('images_test')
        train_masks_dir = config_data.get('masks_train')
        test_masks_dir = config_data.get('masks_test')


if __name__ == "__main__":
    run_pipeline()
