import click
import yaml

from train.train_model import train
from test.test_model import test
from dataset_and_dataloader.dataset import datasets
from dataset_and_dataloader.split_and_copy import split_data
from dataset_and_dataloader.dataloader import dataloaders


@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run_pipeline(config) -> None:
    with open(config, 'r') as file:
        config_data = yaml.safe_load(file)

        batch_size = config_data.get('batch_size')  # load batch size
        num_epochs = config_data.get('epochs')  # load num of epochs

        model_path = config_data.get('model_path')  # path to save model
        scores_path = config_data.get('scores_path')  # patho to save scores
        pretrained = config_data.get('pretrained')

        images_dir = config_data.get('images')  # load images dir
        masks_dir = config_data.get('masks')  # load masks dir
        photos_dir = config_data.get('photos_dir')  # load photos dir

        train_masks_dir = config_data.get('train_masks')  # load dir for train masks
        test_masks_dir = config_data.get('test_masks')  # load dir for test masks
        train_images_dir = config_data.get('train_images')  # load dir for train images
        test_images_dir = config_data.get('test_images')  # load dir for test images
        valid_images_dir = config_data.get('valid_images')  # load dir for test images
        valid_masks_dir = config_data.get('valid_masks')  # load dir for test masks

        split_data(
            images_dir,
            masks_dir,
            train_masks_dir,
            test_masks_dir,
            train_images_dir,
            test_images_dir,
            valid_images_dir,
            valid_masks_dir
        )  # functionality split data and copy to new dir

        train_dataset, test_dataset, valid_dataset = datasets(train_images_dir,
                                                              test_images_dir,
                                                              train_masks_dir,
                                                              test_masks_dir,
                                                              valid_images_dir,
                                                              valid_masks_dir)

        train_dataloader, test_dataloader, valid_dataloader = dataloaders(train_dataset,
                                                                          test_dataset,
                                                                          valid_dataset,
                                                                          batch_size)

        model, device = train(train_dataloader, valid_dataloader, model_path, num_epochs, pretrained)
        test(model, test_dataloader, device, model_path, scores_path,photos_dir)


if __name__ == "__main__":
    run_pipeline()
