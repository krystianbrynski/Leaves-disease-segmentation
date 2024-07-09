from torch.utils.data import DataLoader, Dataset


def dataloaders(train_dataset: Dataset, test_dataset: Dataset, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_dataloader, test_dataloader
