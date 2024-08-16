import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import optim, nn
from tqdm import tqdm
import segmentation_models_pytorch as smp
from src.train.earlystopping import EarlyStopping
from src.model.model import Unet


def train(train_dataloader, valid_dataloader, model_path: str, num_epochs: int, pretrained: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if pretrained:
        model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=1).to(device)
    else:
        model = Unet(in_channels=3,num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=8, path=model_path)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            images = images.float().to(device)
            masks = masks.unsqueeze(1).float().to(device)

            outputs = model(images)
            optimizer.zero_grad()

            loss = criterion(outputs, masks)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        dice_score = 0
        with torch.no_grad():
            for images, masks in tqdm(valid_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
                images = images.float().to(device)
                masks = masks.unsqueeze(1).float().to(device)

                outputs = model(images)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

                outputs = torch.sigmoid(model(images))
                outputs = (outputs >= 0.5).float()

                dice_score += (2 * (outputs * masks).sum().item()) / (
                        (outputs + masks).sum().item() + 1e-8
                )

        average_dice_score = dice_score / len(valid_dataloader)
        validation_loss = val_loss / len(valid_dataloader)
        training_loss = train_loss / len(train_dataloader)

        print(f"Train loss: {training_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Dice score: {average_dice_score:.4f}")

        early_stopping(validation_loss, model)
        if early_stopping.early_stop:
            break

        scheduler.step(validation_loss)

    return model, device
