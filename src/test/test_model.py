import torch
import torchmetrics
import torchvision.transforms as T
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def show_save_images(images, titles=None, filename=None) -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(15, 15))

    for i, img in enumerate(images):
        ax = axes[i] if n > 1 else axes
        ax.imshow(img, cmap='gray')
        if titles:
            ax.set_title(titles[i])
        ax.axis('off')

    # Show photos
    plt.show()

    # Save the figure to a file
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def test(model, test_dataloader: DataLoader, device: torch.device, model_path: str, scores_path: str, photos_dir: str) -> None:
    model.load_state_dict(torch.load(model_path))
    model.eval()
    precision_metric = torchmetrics.Precision(num_classes=1, threshold=0.5, task='binary').to(device)
    recall_metric = torchmetrics.Recall(num_classes=1, threshold=0.5, task='binary').to(device)

    num_corrects = 0
    num_pixels = 0
    dice_score = 0

    first_batch = True
    with torch.no_grad():
        for images, masks in test_dataloader:
            images = images.float().to(device)
            masks = masks.unsqueeze(1).float().to(device)
            outputs = torch.sigmoid(model(images))
            outputs = (outputs >= 0.5).float()

            # metrics
            dice_score += (2 * (outputs * masks).sum().item()) / (
                    (outputs + masks).sum().item() + 1e-8
            )
            num_corrects += (outputs == masks).sum()
            num_pixels += torch.numel(outputs)
            precision_metric.update(outputs, masks)
            recall_metric.update(outputs, masks)

            # show photos from first batch
            if first_batch:
                for i in range(min(8, len(images))):
                    x_img = T.ToPILImage()(images[i].cpu().squeeze())
                    y_img = T.ToPILImage()(masks[i].cpu().squeeze())
                    pred_img = T.ToPILImage()(outputs[i].cpu().squeeze())
                    show_save_images([x_img, y_img, pred_img], titles=["Input Image", "True Mask", "Predicted Mask"],
                                     filename=photos_dir + str(i) + '.png')
                first_batch = False

    accuracy = num_corrects / num_pixels * 100
    average_dice_score = dice_score / len(test_dataloader)
    avg_val_precision = precision_metric.compute()
    avg_val_recall = recall_metric.compute()

    with open(scores_path, "w") as file:  # save scores to txt
        file.write(f"Got {num_corrects}/{num_pixels} with acc {accuracy:.4f}%\n")
        file.write(f"Dice score: {average_dice_score:.4f}\n")
        file.write(f"Precision: {avg_val_precision:.4f}\n")
        file.write(f"Recall: {avg_val_recall:.4f}\n")
