"""
Script that trains the LPD model with the UCM3-LP dataset.

Usage:
    python train_LPD_model.py [--model-path path] [--output-path out] [--epochs num].

Input:
    path (str, optional): Trained model path (.pth) for loading.
    out (str, optional): Path to save the trained model (.pth).
    num (int, optional): Number of epochs during the training.

Output:
    If out argument was passed, the model is saved in the following format:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item(),
    }, out)
"""

import argparse
import torch
from torch.utils.data import DataLoader
from LPD_dataset import CarPlateDetectionDataset
from torchvision.ops import MultiScaleRoIAlign, box_iou
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator




def collate_fn(batch: list) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Divides the batch into two, one of images and another of labels.
    :param batch: The batch made of tuples of images and labels.
    :return: Two separate batches, one for each image and one for each label.
    """
    return tuple(zip(*batch))

def main():
    parser = argparse.ArgumentParser(description='LPD_model training')
    parser.add_argument('--model-path', type=str, default=None, help='Trained model path (.pth) for loading')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs during the training')
    parser.add_argument('--output-path', type=str, default='models/OCR_model.pth', help='Path to save the trained model (.pth)')
    args = parser.parse_args()
    device = torch.device('cpu')

    # Model creation
    # Use pretrained resnet as backbone with FPN to treat each return layer as an output
    resnet = resnet50(weights='DEFAULT')
    return_layers = {
        'layer1': '0',
        'layer2': '1',
        'layer3': '2',
        'layer4': '3'
    }
    in_channels_list = [256, 512, 1024, 2048]
    backbone = BackboneWithFPN(resnet, return_layers=return_layers, in_channels_list=in_channels_list, out_channels=256)

    # RPN to adjust each anchor
    anchor_generator = AnchorGenerator(
        sizes=((32,), (60,), (90,), (128,), (256,)),
        aspect_ratios=((0.33, 0.5, 1.0),) * 5
    )

    # ROI to extract feature maps out of each anchor proposal
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Load
    if args.model_path:
        print(f'Loading model from {args.model_path}')
        checkpoint = torch.load(args.model_path, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        history = checkpoint['losses']
        print(f'Checkpoint loaded. Last epoch: {last_epoch}, with history:\n {history}')
    else:
        history = {
            'loss': [],
            'objectness': [],
            'rpn_box': [],
            'classifier': [],
            'box': []
        }
        last_epoch = 0

    train_dataset = CarPlateDetectionDataset(r'C:/Repositorio/license_plate-recognizer/dataset/', 'train')
    train_loader = DataLoader(train_dataset, batch_size=3, collate_fn=collate_fn)
    test_dataset = CarPlateDetectionDataset(r'C:/Repositorio/license_plate-recognizer/dataset/', 'test')
    test_loader = DataLoader(test_dataset, batch_size=3, collate_fn=collate_fn)

    # Train
    model.train()
    num_epochs = args.epochs + last_epoch
    num_batches = len(train_loader)
    for epoch in range(last_epoch + 1, num_epochs + 1):
        epoch_losses = {
            'loss': 0.0,
            'objectness': 0.0,
            'rpn_box': 0.0,
            'classifier': 0.0,
            'box': 0.0
        }
        print(f'Epoch {epoch}/{num_epochs}')
        for images, targets in train_loader:
            # Load images and targets in device
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # Backward
            losses.backward()
            optimizer.step()
            # Save losses
            epoch_losses['loss'] += losses.item()
            epoch_losses['objectness'] += loss_dict['loss_objectness'].item()
            epoch_losses['rpn_box'] += loss_dict['loss_rpn_box_reg'].item()
            epoch_losses['classifier'] += loss_dict['loss_classifier'].item()
            epoch_losses['box'] += loss_dict['loss_box_reg'].item()

        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            history[key].append(epoch_losses[key])

        print(
            f'Epoch {epoch}:\n'
            f"loss={epoch_losses['loss']:.4f}\n"
            f"objectness={epoch_losses['objectness']:.4f}\n"
            f"rpn_box={epoch_losses['rpn_box']:.4f}\n"
            f"classifier={epoch_losses['classifier']:.4f}\n"
            f"box={epoch_losses['box']:.4f}"
        )

    # Test
    model.eval()
    score_threshold = 0.8
    iou_threshold = 0.75
    total_gt = 0
    total_detected = 0
    images_all_detected = 0

    with torch.no_grad():
        for images, targets in test_loader:
            # Load images in device
            images = [image.to(device) for image in images]
            # Forward
            predictions = model(images)
            # Score
            for prediction, target in zip(predictions, targets):
                gt_boxes = target['boxes'].to(device)
                # Keep the predictions with score over a threshold
                keep = ((prediction['scores'] >= score_threshold) & (prediction['labels'] == 1))
                pred_boxes = prediction['boxes'][keep]

                # Evaluation
                num_gt = len(gt_boxes)
                total_gt += num_gt
                # If no GT
                if num_gt == 0:
                    if len(pred_boxes) == 0:
                        images_all_detected += 1
                    continue
                # If there are GT but no predictions
                if len(pred_boxes) == 0:
                    continue
                # If there are GT and predictions
                # Check IoU
                ious = box_iou(pred_boxes, gt_boxes)

                # Pair each GT with its prediction
                pairs = [(ious[pred, gt].item(), pred, gt)
                         for pred in range(ious.shape[0])
                         for gt in range(ious.shape[1])
                         if ious[pred, gt] >= iou_threshold
                         ]
                pairs.sort(reverse=True)
                matched_preds = set()
                matched_gts = set()
                for iou, pred, gt in pairs:
                    # Each prediction and GT can only be used once
                    if pred not in matched_preds and gt not in matched_gts:
                        matched_preds.add(pred)
                        matched_gts.add(gt)

                # Percentage detection
                detected = len(matched_gts)
                total_detected += detected
                if detected == num_gt:
                    images_all_detected += 1

    recall = 100 * total_detected / total_gt
    all_detected_rate = 100 * images_all_detected / len(test_dataset)

    print(f'Detection recall: {recall:.2f}%')
    print(f'Imagen in which all GT were detected: {all_detected_rate:.2f}%')
    print(f'Detected GT: {total_detected}/{total_gt}')

    # Save
    if args.output_path:
        torch.save({
            'epoch': num_epochs + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': history
        }, args.output_path)

if __name__ == '__main__':
    main()