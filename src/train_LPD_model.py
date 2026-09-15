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
from torchvision.ops import box_iou, nms
from torchvision.models.detection.image_list import ImageList
from LPD_dataset import CarPlateTrainDataset, CarPlateTestDataset

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import resnet50


def collate_fn(batch: list) -> tuple[torch.Tensor, list[torch.Tensor]]:
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = list(labels)
    return imgs, labels

def main():

    parser = argparse.ArgumentParser(description='LPD_model training')
    parser.add_argument('--model-path', type=str, default=None, help='Trained model path (.pth) for loading')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs during the training')
    parser.add_argument('--output-path', type=str, default='models/OCR_model.pth', help='Path to save the trained model (.pth)')
    args = parser.parse_args()

    # Model loading
    device = torch.device('cpu')

    # Use pretrained resnet as backbone with FPN to treat each return layer as an output
    resnet = resnet50(weights="DEFAULT")
    return_layers = {
        "layer1": "0",
        "layer2": "1",
        "layer3": "2",
        "layer4": '3'
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
        featmap_names=["0", "1", "2", "3"],
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

    if args.model_path:
        print(f'Loading model from {args.model_path}')
        checkpoint = torch.load(args.model_path, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Checkpoint loaded. Last epoch: {last_epoch}, with loss: {loss}')
    else:
        last_epoch = 0

    #train_dataset = CarPlateTrainDataset('dataset/', compact=True)
    #test_dataset = CarPlateTestDataset('dataset/')
    train_dataset = CarPlateTrainDataset(r'C:\Repositorio\license_plate-recognizer\dataset\\', compact=False)
    train_loader = DataLoader(train_dataset, batch_size=10, collate_fn=collate_fn)
    #test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Train
    model.train()
    num_epochs = args.epochs + last_epoch
    last_epoch = 0
    num_epochs = 10
    for epoch in range(last_epoch+1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}', end=' ')
        epoch_loss = 0.0
        a = next(iter(train_loader))
        images, targets = a
        for target in targets:
            boxes = target["boxes"]

            print("xmin:", boxes[:, 0].min().item())
            print("ymin:", boxes[:, 1].min().item())
            print("xmax:", boxes[:, 2].max().item())
            print("ymax:", boxes[:, 3].max().item())

            print("¿NaN?:", torch.isnan(boxes).any().item())
            print("¿Inf?:", torch.isinf(boxes).any().item())
            print("¿xmin < xmax?:", (boxes[:, 0] < boxes[:, 2]).all().item())
            print("¿ymin < ymax?:", (boxes[:, 1] < boxes[:, 3]).all().item())
        #for images, targets in next(iter(train_loader)):
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()

        print(f'loss: {epoch_loss:.4f}')
    exit()

    # Test
    model.eval()
    total_gt = 0
    correct_detections = 0
    with torch.no_grad():
        for images, targets in test_loader:

            cls_preds, reg_preds, anchors = model(images)
            batch_size = images.size(0)

            for i in range(batch_size):

                gt_boxes = targets[i]
                total_gt += len(gt_boxes)

                scores = cls_preds[i].softmax(dim=1)[:, 1]
                mask = scores > 0.6

                if mask.sum() == 0:
                    continue

                # Decodificar cajas
                print('anchors y cajas predichas antes de codificar')
                print(anchors[i][mask])
                print(reg_preds[i][mask])
                pred_boxes = box_coder.decode(anchors[i][mask], [reg_preds[i][mask]]).reshape(1, -1, 4).squeeze(0)
                pred_scores = scores[mask]
                print('cajas predichas y probabilidades despues de codificar')
                print(pred_boxes)
                print(pred_scores)

                # NMS
                keep = nms(pred_boxes, pred_scores, iou_threshold=0.5)
                pred_boxes = pred_boxes[keep]
                print('cajas predichas tras nms')
                print(pred_boxes)

                # Comparar cada GT con las predicciones
                if len(pred_boxes) > 0:
                    iou = box_iou(gt_boxes, pred_boxes)
                    max_iou_per_gt, _ = iou.max(dim=1)

                    # Contar GT detectadas con IoU suficiente
                    correct_detections += (max_iou_per_gt >= 0.5).sum().item()
            break

    recall = correct_detections / total_gt if total_gt > 0 else 0
    print("Recall:", recall)
    print(f"Accuracy of detection = {recall * 100:.2f}%")

    if args.output_path:
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss.item()
        }, args.output_path)


if __name__ == "__main__":
    main()
