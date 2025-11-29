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
from torch import nn
from torch.functional import F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, nms
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection._utils import Matcher, BoxCoder
from LPD_dataset import CarPlateTrainDataset, CarPlateTestDataset


def detection_loss_3v1(cls_preds: torch.Tensor, reg_preds: torch.Tensor, anchors: list[torch.Tensor], gt_boxes: list[torch.Tensor], matcher: Matcher, box_coder: BoxCoder, neg_pos_ratio: int = 3):
    batch_size = cls_preds.size(0)
    cls_loss = 0.0
    reg_loss = 0.0

    for i in range(batch_size):
        # Match anchors with GT
        iou_matrix = box_iou(gt_boxes[i], anchors[i])
        matched_idxs = matcher(iou_matrix)
        matched_mask = matched_idxs >= 0

        # Classification targets: 1 for positive anchors, 0 for negatives
        labels = torch.zeros(cls_preds[i].size(0), dtype=torch.long, device=cls_preds.device)
        labels[matched_mask] = 1

        max_iou_per_anchor, _ = iou_matrix.max(dim=0)
        print("Max IoU anchors:", (max_iou_per_anchor > 0.5).sum().item())
        print("Max IoU > 0.3:", (max_iou_per_anchor > 0.3).sum().item())
        print("Max IoU (max):", max_iou_per_anchor.max().item())
        aw = anchors[i][:, 2] - anchors[i][:, 0]
        ah = anchors[i][:, 3] - anchors[i][:, 1]
        print(anchors[i][20:40])
        print("Anchor width mean:", aw.mean().item())
        print("Anchor height mean:", ah.mean().item())
        aw = gt_boxes[i][:, 2] - gt_boxes[i][:, 0]
        ah = gt_boxes[i][:, 3] - gt_boxes[i][:, 1]
        print("GT box:", gt_boxes[i])
        print("GT width mean:", aw.mean().item())
        print("GT height mean:", ah.mean().item())
        print("Cls logits mean:", cls_preds.mean().item())
        print("Cls logits std:", cls_preds.std().item())


        # Hard negative mining
        num_pos = matched_mask.sum().item()
        if num_pos > 0:
            # Probabilities of being positive (before softmax)
            with torch.no_grad():
                probs = F.softmax(cls_preds[i], dim=1)[:, 1]  # probability of class 1 (license plate)
                # Only consider negatives
                neg_probs = probs[labels == 0]
                # Take top-k hardest negatives
                k = min(neg_pos_ratio * num_pos, neg_probs.numel())
                if k > 0:
                    topk_vals, topk_idx = torch.topk(neg_probs, k)
                    hard_neg_mask = torch.zeros_like(labels, dtype=torch.bool)
                    neg_idx_in_all = (labels == 0).nonzero(as_tuple=True)[0]
                    hard_neg_mask[neg_idx_in_all[topk_idx]] = True
                else:
                    hard_neg_mask = torch.zeros_like(labels, dtype=torch.bool)
        else:
            hard_neg_mask = torch.zeros_like(labels, dtype=torch.bool)

        # Combine positives and hard negatives
        cls_mask = matched_mask | hard_neg_mask
        cls_loss += F.cross_entropy(cls_preds[i][cls_mask], labels[cls_mask])

        # Regression loss only for positives
        if matched_mask.sum() > 0:
            matched_gt_boxes = gt_boxes[i][matched_idxs[matched_mask]]
            matched_anchors = anchors[i][matched_mask]
            encoded_targets = box_coder.encode([matched_anchors], [matched_gt_boxes])[0]
            reg_loss += F.smooth_l1_loss(reg_preds[i][matched_mask], encoded_targets)

    return cls_loss + reg_loss



def detection_loss(cls_preds: torch.Tensor, reg_preds: torch.Tensor, anchors: list[torch.Tensor], gt_boxes: list[torch.Tensor], matcher: Matcher, box_coder: BoxCoder):
    batch_size = cls_preds.size(0)
    cls_loss = 0.0
    reg_loss = 0.0

    for i in range(batch_size):
        # Match anchors with GT
        iou_matrix = box_iou(gt_boxes[i], anchors[i])
        matched_idxs = matcher(iou_matrix)
        matched_mask = matched_idxs >= 0

        max_iou_per_anchor, _ = iou_matrix.max(dim=0)
        print("Max IoU anchors:", (max_iou_per_anchor > 0.5).sum().item())
        print("Max IoU > 0.3:", (max_iou_per_anchor > 0.3).sum().item())
        print("Max IoU (max):", max_iou_per_anchor.max().item())
        aw = anchors[i][:, 2] - anchors[i][:, 0]
        ah = anchors[i][:, 3] - anchors[i][:, 1]
        print(anchors[i][10:20])
        print("Anchor width mean:", aw.mean().item())
        print("Anchor height mean:", ah.mean().item())
        aw = gt_boxes[i][:, 2] - gt_boxes[i][:, 0]
        ah = gt_boxes[i][:, 3] - gt_boxes[i][:, 1]
        print(gt_boxes[i])
        print("GT width mean:", aw.mean().item())
        print("GT height mean:", ah.mean().item())
        print("Cls logits mean:", cls_preds.mean().item())
        print("Cls logits std:", cls_preds.std().item())


        # Classification loss
        labels = torch.zeros_like(cls_preds[i][:, 0], dtype=torch.long)
        labels[matched_mask] = 1
        cls_loss += F.cross_entropy(cls_preds[i], labels)

        # Regression loss for matched anchors
        if matched_mask.sum() > 0:
            matched_gt_boxes = gt_boxes[i][matched_idxs[matched_mask]]
            matched_anchors = anchors[i][matched_mask]
            encoded_targets = box_coder.encode([matched_anchors], [matched_gt_boxes])[0]
            reg_loss += F.smooth_l1_loss(reg_preds[i][matched_mask], encoded_targets)

    return cls_loss + reg_loss

def collate_fn(batch: list) -> tuple[torch.Tensor, list[torch.Tensor]]:
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = list(labels)
    return imgs, labels


class InceptionResidual(nn.Module):

    def __init__(self, inc: int, outc: int, *args: tuple[int, int] | int, include_max_pool: bool =True) -> None:
        super().__init__()

        # Inception module
        num_branches = len(args) + (1 if include_max_pool else 0)
        outc_hidden = outc//num_branches
        self.inception = nn.ModuleList()
        for kernel in args:
            if isinstance(kernel, int):
                pad = kernel // 2
            else:
                pad = (kernel[0] // 2, kernel[1] // 2)
            self.inception.append(
                nn.Sequential(
                    nn.Conv2d(inc, outc_hidden, kernel, padding=pad),
                    nn.BatchNorm2d(outc_hidden),
                    nn.ReLU()
                )
            )
        if include_max_pool:
            self.inception.append(nn.Sequential(
                nn.MaxPool2d(3, 1, padding=1),
                nn.Conv2d(inc, outc_hidden, 1),
                nn.BatchNorm2d(outc_hidden),
                nn.ReLU()
            ))
        # Conv2D for the inception module
        self.final_conv =  nn.Conv2d(outc, outc, 1, bias=False)
        # Decoder
        self.decoder = nn.ReLU()

        # Residual module
        if inc == outc:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(inc, outc, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [module(x) for module in self.inception]
        out = torch.cat(out, dim=1)

        out = self.final_conv(out) + self.residual(x)

        out = self.decoder(out)
        return out



class InResBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(outc, outc, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(outc, outc, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.merge = nn.Conv2d(3 * outc, inc, kernel_size=1)

        # Activación final
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        # 3 ramas en paralelo
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        # append = concatenación
        merged = torch.cat([b1, b2, b3], dim=1)

        # 1×1 conv
        merged = self.merge(merged)

        # add + ReLU
        out = merged + identity
        out = self.relu(out)
        return out

class Prediction(nn.Module):

    def __init__(self, inc: int, num_anchors: int) -> None:
        super().__init__()
        self.cls = nn.Conv2d(inc, 2 * num_anchors, kernel_size=3, padding=1) # (32, 16, 16) -> (2*num_anchors, 16, 16) or (48, 8, 8) -> (2*num_anchors, 8, 8)
        self.reg = nn.Conv2d(inc, 4 * num_anchors, kernel_size=3, padding=1) # (32, 16, 16) -> (4*num_anchors, 16, 16) or (48, 8, 8) -> (4*num_anchors, 8, 8)

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.cls(x)
        reg = self.reg(x)

        cls = cls.permute(0, 2, 3, 1).reshape(cls.size(0), -1, 2) # (2*num_anchors, 16, 16) -> (256*num_anchors, 2) or (2*num_anchors, 8, 8) -> (64*num_anchors, 2)
        reg = reg.permute(0, 2, 3, 1).reshape(reg.size(0), -1, 4) # (4*num_anchors, 16, 16) -> (256*num_anchors, 4) or (4*num_anchors, 8, 8) -> (64*num_anchors, 4)

        return cls, reg

class LPD(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=3, padding=2), # (3, 384, 384) -> (16, 128, 128)
            nn.ReLU(),
            InResBlock(16, 16) #(16, 128, 128) -> (16, 128, 128)
            #InceptionResidual(16, 16, 5, 3, 3, include_max_pool=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 24, 3, stride=2, padding=1),  # (16, 128, 128) -> (24, 64, 64)
            nn.ReLU(),
            InResBlock(24, 24) # (24, 64, 64) -> (24, 64, 64)
            #InceptionResidual(24, 24, 5, 3, 3, include_max_pool=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(24, 24, 3, stride=2, padding=1),  # (16, 128, 128) -> (24, 64, 64)
            nn.ReLU(),
            InResBlock(24, 24)  # (24, 64, 64) -> (24, 64, 64)
            # InceptionResidual(24, 24, 5, 3, 3, include_max_pool=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(24, 32, 3, stride=2, padding=1),  # (16, 128, 128) -> (24, 64, 64)
            nn.ReLU(),
            InResBlock(32, 32)  # (24, 64, 64) -> (24, 64, 64)
            # InceptionResidual(24, 24, 5, 3, 3, include_max_pool=True)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # (24, 128, 128) -> (32, 32, 32)
            nn.ReLU(),
            InResBlock(32, 32) # (32, 32, 32) -> (32, 32, 32)
            #InceptionResidual(32, 32, 5, 3, 3, include_max_pool=True)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(32, 48, 3, stride=2, padding=1),  # (32, 32, 32) -> (32, 16, 16)
            nn.ReLU(),
            InResBlock(48, 48) # (32, 16, 16) -> (32, 16, 16)
            #InceptionResidual(32, 32, 5, 3, 3, include_max_pool=True)
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(48, 56, 3, stride=2, padding=1),  # (32, 16, 16) -> (48, 8, 8)
            nn.ReLU(),
            InResBlock(56, 56)  # (48, 8, 8) -> (48, 8, 8)
            #InceptionResidual(48, 48, 5, 3, 3, include_max_pool=True)
        )

        self.FM0 = Prediction(48, num_anchors=3) #(32, 16, 16) -> ((2*num_anchors, 16, 16) # LP/no LP, or (4*num_anchors, 16, 16) # (minx, miny, maxx, maxy))
        self.FM1 = Prediction(56, num_anchors=3) #(48, 8, 8) -> ((2*num_anchors, 8, 8)# LP/no LP, or (4*num_anchors, 8, 8) # (minx, miny, maxx, maxy))

        self.anchors = AnchorGenerator(sizes=((30,), (60,)), aspect_ratios=((1.0, 0.5, 0.25), (1.0, 0.5, 0.25))
    )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        images = x
        batch, channels, height, width = images.size()
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        fm0 = self.block6(x)
        fm1 = self.block7(fm0)

        image_list = ImageList(images, [(height, width)] * batch)
        anchors = self.anchors(image_list, [fm0, fm1])

        cls0, reg0 = self.FM0(fm0)
        cls1, reg1 = self.FM1(fm1)

        cls = torch.cat([cls0, cls1], dim=1)
        reg = torch.cat([reg0, reg1], dim=1)

        return cls, reg, anchors


def main():

    parser = argparse.ArgumentParser(description='LPD_model training')
    parser.add_argument('--model-path', type=str, default=None, help='Trained model path (.pth) for loading')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs during the training')
    parser.add_argument('--output-path', type=str, default='models/OCR_model.pth', help='Path to save the trained model (.pth)')
    args = parser.parse_args()

    # Model loading
    device = torch.device('cpu')
    model = LPD()
    matcher = Matcher(0.55, 0.3, allow_low_quality_matches=True)
    box_coder = BoxCoder((1., 1., 1., 1.))
    optimizer = AdamW(model.parameters(), lr=0.0002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if args.model_path:
        print(f'Loading model from {args.model_path}')
        checkpoint = torch.load(args.model_path, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Checkpoint loaded. Last epoch: {last_epoch}, with loss: {loss}')
    else:
        last_epoch = 0

    train_dataset = CarPlateTrainDataset('dataset/')
    test_dataset = CarPlateTestDataset('dataset/')
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Train
    model.train()
    num_epochs = args.epochs + last_epoch
    for epoch in range(last_epoch+1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}', end=' ')
        for images, targets in train_loader:
            optimizer.zero_grad()

            cls_preds, reg_preds, anchors = model(images)
            loss = detection_loss_3v1(cls_preds, reg_preds, anchors, targets, matcher, box_coder)

            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f'loss: {loss.item()}')

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
