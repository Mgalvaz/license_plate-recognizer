"""
Script that trains the OCR model with synthetic generated license plates.

Usage:
    python train_OCR_model.py [--model-path path] [--output-path out] [--epochs num] [N].

Input:
    path (str, optional): Trained model path (.pth) for loading.
    out (str, optional): Path to save the trained model (.pth).
    num (int, optional): Number of epochs during the training.
    N (int, optional): Number of train images in the dataset. Default is 10000

Output:
    If out argument was passed, the model is saved in the following format:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, out)
"""
import argparse

import torch
from torch import nn
from torch.functional import F
from torch.optim import AdamW
from torchvision.ops import box_iou
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection._utils import Matcher, BoxCoder

def detection_loss(cls_preds: torch.Tensor, reg_preds: torch.Tensor, anchors: list[torch.Tensor], gt_boxes: list[torch.Tensor], matcher: Matcher, box_coder: BoxCoder):
    batch_size = cls_preds.size(0)
    cls_loss = 0.0
    reg_loss = 0.0

    for i in range(batch_size):
        # Match anchors with GT
        iou_matrix = box_iou(anchors[i], gt_boxes[i])
        matched_idxs = matcher(iou_matrix)
        matched_mask = matched_idxs >= 0

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

    cls_loss /= batch_size
    reg_loss /= batch_size

    return cls_loss + reg_loss

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

class Prediction(nn.Module):

    def __init__(self, inc: int) -> None:
        super().__init__()
        self.cls = nn.Conv2d(inc, 2, kernel_size=3, padding=1) # (32, 16, 16) -> (2, 16, 16) or (48, 8, 8) -> (2, 8, 8)
        self.reg = nn.Conv2d(inc, 4, kernel_size=3, padding=1) # (32, 16, 16) -> (4, 16, 16) or (48, 8, 8) -> (4, 8, 8)

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.cls(x)
        reg = self.reg(x)

        cls = cls.permute(0, 2, 3, 1).reshape(cls.size(0), -1, 2) # (2, 16, 16) -> (256, 2) or (2, 8, 8) -> (64, 2)
        reg = reg.permute(0, 2, 3, 1).reshape(reg.size(0), -1, 4) # (4, 16, 16) -> (256, 4) or (4, 8, 8) -> (64, 4)

        return cls, reg

class LPD(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=3, padding=2), # (3, 384, 384) -> (16, 128, 128)
            InceptionResidual(16, 16, 5, 3, 3, include_max_pool=True), #(16, 128, 128) -> (16, 128, 128)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 24, 3, stride=2, padding=1),  # (16, 128, 128) -> (24, 64, 64)
            InceptionResidual(24, 24, 5, 3, 3, include_max_pool=True), # (24, 64, 64) -> (24, 64, 64)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(24, 32, 3, stride=2, padding=1),  # (24, 128, 128) -> (32, 32, 32)
            InceptionResidual(32, 32, 5, 3, 3, include_max_pool=True) , # (32, 32, 32) -> (32, 32, 32)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # (32, 32, 32) -> (32, 16, 16)
            InceptionResidual(32, 32, 5, 3, 3, include_max_pool=True),  # (32, 16, 16) -> (32, 16, 16)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(32, 48, 3, stride=2, padding=1),  # (32, 16, 16) -> (48, 8, 8)
            InceptionResidual(48, 48, 5, 3, 3, include_max_pool=True)  # (48, 8, 8) -> (48, 8, 8)
        )

        self.FM0 = Prediction(32) #(32, 16, 16) -> ((2, 16, 16) # matricula/no matricula, (4, 16, 16) # (dx, dy, dw, dh))
        self.FM1 = Prediction(48) #(48, 8, 8) -> ((2, 8, 8)# matricula/no matricula, (4, 8, 8) # (dx, dy, dw, dh))

        self.anchors = AnchorGenerator(sizes=((39.6,), (79.2,)), aspect_ratios=((2.0,), (2.0,))
    )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        images = x
        batch, channels, height, width = images.size()
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        fm0 = self.block4(x)
        fm1 = self.block5(fm0)

        image_list = ImageList(images, [(height, width)] * batch)
        anchors = self.anchors(image_list, [fm0, fm1])

        cls0, reg0 = self.FM0(fm0)
        cls1, reg1 = self.FM1(fm1)

        cls = torch.cat([cls0, cls1], dim=1)
        reg = torch.cat([reg0, reg1], dim=1)

        return cls, reg, anchors




def main():

    parser = argparse.ArgumentParser(description='OCR_model training')
    parser.add_argument('total', metavar='N', type=int, nargs='?', default=10000, help='Number of train images in the dataset')
    parser.add_argument('--model-path', type=str, default=None, help='Trained model path (.pth) for loading')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs during the training')
    parser.add_argument('--output-path', type=str, default='models/OCR_model.pth', help='Path to save the trained model (.pth)')
    args = parser.parse_args()

    model = LPD()
    matcher = Matcher(0.5, 0.3)
    box_coder = BoxCoder((1., 1., 1., 1.))
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    num_epochs = 1

    model.train()
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}', end=' ')
        for images, targets in dataloader:

            # Targets del dataset
            gt_boxes = [t["boxes"] for t in targets]  # each [num_gt,4]
            gt_labels = [t["labels"] for t in targets]  # each [num_gt]

            # --------------------------------------------------------
            # 1. Forward del modelo
            # --------------------------------------------------------
            cls_preds, reg_preds, anchors = model(images)
            # cls_preds: [B, N, 2]
            # reg_preds: [B, N, 4]
            # anchors:   list len B, each [N,4]

            loss = detection_loss(cls_preds, reg_preds, anchors, gt_boxes, matcher, box_coder)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f'loss: {loss.item()}')




if __name__ == "__main__":
    main()
