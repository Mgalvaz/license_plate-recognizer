import torch
from PIL import Image
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes

from src.train_OCR_model import CRNN, ctc_decode, REVERSE_TRANSLATOR

if __name__ == '__main__':
    device = torch.device('cpu')
    threshold = 0.5

    # Load LPD model
    resnet = resnet50(weights='DEFAULT')
    return_layers = {
        'layer1': '0',
        'layer2': '1',
        'layer3': '2',
        'layer4': '3'
    }
    in_channels_list = [256, 512, 1024, 2048]
    backbone = BackboneWithFPN(resnet, return_layers=return_layers, in_channels_list=in_channels_list, out_channels=256)
    anchor_generator = AnchorGenerator(
        sizes=((90,), (220,), (360,), (512,), (1204,)),
        aspect_ratios=((0.33, 0.5, 1.0),) * 5
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )
    lpd_model = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    lpd_checkpoint = torch.load(r'C:\Repositorio\license_plate-recognizer\models\lpd_checkpoint_1.pth', map_location=device, weights_only=True)
    #print(lpd_checkpoint['epoch'], lpd_checkpoint['losses'])
    lpd_model.load_state_dict(lpd_checkpoint["model_state_dict"])

    # Load OCR model
    ocr_model = CRNN()
    ocr_checkpoint = torch.load(r'C:\Repositorio\license_plate-recognizer\models\OCR_model_4.pth', weights_only=True, map_location=device)
    ocr_model.load_state_dict(ocr_checkpoint['model_state_dict'])

    # Evaluation
    lpd_model.eval()
    ocr_model.eval()
    image = Image.open(r'C:\Repositorio\license_plate-recognizer\src\prueba9.jpg').convert("RGB")

    # LPD
    lpd_transform = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float, True)])
    image_tensor = lpd_transform(image)
    with torch.no_grad():
        predictions = lpd_model([image_tensor])
    prediction = predictions[0]
    boxes = prediction['boxes']
    scores = prediction['scores']
    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]

    # Check the results of the LPD
    image_with_boxes = draw_bounding_boxes(image_tensor, boxes=boxes, labels=None, width=3)
    v2.ToPILImage()(image_with_boxes).show()
    print('Bounding boxes:', boxes)
    print('Scores:', scores)

    # OCR
    box = boxes[1]
    xmin, ymin, xmax, ymax = box.int()
    crop_img = image_tensor[:, ymin:ymax, xmin:xmax]
    ocr_transform = v2.Compose([v2.Grayscale(num_output_channels=1), v2.Resize((32, 150))])
    crop_img = ocr_transform(crop_img).unsqueeze(0)
    v2.ToPILImage()(crop_img[0]).show() # Check the plate
    with torch.no_grad():
        output = ocr_model(crop_img).permute(1, 0, 2)

    # Check the results of the OCR
    pred = output.argmax(dim=2).T
    decoded_preds = ctc_decode(pred[0].tolist())
    print([REVERSE_TRANSLATOR[n] for n in decoded_preds])