# Automatic License Plate Detection and Recognition

This repository contains a complete Automatic License Plate Recognition (ALPR) pipeline developed entirely from scratch in PyTorch, including:

- A custom License Plate Detector (LPD)
- A custom Optical Character Recognizer (OCR)

Both components are implemented without relying on YOLO, Faster R-CNN, or pretrained architectures.
The work is inspired by the following research papers:

[Effective Deep Neural Networks for License Plate Detection and Recognition (2023)]

[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition]

[Effective Deep Neural Networks for License Plate Detection and Recognition (2023)]: https://www.researchgate.net/publication/358006875_Effective_deep_neural_networks_for_license_plate_detection_and_recognition

[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition]:https://arxiv.org/abs/1507.05717

---
## Project Overview
### License Plate Detector
A custom convolutional network designed to detect license plates in images.

- Multi-stage CNN with Inception-Residual blocks.
- Two feature maps used for detection (FM0 and FM1)
- Custom loss function combining:
  - CrossEntropy (classification)
  - Smooth L1 (regression)

This detector was trained on the UC3M-LP dataset, available here:

https://github.com/ramajoballester/UC3M-LP

### Optical Character Recognizer
A second model responsible for recognizing the characters inside the detected license plate.

- RCNN-based classifier for numbers and
- Supports A–Z and 0–9 recognition
- Compatible with cropped plates extracted from the LPD output
- Trained on a synthetic dataset generated specifically for alphanumeric license plate characters (Spanish license plates)

---





---

## Project Sctructure
```
├── dataset/                # UC3M-LP dataset
│   ├── test/
│   ├── train/
│   ├── test.txt
│   └── train.txt
│
├── models/            
│   ├── OCR_model.pth
│   └── LPD_model.pth
│
├── notebooks/              # Google Colab Notebooks for model training
│   ├── train_LPD_model.ipynb
│   └── train_OCR_model.ipynb
│
├── src/
│   ├── LPD_dataset.py      # Train and Test dataset classes for the LPD model
│   ├── OCR_dataset.py      # Dataset class for the OCR model
│   ├── train_LPD_model.py  # Script to train the LPD model
│   └── train_OCR_model.py  # Script to train the OCR 
│
├── LICENSE
├── README.md
└── requirements.txt
```