import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

class OccupancyModel:
    """
    Binary occupancy classifier: empty vs occupied.
    A model file-t (occupancy_resnet18_best.pt) a Colabban mentetted.
    """

    def __init__(self, weights_path="occupancy_resnet18_best.pt", device=None, img_size=100):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size

        ckpt = torch.load(weights_path, map_location=self.device)
        self.class_to_idx = ckpt["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # ugyanaz az architektúra, mint tréningkor
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval().to(self.device)

    def _preprocess(self, roi_gray):
        """
        roi_gray: numpy (H,W) vagy (H,W,3)
        -> torch tensor (1,3,H,W) normalizálva
        """
        if roi_gray is None or roi_gray.size == 0:
            return None

        if roi_gray.ndim == 2:
            roi = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        else:
            roi = roi_gray

        roi = cv2.resize(roi, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        # [0..255] -> [0..1]
        x = torch.from_numpy(roi).permute(2, 0, 1).float() / 255.0

        # ugyanaz a normalize, mint tréningkor: mean=0.5 std=0.25
        x = (x - 0.5) / 0.25
        x = x.unsqueeze(0).to(self.device)
        return x

    @torch.no_grad()
    def predict_square(self, roi_gray):
        """
        Visszaad: (label, confidence)
        label: "empty" vagy "occupied"
        confidence: 0..1
        """
        x = self._preprocess(roi_gray)
        if x is None:
            return "empty", 0.0

        logits = self.model(x)
        prob = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(prob).item())
        label = self.idx_to_class[idx]
        conf = float(prob[idx].item())
        return label, conf
