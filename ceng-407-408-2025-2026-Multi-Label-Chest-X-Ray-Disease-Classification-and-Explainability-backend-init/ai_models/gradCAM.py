import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class GradCAMPlusPlus:
    def __init__(self, model, target_layer, threshold=0.3, line_width=2):
        self.model = model
        self.target_layer = target_layer
        self.threshold = threshold
        self.line_width = line_width
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx, resize_to=None):
        """
        x: input tensor [1,3,H,W]
        class_idx: class index to compute CAM for
        resize_to: (H, W) tuple to resize CAM to
        """
        self.model.zero_grad()
        output = self.model(x)
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        # Grad-CAM++ weighting
        grads2 = grads ** 2
        grads3 = grads ** 3
        eps = 1e-8
        alpha = grads2 / (2 * grads2 + (acts * grads3).sum(dim=(2,3), keepdim=True) + eps)
        weights = (alpha * F.relu(grads)).sum(dim=(2,3), keepdim=True)

        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)

        # Normalize
        cam -= cam.min()
        cam /= cam.max() + eps
        cam_np = cam[0].detach().cpu().numpy()

        # Resize to original image size if requested
        if resize_to is not None:
            cam_np = np.array(
                Image.fromarray(cam_np).resize(resize_to[::-1], resample=Image.BILINEAR)
            )

        return cam_np







