
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def visualize_cam(mask, img):
    """
    Overlay the heatmap from Grad-CAM on the original image.
    Args:
        mask (torch.Tensor): The Grad-CAM mask (H, W) or (1, 1, H, W).
        img (torch.Tensor): The input image tensor (C, H, W) with values in [0, 1].
    Returns:
        heatmap (torch.Tensor): The resized heatmap tensor (C, H, W).
        result (torch.Tensor): The combined image with heatmap overlay.
    """
    # Ensure the mask has the correct dimensions
    if mask.ndimension() == 4:
        mask = mask.squeeze(0).squeeze(0)

    # Resize the heatmap to match the original image dimensions
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(img.shape[1], img.shape[2]), mode="bilinear", align_corners=False)
    mask = mask.squeeze().cpu().numpy()

    # Convert mask to heatmap (Apply JET colormap for blue-based heatmaps)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)  # Normalize heatmap
    heatmap = heatmap.to(img.device)

    # Convert BGR to RGB (since OpenCV loads in BGR format)
    r, g, b = heatmap[2], heatmap[1], heatmap[0]
    heatmap = torch.stack([r, g, b])  # Correct color order

    ### FIX: Normalize Image to Ensure Visibility ###
    img = img.cpu()
    img = img - img.min()  # Shift min to 0
    img = img / img.max()  # Scale max to 1

    ### FIX: Adjust Overlay Blending to Prevent Blackout ###
    alpha = 0.5  # Transparency weight
    result = alpha * heatmap + (1 - alpha) * img
    result = result / result.max()  # Ensure values remain in valid range

    return heatmap, result
