import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


img1 = cv2.imread("")
img2 = cv2.imread("")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

plt.imshow(img2)

diferenca_absoluta = cv2.absdiff(img1, img2)
    
similaridade, _ = ssim(img1, img2, full=True)

print(f"Similaridade (SSIM): {similaridade:.4f}")
print("(1.0 = idênticas, 0.0 = completamente diferentes)")




