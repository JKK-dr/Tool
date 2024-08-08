import numpy as np
import cv2
import matplotlib.pyplot as plt

# 该api主要对mask2d进行处理，包括轮廓处理等，主要用于度量导航目标点云处理

def get_segment_islands_pos(segment_map, label_id, detect_internal_contours=False):
    mask = segment_map == label_id
    mask = mask.astype(np.uint8)
    detect_type = cv2.RETR_EXTERNAL
    if detect_internal_contours:
        detect_type = cv2.RETR_TREE

    contours, hierarchy = cv2.findContours(mask, detect_type, cv2.CHAIN_APPROX_SIMPLE)
    # convert contours back to numpy index order
    contours_list = []
    for contour in contours:
        tmp = contour.reshape((-1, 2))
        tmp_1 = np.stack([tmp[:, 1], tmp[:, 0]], axis=1)
        contours_list.append(tmp_1)

    centers_list = []
    bbox_list = []
    for c in contours_list:
        xmin = np.min(c[:, 0])
        xmax = np.max(c[:, 0])
        ymin = np.min(c[:, 1])
        ymax = np.max(c[:, 1])
        bbox_list.append([xmin, xmax, ymin, ymax])

        centers_list.append([(xmin + xmax) / 2, (ymin + ymax) / 2])

    return contours_list, centers_list, bbox_list, hierarchy

mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
# print(f"showing mask for object cat {name}")
# cv2.imshow(f"mask_{name}", (mask_2d.astype(np.float32) * 255).astype(np.uint8))
# cv2.waitKey()

foreground = binary_closing(mask_2d, iterations=3)
foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
foreground = foreground > 0.5
# cv2.imshow(f"mask_{name}_gaussian", (foreground * 255).astype(np.uint8))
foreground = binary_dilation(foreground)

contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)
# print("centers", centers)

# whole map position
for i in range(len(contours)):
    centers[i][0] += self.rmin
    centers[i][1] += self.cmin
    bbox_list[i][0] += self.rmin
    bbox_list[i][1] += self.rmin
    bbox_list[i][2] += self.cmin
    bbox_list[i][3] += self.cmin
    for j in range(len(contours[i])):
        contours[i][j, 0] += self.rmin
        contours[i][j, 1] += self.cmin
