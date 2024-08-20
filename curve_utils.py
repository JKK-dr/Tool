import numpy as np
from scipy.special import comb


def bezier_curve(points, num_points, item_number):
    n = len(points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, item_number))
    for i in range(num_points):
        for j in range(n + 1):
            curve[i] += comb(n, j) * (1 - t[i])**(n - j) * t[i]**j * points[j]
    return curve

def fitting_curve(raw_data, num_points, item_number):
    control_points = np.array(raw_data)
    smoothed_curve = bezier_curve(control_points, num_points, item_number)
    return smoothed_curve.tolist()
