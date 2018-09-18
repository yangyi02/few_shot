"""
    Simple Usage example (with 3 images)
"""
from detection_map import DetectionMAP
from show_frame import show_frame
import numpy as np
import matplotlib.pyplot as plt

pred_bb1 = np.array([[0.880688, 0.44609185, 0.95696718, 0.6476958],
                     [0.84020283, 0.45787981, 0.99351478, 0.64294884],
                     [0.78723741, 0.61799151, 0.9083041, 0.75623035],
                     [0.22078986, 0.30151826, 0.36679274, 0.40551913],
                     [0.0041579, 0.48359361, 0.06867643, 0.60145104],
                     [0.4731401, 0.33888632, 0.75164948, 0.80546954],
                     [0.75489414, 0.75228018, 0.87922037, 0.88110524],
                     [0.21953127, 0.77934921, 0.34853417, 0.90626764],
                     [0.81, 0.11, 0.91, 0.21]])
pred_cls1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
pred_conf1 = np.array([0.95, 0.75, 0.4, 0.3, 1, 1, 0.75, 0.5, 0.8])
gt_bb1 = np.array([[0.86132812, 0.48242188, 0.97460938, 0.6171875],
                   [0.18554688, 0.234375, 0.36132812, 0.41601562],
                   [0., 0.47265625, 0.0703125, 0.62109375],
                   [0.47070312, 0.3125, 0.77929688, 0.78125],
                   [0.8, 0.1, 0.9, 0.2]])
gt_cls1 = np.array([0, 0, 0, 0, 0])

if __name__ == '__main__':
    frames = [(pred_bb1, pred_cls1, pred_conf1, gt_bb1, gt_cls1)]
    n_class = 2

    mAP = DetectionMAP(n_class)
    for i, frame in enumerate(frames):
        print("Evaluate frame {}".format(i))
        show_frame(*frame)
        mAP.evaluate(*frame)

    mAP.plot()
    plt.show()
    #plt.savefig("pr_curve_example.png")
