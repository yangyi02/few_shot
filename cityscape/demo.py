import torch
from PIL import Image
import cv2
import scipy.misc as sm
import numpy as np
import argparse
from networks.base import BaseNet
from networks.base_3d import Base3DNet

def main():
    arg_parser = argparse.ArgumentParser(description='', add_help=False)
    arg_parser.add_argument('--image_name', default='image.png')
    arg_parser.add_argument('--depth_name', default='depth.png')
    arg_parser.add_argument('--flow_name', default='flow.png')
    arg_parser.add_argument('--output_name', default='prediction.png')
    arg_parser.add_argument('--model_type', default='base')
    arg_parser.add_argument('--model_path', default='cache/models/base.pth')
    arg_parser.add_argument('--im_height', type=int, default=256)
    arg_parser.add_argument('--im_width', type=int, default=512)
    arg_parser.add_argument('--image_channel', type=int, default=3)
    arg_parser.add_argument('--depth_channel', type=int, default=1)
    arg_parser.add_argument('--num_class', type=int, default=34)
    args = arg_parser.parse_args()

    # Load model
    if args.model_type == 'base':
        model = BaseNet(args.image_channel, args.num_class)
    elif args.model_type == 'base_3d':
        model = Base3DNet(args.image_channel, args.depth_channel, args.num_class)
    else:
        print('Model not implemented yet')
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path))

    # Load data
    image = np.array(Image.open(args.image_name))
    image = image / 255.0
    depth = np.array(Image.open(args.depth_name))
    depth = depth.astype(np.float32) / 256.0
    # flow = np.array(Image.open(args.flow_name))

    # Process data
    images = np.zeros((1, args.im_height, args.im_width, args.image_channel))
    depths = np.zeros((1, args.im_height, args.im_width, args.depth_channel))
    flows = np.zeros((1, args.im_height, args.im_width, 2))
    images[0, :, :, :] = cv2.resize(image, (args.im_width, args.im_height),
                                    interpolation=cv2.INTER_AREA)
    depths[0, :, :, 0] = cv2.resize(depth, (args.im_width, args.im_height),
                                    interpolation=cv2.INTER_AREA)
    # flows[0, :, :, :] = cv2.resize(flow, (args.im_width, args.im_height),
    #                                interpolation=cv2.INTER_AREA)
    im = torch.from_numpy(images.transpose((0, 3, 1, 2))).float().to(device)
    dp = torch.from_numpy(depths.transpose((0, 3, 1, 2))).float().to(device)
    fl = torch.from_numpy(flows.transpose((0, 3, 1, 2))).float().to(device)

    # Make prediction
    pred = model([im], [dp], [fl])
    _, pred_label = torch.max(pred, 1)
    print('input image shape: ', images[0].shape)
    print('output segmentaiton shape: ', pred_label[0].shape)
    prediction = pred_label[0].cpu().numpy()
    sm.imsave(args.output_name, prediction)


if __name__ == '__main__':
    main()
