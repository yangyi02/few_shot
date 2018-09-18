import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import flowlib
import detect_loss
from utils import nms

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class DetectInterface(object):
    def __init__(self, data, model, learning_rate, train_iteration, test_iteration, test_interval,
                 save_interval, init_model_path, save_model_path, tensorboard_path):
        self.data = data
        self.model = model
        self.learning_rate = learning_rate
        self.train_iter = train_iteration
        self.test_iter = test_iteration
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.best_test_acc = -1e10
        self.init_model_path = init_model_path
        self.save_model_path = save_model_path
        self.tensorboard_path = tensorboard_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_model()

    def init_model(self):
        # model = torch.nn.DataParallel(model).cuda()
        self.model = self.model.to(self.device)
        if self.init_model_path is not '':
            self.model.load_state_dict(torch.load(self.init_model_path))
        return self.model

    @staticmethod
    def get_accuracy(pred_heatmap, heatmap):
        acc, cnt = 0, 0
        for i in range(len(heatmap)):
            pred_label = pred_heatmap[i] > 0
            acc = acc + (pred_label.int() == heatmap[i].int()).sum()
            cnt = cnt + heatmap[i].numel()
        mean_acc = acc.float() / cnt
        base_acc = 0
        for i in range(len(heatmap)):
            base_acc = base_acc + (heatmap[i] < 0.5).int().sum()
        base_acc = base_acc.float() / cnt
        return mean_acc, base_acc

    def train(self):
        self.model.train()
        torch.set_grad_enabled(True)
        writer = SummaryWriter(self.tensorboard_path)
        criterion_conf = detect_loss.BCELoss2d()
        criterion_loc = detect_loss.MSELoss2d()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_loss_all, train_acc_all, base_acc_all = [], [], []
        index, cnt = np.random.permutation(len(self.data.train_anno['img'])), 0
        for it in range(self.train_iter):
            im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of, cnt, restart = \
                self.data.get_next_batch('train', cnt, index)
            im = [torch.from_numpy(i_s).float().to(self.device) for i_s in im]
            dp = [torch.from_numpy(d_s).float().to(self.device) for d_s in dp]
            fl = [torch.from_numpy(f_s).float().to(self.device) for f_s in fl]
            lb = [torch.from_numpy(l_s).float().to(self.device) for l_s in lb]
            of = [torch.from_numpy(o_s).float().to(self.device) for o_s in of]

            optimizer.zero_grad()
            pred, pred_of = self.model(im, dp, fl)
            loss = 0
            for i in range(len(lb)):
                loss = loss + criterion_conf(pred[i], lb[i])
            loss = loss * 10
            for i in range(len(lb)):
                mask = lb[i] > 0
                mask = mask.float()
                loss = loss + criterion_loc(pred_of[i], of[i], mask)
            loss.backward()
            optimizer.step()

            acc, base_acc = self.get_accuracy(pred, lb)
            writer.add_scalar('train_loss', loss, it)
            train_loss_all.append(loss)
            if len(train_loss_all) > 100:
                train_loss_all.pop(0)
            ave_train_loss = sum(train_loss_all) / float(len(train_loss_all))
            logging.info('iteration %d, train loss: %.4f, average train loss: %.4f', it, loss, ave_train_loss)
            writer.add_scalar('train_acc', acc, it)
            train_acc_all.append(acc)
            if len(train_acc_all) > 100:
                train_acc_all.pop(0)
            ave_train_acc = sum(train_acc_all) / float(len(train_acc_all))
            logging.info('iteration %d, train accuracy: %.4f, average train accuracy: %.4f', it, acc, ave_train_acc)
            base_acc_all.append(base_acc)
            if len(base_acc_all) > 100:
                base_acc_all.pop(0)
            ave_base_acc = sum(base_acc_all) / float(len(base_acc_all))
            logging.info('iteration %d, base acc: %.4f, average base acc: %.4f', it, base_acc, ave_base_acc)
            if (it + 1) % self.save_interval == 0:
                logging.info('iteration %d, saving model', it)
                with open(self.save_model_path, 'w') as handle:
                    torch.save(self.model.state_dict(), handle)
            if (it + 1) % self.test_interval == 0:
                logging.info('iteration %d, testing', it)
                test_loss, test_acc = self.validate()
                writer.add_scalar('test_loss', test_loss, it)
                writer.add_scalar('test_acc', test_acc, it)
                self.model.train()
                torch.set_grad_enabled(True)
            if restart:
                index, cnt = np.random.permutation(len(self.data.train_anno['img'])), 0
        writer.close()

    def validate(self):
        test_loss, test_acc = self.test()
        if test_acc >= self.best_test_acc:
            with open(self.save_model_path, 'w') as handle:
                torch.save(self.model.state_dict(), handle)
            logging.info('model save to %s', self.save_model_path)
            self.best_test_acc = test_acc
        logging.info('current best test accuracy: %.4f', self.best_test_acc)
        return test_loss, test_acc

    def test(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        criterion_conf = detect_loss.BCELoss2d()
        criterion_loc = detect_loss.MSELoss2d()
        test_loss_all, test_acc_all, base_acc_all = [], [], []
        index, cnt = np.random.permutation(len(self.data.test_anno['img'])), 0
        for it in range(self.test_iter):
            im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of, cnt, _ = \
                self.data.get_next_batch('test', cnt, index)
            im = [torch.from_numpy(i_s).float().to(self.device) for i_s in im]
            dp = [torch.from_numpy(d_s).float().to(self.device) for d_s in dp]
            fl = [torch.from_numpy(f_s).float().to(self.device) for f_s in fl]
            lb = [torch.from_numpy(l_s).float().to(self.device) for l_s in lb]
            of = [torch.from_numpy(o_s).float().to(self.device) for o_s in of]

            pred, pred_of = self.model(im, dp, fl)
            loss = 0
            for i in range(len(lb)):
                loss = loss + criterion_conf(pred[i], lb[i])
            loss = loss * 10
            for i in range(len(lb)):
                mask = lb[i] > 0
                mask = mask.float()
                loss = loss + criterion_loc(pred_of[i], of[i], mask)
            acc, base_acc = self.get_accuracy(pred, lb)

            test_loss_all.append(loss)
            if len(test_loss_all) > 100:
                test_loss_all.pop(0)
            test_acc_all.append(acc)
            if len(test_acc_all) > 100:
                test_acc_all.pop(0)
            base_acc_all.append(base_acc)
            if len(base_acc_all) > 100:
                base_acc_all.pop(0)

        test_loss = np.mean(np.array(test_loss_all))
        test_acc = np.mean(np.array(test_acc_all))
        base_acc = np.mean(np.array(base_acc_all))
        logging.info('average test loss: %.4f', test_loss)
        logging.info('average test accuracy: %.4f', test_acc)
        logging.info('average base accuracy: %.4f', base_acc)
        return test_loss, test_acc

    def test_all(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        criterion_conf = detect_loss.BCELoss2d()
        criterion_loc = detect_loss.MSELoss2d()
        test_loss_all, test_acc_all, base_acc_all = [], [], []
        cnt = 0
        while True:
            im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of, cnt, restart = \
                self.data.get_next_batch('test', cnt)
            if restart:
                break
            im = [torch.from_numpy(i_s).float().to(self.device) for i_s in im]
            dp = [torch.from_numpy(d_s).float().to(self.device) for d_s in dp]
            fl = [torch.from_numpy(f_s).float().to(self.device) for f_s in fl]
            lb = [torch.from_numpy(l_s).float().to(self.device) for l_s in lb]
            of = [torch.from_numpy(o_s).float().to(self.device) for o_s in of]

            pred, pred_of = self.model(im, dp, fl)
            loss = 0
            for i in range(len(lb)):
                loss = loss + criterion_conf(pred[i], lb[i])
            loss = loss * 10
            for i in range(len(lb)):
                mask = lb[i] > 0
                mask = mask.float()
                loss = loss + criterion_loc(pred_of[i], of[i], mask)
            acc, base_acc = self.get_accuracy(pred, lb)

            test_loss_all.append(loss)
            test_acc_all.append(acc)
            base_acc_all.append(base_acc)
            logging.info('at image %d, test loss: %.4f', cnt, loss)
            logging.info('at image %d, test accuracy: %.4f', cnt, acc)
            logging.info('at image %d, base accuracy: %.4f', cnt, base_acc)

        test_loss = np.mean(np.array(test_loss_all))
        test_acc = np.mean(np.array(test_acc_all))
        base_acc = np.mean(np.array(base_acc_all))
        logging.info('overall average test loss: %.4f', test_loss)
        logging.info('overall average test accuracy: %.4f', test_acc)
        logging.info('overall average base accuracy: %.4f', base_acc)

    def create_box_pair(self, pred_box, box):
        frames = []
        for i in range(len(box)):
            # print pred_box[i], box[i]
            pred_bb = pred_box[i][:, 0:4]
            pred_conf = pred_box[i][:, 4]
            pred_cls = np.zeros((pred_bb.shape[0],))
            gt_bb = box[i][:, 0:4]
            gt_cls = np.zeros((gt_bb.shape[0],))
            frames.append((pred_bb, pred_cls, pred_conf, gt_bb, gt_cls))
        return frames

    def predict(self, image, depth, flow, label_map, offset):
        self.model.eval()
        torch.set_grad_enabled(False)
        im = [torch.from_numpy(i_s).float().to(self.device) for i_s in image]
        dp = [torch.from_numpy(d_s).float().to(self.device) for d_s in depth]
        fl = [torch.from_numpy(f_s).float().to(self.device) for f_s in flow]
        lb = [torch.from_numpy(l_s).float().to(self.device) for l_s in label_map]
        of = [torch.from_numpy(o_s).float().to(self.device) for o_s in offset]
        pred, pred_of = self.model(im, dp, fl)
        criterion_conf = detect_loss.BCELoss2d()
        criterion_loc = detect_loss.MSELoss2d()
        loss = 0
        for i in range(len(lb)):
            loss = loss + criterion_conf(pred[i], lb[i])
        loss = loss * 10
        for i in range(len(lb)):
            mask = lb[i] > 0
            mask = mask.float()
            loss = loss + criterion_loc(pred_of[i], of[i], mask)
        for i in range(len(pred)):
            pred[i] = torch.sigmoid(pred[i])
            pred[i] = pred[i].cpu().numpy()
            pred_of[i] = pred_of[i].cpu().numpy()
        return pred, pred_of, loss

    def visualize_all(self, image_list, figure_path):
        lines = open(image_list).readlines()
        for line in lines:
            line = line.strip()
            dirs = line.split('/')
            image_name = dirs[-1]
            file_name, file_ext = os.path.splitext(image_name)
            file_id = file_name.split('_')[0]
            image_name = os.path.join(self.data.image_dir, file_id + '.png')
            depth_name = os.path.join(self.data.depth_dir, file_id + '.png')
            flow_name = os.path.join(self.data.flow_dir, file_id + '.png')
            box_name = os.path.join(self.data.box_dir, file_id + '.txt')
            self.visualize(image_name, depth_name, flow_name, box_name, figure_path)

    def visualize(self, image_name, depth_name, flow_name, box_name, figure_path):
        im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of = \
            self.data.get_one_sample(image_name, depth_name, flow_name, box_name)
        pred, pred_of, loss = self.predict(im, dp, fl, lb, of)
        pred_box = nms(pred, pred_of, self.data.orig_im_size[0], self.data.orig_im_size[1])
        if figure_path is '':
            self.visualize_groundtruth(orig_im, im, orig_dp, dp, orig_fl, fl, box, lb, of)
            self.visualize_prediction(im, dp, fl, pred, pred_of, loss)
            self.visualize_box(orig_im, orig_dp, orig_fl, pred_box, loss)
            plt.show()
            plt.close('all')
        else:
            if not os.path.exists(figure_path):
                os.makedirs(figure_path)
            dirs = image_name.split('/')
            sub_dir, image_name = dirs[-2], dirs[-1]
            file_name, file_ext = os.path.splitext(image_name)
            file_id = file_name.split('_')[0]
            figure_prefix = os.path.join(figure_path, sub_dir + '_' + file_id)
            self.visualize_groundtruth(orig_im, im, orig_dp, dp, orig_fl, fl, box, lb, of, figure_prefix)
            self.visualize_prediction(im, dp, fl, pred, pred_of, loss, figure_prefix)
            self.visualize_box(orig_im, orig_dp, orig_fl, pred_box, loss, figure_prefix)

    def visualize_prediction(self, images, depths, flows, prediction, prediction_offset, loss, figure_prefix=''):
        # construct a full image containing all scale images and
        # a full attention map containing all scale attention maps
        max_im_size = np.max(np.array(self.data.im_widths))
        sum_im_size = np.sum(np.array(self.data.im_heights))
        im_all = np.zeros((sum_im_size, max_im_size, 3))
        dp_all = np.zeros((sum_im_size, max_im_size, 3))
        fl_all = np.zeros((sum_im_size, max_im_size, 3))
        max_ou_size = np.max(np.array(self.data.output_widths))
        sum_ou_size = np.sum(np.array(self.data.output_heights))
        pred_all = np.zeros((sum_ou_size, max_ou_size))
        cnt_im, cnt_pred = 0, 0
        for i in range(len(images)):
            im = images[i][0, :, :, :].transpose(1, 2, 0)
            height, width = im.shape[0], im.shape[1]
            im_all[cnt_im:cnt_im + height, 0:width, :] = im

            dp = depths[i][0, 0, :, :]
            dp = flowlib.visualize_disp(dp)
            dp_all[cnt_im:cnt_im + height, 0:width, :] = dp

            fl = flows[i][0, :, :, :].transpose(1, 2, 0)
            fl = flowlib.visualize_flow(fl)
            fl_all[cnt_im:cnt_im + height, 0:width, :] = fl
            cnt_im = cnt_im + height

            pred = prediction[i][0, 0, :, :]
            height, width = pred.shape[0], pred.shape[1]
            pred_all[cnt_pred:cnt_pred + height, 0:width] = pred
            cnt_pred = cnt_pred + height

        pred_on_image = self.visualize_prediction_on_image(im_all, pred_all)

        if figure_prefix is '':
            fig, ax = plt.subplots(1)
            ax.imshow(im_all)
            fig, ax = plt.subplots(1)
            ax.imshow(dp_all)
            fig, ax = plt.subplots(1)
            ax.imshow(fl_all)
            fig, ax = plt.subplots(1)
            ax.imshow(pred_all)
            fig, ax = plt.subplots(1)
            ax.imshow(pred_on_image)
        else:
            # self.save_image(im_all, figure_prefix + '_image_all.jpg')
            # self.save_image(dp_all, figure_prefix + '_depth_all.jpg')
            # self.save_image(fl_all, figure_prefix + '_flow_all.jpg')
            # self.save_image(pred_all, figure_prefix + '_pred_all.jpg')
            self.save_image(pred_on_image, figure_prefix + '_pred_loss_%.4f.jpg' % loss)

    @staticmethod
    def visualize_prediction_on_image(image, prediction):
        im_height, im_width = image.shape[0], image.shape[1]
        pred = cv2.resize(prediction, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
        pred = np.dstack((pred, pred, pred))
        pred_on_image = image * 0.2 + pred * 0.8
        return pred_on_image

    def visualize_groundtruth(self, orig_image, images, orig_depth, depths, orig_flow, flows, boxes, label_maps, offsets, figure_prefix=''):
        orig_im = orig_image[0, :, :, :].transpose(1, 2, 0)
        im_height, im_width = orig_im.shape[0], orig_im.shape[1]

        # Plot original large image with bounding box
        if len(boxes[0]) > 0:
            b = boxes[0].copy()
            b[:, 0], b[:, 2] = b[:, 0] * im_width, b[:, 2] * im_width
            b[:, 1], b[:, 3] = b[:, 1] * im_height, b[:, 3] * im_height
            b = b.astype(np.int)
        else:
            b = []

        orig_im = orig_im * 255.0
        orig_im = orig_im.astype(np.uint8).copy()
        if len(b) > 0:
            for i in range(b.shape[0]):
                cv2.rectangle(orig_im, (b[i, 0], b[i, 1]), (b[i, 2], b[i, 3]), (0, 255, 0), 3)
                cv2.putText(orig_im, self.data.inverse_class_map[1], (b[i, 0] + 2, b[i, 1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Plot original depth with bounding box
        orig_dp = orig_depth[0, 0, :, :]
        orig_dp = flowlib.visualize_disp(orig_dp)
        orig_dp = orig_dp * 255.0
        orig_dp = orig_dp.astype(np.uint8).copy()
        if len(b) > 0:
            for i in range(b.shape[0]):
                cv2.rectangle(orig_dp, (b[i, 0], b[i, 1]), (b[i, 2], b[i, 3]), (255, 255, 255), 3)
                cv2.putText(orig_dp, self.data.inverse_class_map[1], (b[i, 0] + 2, b[i, 1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Plot original flow with bounding box
        orig_fl = orig_flow[0, :, :, :].transpose(1, 2, 0)
        orig_fl = flowlib.visualize_flow(orig_fl)
        orig_fl = orig_fl.astype(np.uint8).copy()
        if len(b) > 0:
            for i in range(b.shape[0]):
                cv2.rectangle(orig_fl, (b[i, 0], b[i, 1]), (b[i, 2], b[i, 3]), (0, 0, 0), 3)
                cv2.putText(orig_fl, self.data.inverse_class_map[1], (b[i, 0] + 2, b[i, 1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        max_im_size = np.max(np.array(self.data.im_widths))
        sum_im_size = np.sum(np.array(self.data.im_heights))
        im_all = np.zeros((sum_im_size, max_im_size, 3))
        dp_all = np.zeros((sum_im_size, max_im_size, 3))
        fl_all = np.zeros((sum_im_size, max_im_size, 3))
        max_ou_size = np.max(np.array(self.data.output_widths))
        sum_ou_size = np.sum(np.array(self.data.output_heights))
        pred_all = np.zeros((sum_ou_size, max_ou_size))
        cnt_im = 0
        for i in range(len(images)):
            im = images[i][0, :, :, :].transpose(1, 2, 0)
            height, width = im.shape[0], im.shape[1]
            im_all[cnt_im:cnt_im + height, 0:width, :] = im
            cnt_im = cnt_im + height

        max_ou_size = np.max(np.array(self.data.output_widths))
        sum_ou_size = np.sum(np.array(self.data.output_heights))
        lb_all = np.zeros((sum_ou_size, max_ou_size))
        cnt_lb = 0
        for i in range(len(label_maps)):
            lb = label_maps[i][0, 0, :, :]
            height, width = lb.shape[0], lb.shape[1]
            lb_all[cnt_lb:cnt_lb + height, 0:width] = lb
            cnt_lb = cnt_lb + height
        label_on_image = self.visualize_prediction_on_image(im_all, lb_all)

        orig_im2 = orig_image[0, :, :, :].transpose(1, 2, 0)
        im_height, im_width = orig_im2.shape[0], orig_im2.shape[1]
        orig_im2 = orig_im2 * 255.0
        orig_im2 = orig_im2.astype(np.uint8).copy()
        for i in range(len(label_maps)):
            of = offsets[i][0, :, :, :].transpose(1, 2, 0)
            lb = label_maps[i][0, 0, :, :]
            ou_height, ou_width = lb.shape[0], lb.shape[1]
            b_idx = np.nonzero(lb)
            y_c, x_c = b_idx[0], b_idx[1]
            b = np.zeros((len(b_idx[0]), 4))
            for k in range(b.shape[0]):
                offset = of[y_c[k], x_c[k], :]
                b[k, 0] = x_c[k] * 1.0 / ou_width + offset[0]
                b[k, 1] = y_c[k] * 1.0 / ou_height + offset[1]
                b[k, 2] = x_c[k] * 1.0 / ou_width + offset[2]
                b[k, 3] = y_c[k] * 1.0 / ou_height + offset[3]
                b[k, 0], b[k, 2] = b[k, 0] * im_width, b[k, 2] * im_width
                b[k, 1], b[k, 3] = b[k, 1] * im_height, b[k, 3] * im_height
            b = b.astype(np.int)
            for k in range(b.shape[0]):
                cv2.rectangle(orig_im2, (b[k, 0], b[k, 1]), (b[k, 2], b[k, 3]), (0, 255, 0), 3)
                cv2.putText(orig_im2, self.data.inverse_class_map[1], (b[k, 0] + 2, b[k, 1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if figure_prefix is '':
            fig, ax = plt.subplots(1)
            ax.imshow(orig_im)
            fig, ax = plt.subplots(1)
            ax.imshow(orig_dp)
            fig, ax = plt.subplots(1)
            ax.imshow(orig_fl)
            fig, ax = plt.subplots(1)
            ax.imshow(label_on_image)
            fig, ax = plt.subplots(1)
            ax.imshow(orig_im2)
        else:
            self.save_image(orig_im, figure_prefix + '_orig_image.jpg')
            self.save_image(orig_dp, figure_prefix + '_orig_depth.jpg')
            self.save_image(orig_fl, figure_prefix + '_orig_flow.jpg')
            self.save_image(label_on_image, figure_prefix + '_label.jpg')
            self.save_image(orig_im2, figure_prefix + '_image_with_box.jpg')

    def visualize_box(self, orig_image, orig_depth, orig_flow, boxes, loss, figure_prefix=''):
        orig_im = orig_image[0, :, :, :].transpose(1, 2, 0)
        im_height, im_width = orig_im.shape[0], orig_im.shape[1]
        assert(im_height == self.data.orig_im_size[0])
        assert(im_width == self.data.orig_im_size[1])

        # Plot original large image with bounding box
        boxes = boxes[0]
        if len(boxes) > 0:
            b = boxes.copy()
            sc = b[:, 4]
            b[:, 0], b[:, 2] = b[:, 0] * im_width, b[:, 2] * im_width
            b[:, 1], b[:, 3] = b[:, 1] * im_width, b[:, 3] * im_width
            b = b.astype(np.int)
        else:
            b = []

        orig_im = orig_im * 255.0
        orig_im = orig_im.astype(np.uint8).copy()
        if len(b) > 0:
            for i in range(b.shape[0]):
                cv2.rectangle(orig_im, (b[i, 0], b[i, 1]), (b[i, 2], b[i, 3]), (0, 255, 0), 3)
                cv2.putText(orig_im, '%.4f' % sc[i], (b[i, 0] + 2, b[i, 1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Plot original depth with bounding box
        orig_dp = orig_depth[0, 0, :, :]
        orig_dp = flowlib.visualize_disp(orig_dp)
        orig_dp = orig_dp * 255.0
        orig_dp = orig_dp.astype(np.uint8).copy()
        if len(b) > 0:
            for i in range(b.shape[0]):
                cv2.rectangle(orig_dp, (b[i, 0], b[i, 1]), (b[i, 2], b[i, 3]), (255, 255, 255), 3)
                cv2.putText(orig_dp, '%.4f' % sc[i], (b[i, 0] + 2, b[i, 1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Plot original flow with bounding box
        orig_fl = orig_flow[0, :, :, :].transpose(1, 2, 0)
        orig_fl = flowlib.visualize_flow(orig_fl)
        orig_fl = orig_fl.astype(np.uint8).copy()
        if len(b) > 0:
            for i in range(b.shape[0]):
                cv2.rectangle(orig_fl, (b[i, 0], b[i, 1]), (b[i, 2], b[i, 3]), (0, 0, 0), 3)
                cv2.putText(orig_fl, '%.4f' % sc[i], (b[i, 0] + 2, b[i, 1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        if figure_prefix is '':
            fig, ax = plt.subplots(1)
            ax.imshow(orig_im)
            fig, ax = plt.subplots(1)
            ax.imshow(orig_dp)
            fig, ax = plt.subplots(1)
            ax.imshow(orig_fl)
            fig, ax = plt.subplots(1)
            ax.imshow(label_on_image)
        else:
            self.save_image(orig_im, figure_prefix + '_pred_orig_image_loss_%.4f.jpg' % loss)
            self.save_image(orig_dp, figure_prefix + '_pred_orig_depth_loss_%.4f.jpg' % loss)
            self.save_image(orig_fl, figure_prefix + '_pred_orig_flow_loss_%.4f.jpg' % loss)

    @staticmethod
    def save_image(image, image_name):
        if np.max(image) <= 1.01:
            image = image * 255.0
        if not image.dtype == np.uint8:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image.save(image_name)
