import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import flowlib

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class SegmentInterface(object):
    def __init__(self, train_data, test_data, model, learning_rate, train_epoch, test_interval,
                 test_iteration, save_interval, init_model_path, save_model_path, tensorboard_path):
        self.train_dataloader = train_data
        self.test_dataloader = test_data
        self.model = model
        self.learning_rate = learning_rate
        self.train_epoch = train_epoch
        self.test_interval = test_interval
        self.test_iteration = test_iteration
        self.save_interval = save_interval
        self.best_test_acc = 0
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

    def to_tensor(self, sample):
        im = sample['images']
        im = [torch.from_numpy(i_s.transpose((0, 3, 1, 2))).float().to(self.device) for i_s in im]
        dp = sample['depths']
        dp = [torch.from_numpy(d_s.transpose((0, 3, 1, 2))).float().to(self.device) for d_s in dp]
        fl = sample['flows']
        fl = [torch.from_numpy(f_s.transpose((0, 3, 1, 2))).float().to(self.device) for f_s in fl]
        lb = sample['seg']
        lb = torch.from_numpy(lb).long().to(self.device)
        return im, dp, fl, lb

    def train(self):
        self.model.train()
        torch.set_grad_enabled(True)
        writer = SummaryWriter(self.tensorboard_path)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_loss_all, train_acc_all = [], []
        for epoch in range(self.train_epoch):
            self.train_dataloader.shuffle()
            for i in range(self.train_dataloader.num_batch):
                sample = self.train_dataloader.get_next_batch(i)
                im, dp, fl, label = self.to_tensor(sample)

                optimizer.zero_grad()
                pred = self.model(im, dp, fl)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()

                _, pred_label = torch.max(pred, 1)
                acc = (pred_label == label).sum().float() / label.numel()
                it = epoch * self.train_dataloader.num_batch + i
                writer.add_scalar('train_loss', loss, it)
                train_loss_all.append(loss)
                if len(train_loss_all) > 100:
                    train_loss_all.pop(0)
                ave_train_loss = sum(train_loss_all) / float(len(train_loss_all))
                logging.info('epoch %d, batch %d, train loss: %.2f, average train loss: %.2f',
                             epoch, i, loss, ave_train_loss)
                writer.add_scalar('train_acc', acc, it)
                train_acc_all.append(acc)
                if len(train_acc_all) > 100:
                    train_acc_all.pop(0)
                ave_train_acc = sum(train_acc_all) / float(len(train_acc_all))
                logging.info('epoch %d, batch %d, train acc: %.2f, average train acc: %.2f',
                             epoch, i, acc, ave_train_acc)
                if (it + 1) % self.save_interval == 0:
                    logging.info('epoch %d, iteration %d, saving model', epoch, it)
                    with open(self.save_model_path, 'w') as handle:
                        torch.save(self.model.state_dict(), handle)
                if (it + 1) % self.test_interval == 0:
                    logging.info('epoch %d, iteration %d, testing', epoch, it)
                    test_loss, test_acc = self.validate()
                    writer.add_scalar('test_loss', test_loss, it)
                    writer.add_scalar('test_acc', test_acc, it)
                    self.model.train()
                    torch.set_grad_enabled(True)
            with open(self.save_model_path, 'w') as handle:
                torch.save(self.model.state_dict(), handle)
            logging.info('finish epoch %d', epoch)
            logging.info('model save to %s', self.save_model_path)
        writer.close()

    def validate(self):
        test_loss, test_acc = self.test()
        if test_acc >= self.best_test_acc:
            with open(self.save_model_path, 'w') as handle:
                torch.save(self.model.state_dict(), handle)
            logging.info('model save to %s', self.save_model_path)
            self.best_test_acc = test_acc
        logging.info('current best test accuracy: %.2f', self.best_test_acc)
        return test_loss, test_acc

    def test(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        criterion = nn.CrossEntropyLoss()
        test_loss_all, test_acc_all = [], []
        self.test_dataloader.shuffle()
        for i in range(self.test_iteration):
            sample = self.test_dataloader.get_next_batch(i)
            im, dp, fl, label = self.to_tensor(sample)

            pred = self.model(im, dp, fl)
            loss = criterion(pred, label)
            _, pred_label = torch.max(pred, 1)
            acc = (pred_label == label).sum().float() / label.numel()

            test_loss_all.append(loss)
            test_acc_all.append(acc)

        test_loss = np.mean(np.array(test_loss_all))
        test_acc = np.mean(np.array(test_acc_all))
        logging.info('average test loss: %.2f', test_loss)
        logging.info('average test accuracy: %.2f', test_acc)
        return test_loss, test_acc

    def test_all(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        criterion = nn.CrossEntropyLoss()
        test_loss_all, test_acc_all = [], []
        for i in range(self.test_dataloader.num_batch):
            sample = self.test_dataloader.get_next_batch(i)
            im, dp, fl, label = self.to_tensor(sample)

            pred = self.model(im, dp, fl)
            loss = criterion(pred, label)
            _, pred_label = torch.max(pred, 1)
            acc = (pred_label == label).sum().float() / label.numel()

            test_loss_all.append(loss)
            test_acc_all.append(acc)

        test_loss = np.mean(np.array(test_loss_all))
        test_acc = np.mean(np.array(test_acc_all))
        logging.info('overall average test loss: %.2f', test_loss)
        logging.info('overall average test accuracy: %.2f', test_acc)

    def predict(self, image, depth, flow, label):
        self.model.eval()
        torch.set_grad_enabled(False)
        criterion = nn.CrossEntropyLoss()
        im = [torch.from_numpy(i_s).float().to(self.device) for i_s in image]
        dp = [torch.from_numpy(d_s).float().to(self.device) for d_s in depth]
        fl = [torch.from_numpy(f_s).float().to(self.device) for f_s in flow]
        label = torch.from_numpy(label).long().to(self.device)
        pred = self.model(im, dp, fl)
        loss = criterion(pred, label)
        _, pred_label = torch.max(pred, 1)
        acc = (pred_label == label).sum().float() / label.numel()
        pred = pred.cpu().numpy()
        pred_label = pred_label.cpu().numpy()
        return pred, pred_label, acc, loss

    def visualize_all(self, image_list, figure_path):
        lines = open(image_list).readlines()
        for line in lines:
            line = line.strip()
            dirs = line.split('/')
            image_name = dirs[-1]
            file_id, file_ext = os.path.splitext(image_name)
            image_name = os.path.join(self.data.image_dir, file_id + '.png')
            depth_name = os.path.join(self.data.depth_dir, file_id + '.png')
            flow_name = os.path.join(self.data.flow_dir, file_id + '.png')
            seg_name = os.path.join(self.data.seg_dir, file_id + '.png')
            self.visualize(image_name, depth_name, flow_name, seg_name, figure_path)

    def visualize(self, image_name, depth_name, flow_name, seg_name, figure_path):
        sample = self.test_dataloader.get_one_sample(image_name, depth_name, flow_name, seg_name)
        pred, pred_label, acc, loss = self.predict(im, dp, fl, label)
        if figure_path is '':
            self.visualize_prediction(im, orig_im, dp, fl, pred_label, acc)
            plt.show()
            plt.close('all')
        else:
            if not os.path.exists(figure_path):
                os.makedirs(figure_path)
            dirs = image_name.split('/')
            sub_dir, image_name = dirs[-2], dirs[-1]
            file_id, file_ext = os.path.splitext(image_name)
            figure_prefix = os.path.join(figure_path, sub_dir + '_' + file_id)
            self.visualize_prediction(im, orig_im, dp, fl, pred_label, acc, figure_prefix)

    def visualize_prediction(self, images, orig_im, depths, flows, pred_label, acc, figure_prefix=''):
        # construct a full image containing all scale images and
        # a full attention map containing all scale attention maps
        max_im_size = np.max(np.array(self.data.im_widths))
        sum_im_size = np.sum(np.array(self.data.im_heights))
        im_all = np.zeros((sum_im_size, max_im_size, 3))
        dp_all = np.zeros((sum_im_size, max_im_size, 3))
        fl_all = np.zeros((sum_im_size, max_im_size, 3))
        cnt_im = 0
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

        pred = pred_label[0, :, :]
        pred = vdrift.visualize_seg(pred, self.data.inverse_color_map)
        orig_im = orig_im[0, :, :, :].transpose(1, 2, 0)
        pred_on_image = self.data.visualize_seg_on_image(orig_im, pred)

        if figure_prefix is '':
            fig, ax = plt.subplots(1)
            ax.imshow(im_all)
            fig, ax = plt.subplots(1)
            ax.imshow(dp_all)
            fig, ax = plt.subplots(1)
            ax.imshow(fl_all)
            fig, ax = plt.subplots(1)
            ax.imshow(pred)
            fig, ax = plt.subplots(1)
            ax.imshow(pred_on_image)
        else:
            # self.save_image(im_all, figure_prefix + '_image_all.jpg')
            # self.save_image(dp_all, figure_prefix + '_depth_all.jpg')
            # self.save_image(fl_all, figure_prefix + '_flow_all.jpg')
            # self.save_image(pred, figure_prefix + '_pred.jpg')
            self.save_image(pred_on_image, figure_prefix + '_acc_%.4f.jpg' % acc)

    @staticmethod
    def save_image(image, image_name):
        if np.max(image) <= 1.01:
            image = image * 255.0
        if not image.dtype == np.uint8:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image.save(image_name)
