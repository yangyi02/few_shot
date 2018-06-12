import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class SoftAttnInterface(object):
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

    def train(self):
        self.model.train()
        torch.set_grad_enabled(True)
        writer = SummaryWriter(self.tensorboard_path)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_loss_all, train_acc_all = [], []
        index, cnt = np.random.permutation(len(self.data.train_anno['img'])), 0
        for it in range(self.train_iter):
            im, orig_im, dp, orig_dp, box, direction, label, cnt, restart = \
                self.data.get_next_batch('train', cnt, index)
            im = [torch.from_numpy(i_s).float().to(self.device) for i_s in im]
            dp = [torch.from_numpy(d_s).float().to(self.device) for d_s in dp]
            direction = torch.from_numpy(direction).float().to(self.device)
            label = torch.from_numpy(label).long().to(self.device)

            optimizer.zero_grad()
            pred, attn = self.model(im, dp, direction)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            _, pred_label = torch.max(pred, 1)
            acc = (pred_label == label).sum().float() / label.size(0)

            writer.add_scalar('train_loss', loss, it)
            train_loss_all.append(loss)
            if len(train_loss_all) > 100:
                train_loss_all.pop(0)
            ave_train_loss = sum(train_loss_all) / float(len(train_loss_all))
            logging.info('iteration %d, train loss: %.2f, average train loss: %.2f', it, loss, ave_train_loss)
            writer.add_scalar('train_acc', acc, it)
            train_acc_all.append(acc)
            if len(train_acc_all) > 100:
                train_acc_all.pop(0)
            ave_train_acc = sum(train_acc_all) / float(len(train_acc_all))
            logging.info('iteration %d, train accuracy: %.2f, average train accuracy: %.2f', it, acc, ave_train_acc)
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
        logging.info('current best test accuracy: %.2f', self.best_test_acc)
        return test_loss, test_acc

    def test(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        criterion = nn.CrossEntropyLoss()
        test_loss_all, test_acc_all = [], []
        index, cnt = np.random.permutation(len(self.data.test_anno['img'])), 0
        for it in range(self.test_iter):
            im, orig_im, dp, orig_dp, box, direction, label, cnt, _ = \
                self.data.get_next_batch('test', cnt, index)
            im = [torch.from_numpy(i_s).float().to(self.device) for i_s in im]
            dp = [torch.from_numpy(d_s).float().to(self.device) for d_s in dp]
            direction = torch.from_numpy(direction).float().to(self.device)
            label = torch.from_numpy(label).long().to(self.device)

            pred, attn = self.model(im, dp, direction)
            loss = criterion(pred, label)
            _, pred_label = torch.max(pred, 1)
            acc = (pred_label == label).sum().float() / label.size(0)

            test_loss_all.append(loss)
            if len(test_loss_all) > 100:
                test_loss_all.pop(0)
            test_acc_all.append(acc)
            if len(test_acc_all) > 100:
                test_acc_all.pop(0)

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
        cnt = 0
        while True:
            im, orig_im, dp, orig_dp, box, direction, label, cnt, restart = \
                self.data.get_next_batch('test', cnt)
            if restart:
                break
            im = [torch.from_numpy(i_s).float().to(self.device) for i_s in im]
            dp = [torch.from_numpy(d_s).float().to(self.device) for d_s in dp]
            direction = torch.from_numpy(direction).float().to(self.device)
            label = torch.from_numpy(label).long().to(self.device)

            pred, attn = self.model(im, dp, direction)
            loss = criterion(pred, label)
            _, pred_label = torch.max(pred, 1)
            acc = (pred_label == label).sum().float() / label.size(0)
            test_loss_all.append(loss)
            test_acc_all.append(acc)

            logging.info('at instance %d, test loss: %.2f, test accuracy: %.2f', cnt, loss, acc)

        test_loss = np.mean(np.array(test_loss_all))
        test_acc = np.mean(np.array(test_acc_all))
        logging.info('overall average test loss: %.2f', test_loss)
        logging.info('overall average test accuracy: %.2f', test_acc)

    def predict(self, image, depth, direction):
        self.model.eval()
        torch.set_grad_enabled(False)
        im = [torch.from_numpy(i_s).float().to(self.device) for i_s in image]
        dp = [torch.from_numpy(d_s).float().to(self.device) for d_s in depth]
        direction = torch.from_numpy(direction).float().to(self.device)
        pred, attn = self.model(im, dp, direction)
        pred = pred.cpu().numpy()
        attn = attn.cpu().numpy()
        return pred, attn

    def visualize_all(self, image_list, figure_path):
        lines = open(image_list).readlines()
        for line in lines:
            line = line.strip()
            dirs = line.split('/')
            sub_dir, image_name = dirs[-2], dirs[-1]
            file_name, file_ext = os.path.splitext(image_name)
            file_id = file_name.split('_')[0]
            image_name = os.path.join(self.data.img_dir, sub_dir, file_id + '_color.jpg')
            depth_name = os.path.join(self.data.depth_dir, sub_dir, file_id + '_depth.png')
            box_name = os.path.join(self.data.box_dir, sub_dir, file_id + '.txt')
            self.visualize(image_name, depth_name, box_name, figure_path)

    def visualize(self, image_name, depth_name, box_name, figure_path):
        im, orig_im, dp, orig_dp, box, direction, label = \
            self.data.get_one_sample(image_name, depth_name, box_name)
        pred, attn = self.predict(im, dp, direction)
        if figure_path is '':
            self.visualize_groundtruth(orig_im, orig_dp, box, direction, label)
            self.visualize_attn(im, dp, attn, pred)
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
            self.visualize_groundtruth(orig_im, orig_dp, box, direction, label, figure_prefix)
            self.visualize_attn(im, dp, attn, pred, figure_prefix)

    def visualize_attn(self, images, depths, attn, prediction, figure_prefix=''):
        num_instance = prediction.shape[0]
        # reconstruct attention maps in multiple scales from a single vector
        attn_maps, cnt = [], 0
        for i in range(len(images)):
            a = attn[:, cnt:cnt + self.model.attn_size[i] ** 2]
            a = a.reshape(-1, self.model.attn_size[i], self.model.attn_size[i])
            cnt = cnt + self.model.attn_size[i] ** 2
            attn_maps.append(a)
        # construct a full image containing all scale images and
        # a full attention map containing all scale attention maps
        max_im_size = np.max(np.array(self.data.im_size))
        sum_im_size = np.sum(np.array(self.data.im_size))
        im_all = np.zeros((max_im_size, sum_im_size, 3))
        dp_all = np.zeros((max_im_size, sum_im_size, 3))
        max_attn_size = np.max(np.array(self.model.attn_size))
        sum_attn_size = np.sum(np.array(self.model.attn_size))
        attn_all = np.zeros((max_attn_size, sum_attn_size))
        for n in range(num_instance):
            pred = prediction[n, :]
            pred_label = np.argmax(pred)
            print('prediction:', pred, pred_label, self.data.inverse_class_map[pred_label])

            cnt_im, cnt_attn = 0, 0
            for i in range(len(images)):
                im = images[i][n, :, :, :].transpose(1, 2, 0)
                height, width = im.shape[0], im.shape[1]
                im_all[0:height, cnt_im:cnt_im + width, :] = im

                dp = depths[i][n, 0, :, :]
                dp = (dp - np.min(dp)) / (np.max(dp) - np.min(dp))
                cmap = plt.get_cmap('gist_rainbow')
                dp = cmap(dp)
                dp = dp[:, :, 0:3]
                dp_all[0:height, cnt_im:cnt_im + width, :] = dp
                cnt_im = cnt_im + width

                attn = attn_maps[i][n, :, :]
                height, width = attn.shape[0], attn.shape[1]
                attn_all[0:height, cnt_attn:cnt_attn + width] = attn
                cnt_attn = cnt_attn + width

            attn_on_image = self.visualize_attention_on_image(im_all, attn_all)


            if figure_prefix is '':
                fig, ax = plt.subplots(1)
                ax.imshow(im_all)
                fig, ax = plt.subplots(1)
                ax.imshow(dp_all)
                fig, ax = plt.subplots(1)
                ax.imshow(attn_all)
                fig, ax = plt.subplots(1)
                ax.imshow(attn_on_image)
            else:
                self.save_image(im_all, figure_prefix + '_' + str(n) + '_image_all.jpg')
                self.save_image(dp_all, figure_prefix + '_' + str(n) + '_depth_all.jpg')
                self.save_image(attn_on_image, figure_prefix + '_' + str(n) + '_attn_on_image.jpg')

    @staticmethod
    def visualize_attention_on_image(image, attention):
        im_height, im_width = image.shape[0], image.shape[1]
        attn = cv2.resize(attention, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
        attn = np.dstack((attn, attn, attn))
        attn = (attn - np.min(attn)) / (np.max(attn) - np.min(attn))
        attn_on_image = image * 0.2 + attn * 0.8
        return attn_on_image

    def visualize_groundtruth(self, orig_image, orig_depth, box, direction, label, figure_prefix=''):
        num_instance = label.shape[0]
        for n in range(num_instance):
            d = direction[n, :]
            l = label[n]
            print('groundtruth:', d, l, self.data.inverse_class_map[l])

            orig_im = orig_image[n, :, :, :].transpose(1, 2, 0)
            im_height, im_width = orig_im.shape[0], orig_im.shape[1]

            # Plot original large image with bounding box
            b = box[n, 0:4].copy()
            b[0], b[1], b[2], b[3] = b[0] * im_width, b[1] * im_height, b[2] * im_width, b[3] * im_height
            b = b.astype(np.int)

            orig_im = orig_im * 255.0
            orig_im = orig_im.astype(np.uint8).copy()
            cv2.rectangle(orig_im, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
            cv2.putText(orig_im, self.data.inverse_class_map[l], (b[0] + 2, b[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Plot original depth with bounding box
            orig_dp = orig_depth[n, 0, :, :]
            orig_dp = (orig_dp - np.min(orig_dp)) / (np.max(orig_dp) - np.min(orig_dp))
            cmap = plt.get_cmap('gist_rainbow')
            orig_dp = cmap(orig_dp)
            orig_dp = orig_dp[:, :, 0:3]
            orig_dp = orig_dp * 255.0
            orig_dp = orig_dp.astype(np.uint8).copy()
            cv2.rectangle(orig_dp, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 3)
            cv2.putText(orig_dp, self.data.inverse_class_map[l], (b[0] + 2, b[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if figure_prefix is '':
                fig, ax = plt.subplots(1)
                ax.imshow(orig_im)
                fig, ax = plt.subplots(1)
                ax.imshow(orig_dp)
            else:
                self.save_image(orig_im, figure_prefix + '_' + str(n) + '_orig_image.jpg')
                self.save_image(orig_dp, figure_prefix + '_' + str(n) + '_orig_depth.jpg')

    @staticmethod
    def save_image(image, image_name):
        if np.max(image) <= 1.01:
            image = image * 255.0
        if not image.dtype == np.uint8:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image.save(image_name)
