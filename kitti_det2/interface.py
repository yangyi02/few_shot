import os
import numpy as np
import matplotlib.pyplot as plt
import detect_loss
from nms import nms
from visualize import visualize_heatmap, visualize_box, save_image

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class DetectInterface(object):
    def __init__(self, data, train_dataloader, test_dataloader, model, learning_rate, train_epoch,
                 test_interval, test_iteration, save_interval, init_model_path, save_model_path,
                 tensorboard_path):
        self.data = data
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
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
        hm = sample['heatmaps']
        hm = [torch.from_numpy(h_s.transpose((0, 3, 1, 2))).float().to(self.device) for h_s in hm]
        of = sample['offsets']
        of = [torch.from_numpy(o_s.transpose((0, 3, 1, 2))).float().to(self.device) for o_s in of]
        return im, dp, fl, hm, of

    @staticmethod
    def get_loss(pred_hm, pred_of, hm, of, criterion_hm, criterion_of):
        loss = 0
        for i in range(len(hm)):
            loss = loss + criterion_hm(pred_hm[i], hm[i])
        # loss = loss * 10
        # for i in range(len(hm)):
        #     mask = hm[i]
        #     mask = mask.float()
        #     loss = loss + criterion_of(pred_of[i], of[i], mask)
        return loss

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
        criterion_hm = detect_loss.BCELoss2d()
        criterion_of = detect_loss.MSELoss2d()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_loss_all, train_acc_all, base_acc_all = [], [], []
        for epoch in range(self.train_epoch):
            self.train_dataloader.shuffle()
            for i in range(self.train_dataloader.num_batch):
                sample = self.train_dataloader.get_next_batch(i)
                im, dp, fl, hm, of = self.to_tensor(sample)
                optimizer.zero_grad()
                pred_hm, pred_of = self.model(im, dp, fl)
                loss = self.get_loss(pred_hm, pred_of, hm, of, criterion_hm, criterion_of)
                loss.backward()
                optimizer.step()

                acc, base_acc = self.get_accuracy(pred_hm, hm)
                it = epoch * self.train_dataloader.num_batch + i
                writer.add_scalar('train_loss', loss, it)
                train_loss_all.append(loss)
                if len(train_loss_all) > 100:
                    train_loss_all.pop(0)
                ave_train_loss = sum(train_loss_all) / float(len(train_loss_all))
                logging.info('epoch %d, batch %d, train loss: %.4f, average train loss: %.4f',
                             epoch, i, loss, ave_train_loss)
                writer.add_scalar('train_acc', acc, it)
                train_acc_all.append(acc)
                if len(train_acc_all) > 100:
                    train_acc_all.pop(0)
                ave_train_acc = sum(train_acc_all) / float(len(train_acc_all))
                logging.info('epoch %d, batch %d, train acc: %.4f, average train acc: %.4f',
                             epoch, i, acc, ave_train_acc)
                base_acc_all.append(base_acc)
                if len(base_acc_all) > 100:
                    base_acc_all.pop(0)
                ave_base_acc = sum(base_acc_all) / float(len(base_acc_all))
                logging.info('epoch %d, batch %d, base acc: %.4f, average base acc: %.4f',
                             epoch, i, base_acc, ave_base_acc)
                if (it + 1) % self.save_interval == 0:
                    logging.info('epoch %d, batch %d, saving model', epoch, it)
                    with open(self.save_model_path, 'w') as handle:
                        torch.save(self.model.state_dict(), handle)
                if (it + 1) % self.test_interval == 0:
                    logging.info('epoch %d, batch %d, testing', epoch, it)
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
        logging.info('current best test accuracy: %.4f', self.best_test_acc)
        return test_loss, test_acc

    def test(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        criterion_hm = detect_loss.BCELoss2d()
        criterion_of = detect_loss.MSELoss2d()
        test_loss_all, test_acc_all, base_acc_all = [], [], []
        self.test_dataloader.shuffle()
        for i in range(self.test_iteration):
            sample = self.test_dataloader.get_next_batch(i)
            im, dp, fl, hm, of = self.to_tensor(sample)
            pred_hm, pred_of = self.model(im, dp, fl)
            loss = self.get_loss(pred_hm, pred_of, hm, of, criterion_hm, criterion_of)
            acc, base_acc = self.get_accuracy(pred_hm, hm)
            test_loss_all.append(loss)
            test_acc_all.append(acc)
            base_acc_all.append(base_acc)
            # logging.info('batch %d, test loss: %.4f', i, loss)
            # logging.info('batch %d, test accuracy: %.4f', i, acc)
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
        criterion_hm = detect_loss.BCELoss2d()
        criterion_of = detect_loss.MSELoss2d()
        test_loss_all, test_acc_all, base_acc_all = [], [], []
        for i in range(self.test_dataloader.num_batch):
            sample = self.test_dataloader.get_next_batch(i)
            im, dp, fl, hm, of = self.to_tensor(sample)
            pred_hm, pred_of = self.model(im, dp, fl)
            loss = self.get_loss(pred_hm, pred_of, hm, of, criterion_hm, criterion_of)
            acc, base_acc = self.get_accuracy(pred_hm, hm)
            test_loss_all.append(loss)
            test_acc_all.append(acc)
            base_acc_all.append(base_acc)
            logging.info('batch %d, test loss: %.4f', i, loss)
            logging.info('batch %d, test accuracy: %.4f', i, acc)
            logging.info('batch %d, base accuracy: %.4f', i, base_acc)
        test_loss = np.mean(np.array(test_loss_all))
        test_acc = np.mean(np.array(test_acc_all))
        base_acc = np.mean(np.array(base_acc_all))
        logging.info('overall average test loss: %.4f', test_loss)
        logging.info('overall average test accuracy: %.4f', test_acc)
        logging.info('overall average base accuracy: %.4f', base_acc)

    def predict(self, im, dp, fl, hm, of):
        self.model.eval()
        torch.set_grad_enabled(False)
        pred_hm, pred_of = self.model(im, dp, fl)
        criterion_hm = detect_loss.BCELoss2d()
        criterion_of = detect_loss.MSELoss2d()
        loss = self.get_loss(pred_hm, pred_of, hm, of, criterion_hm, criterion_of)
        acc, _ = self.get_accuracy(pred_hm, hm)
        for i in range(len(pred_hm)):
            pred_hm[i] = torch.sigmoid(pred_hm[i])
            pred_hm[i] = pred_hm[i].cpu().numpy().transpose((0, 2, 3, 1))
            pred_of[i] = pred_of[i].cpu().numpy().transpose((0, 2, 3, 1))
        return pred_hm, pred_of, loss, acc

    def visualize(self, image_name, depth_name, flow_name, box_name, figure_path):
        sample = self.test_dataloader.get_one_sample(image_name, depth_name, flow_name, box_name)
        im, dp, fl, hm, of = self.to_tensor(sample)
        pred_hm, pred_of, loss, acc = self.predict(im, dp, fl, hm, of)
        pred_box = nms(pred_hm, pred_of, self.test_dataloader.orig_im_size[0],
                       self.test_dataloader.orig_im_size[1])
        heatmap = visualize_heatmap(sample, heatmaps=pred_hm, display=False)
        image, depth, flow = visualize_box(sample, boxes=pred_box, display=False)

        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        dirs = image_name.split('/')
        sub_dir, image_name = dirs[-2], dirs[-1]
        file_name, file_ext = os.path.splitext(image_name)
        file_id = file_name.split('_')[0]
        figure_prefix = os.path.join(figure_path, sub_dir + '_' + file_id)
        save_image(heatmap, figure_prefix + '_heatmap_%.4f_%.4f.jpg' % (loss, acc))
        save_image(image, figure_prefix + '_image_%.4f_%.4f.jpg' % (loss, acc))
        save_image(depth, figure_prefix + '_depth_%.4f_%.4f.jpg' % (loss, acc))
        save_image(flow, figure_prefix + '_flow_%.4f_%.4f.jpg' % (loss, acc))

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
