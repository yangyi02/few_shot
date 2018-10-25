
    def visualize(self, images, orig_image, depths, orig_depth, flows, orig_flow, label, orig_label, idx=None):
        if idx is None:
            idx = 0
        orig_im = orig_image[idx, :, :, :].transpose(1, 2, 0)
        orig_dp = orig_depth[idx, 0, :, :]
        orig_fl = orig_flow[idx, :, :, :].transpose(1, 2, 0)
        orig_lb = orig_label[idx, :, :]

        fig, ax = plt.subplots(1)
        ax.imshow(orig_im)
        plt.show()
        fig, ax = plt.subplots(1)
        orig_dp = flowlib.visualize_disp(orig_dp)
        ax.imshow(orig_dp)
        plt.show()
        fig, ax = plt.subplots(1)
        orig_fl = flowlib.visualize_flow(orig_fl)
        ax.imshow(orig_fl)
        plt.show()
        fig, ax = plt.subplots(1)
        orig_lb = vdrift.visualize_seg(orig_lb, self.inverse_color_map)
        ax.imshow(orig_lb)
        plt.show()
        # construct a full image containing all scale images
        max_im_size = np.max(np.array(self.im_widths))
        sum_im_size = np.sum(np.array(self.im_heights))
        im_all = np.zeros((sum_im_size, max_im_size, 3))
        dp_all = np.zeros((sum_im_size, max_im_size, 3))
        fl_all = np.zeros((sum_im_size, max_im_size, 3))
        cnt = 0
        for i in range(len(images)):
            im = images[i][idx, :, :, :].transpose(1, 2, 0)
            height, width = im.shape[0], im.shape[1]
            im_all[cnt:cnt + height, 0:width, :] = im
            dp = depths[i][idx, 0, :, :]
            dp = flowlib.visualize_disp(dp)
            dp_all[cnt:cnt + height, 0:width, :] = dp
            fl = flows[i][idx, :, :, :].transpose(1, 2, 0)
            fl = flowlib.visualize_flow(fl)
            fl_all[cnt:cnt + height, 0:width, :] = fl
            cnt = cnt + height
        fig, ax = plt.subplots(1)
        ax.imshow(im_all)
        plt.show()
        fig, ax = plt.subplots(1)
        ax.imshow(dp_all)
        plt.show()
        fig, ax = plt.subplots(1)
        ax.imshow(fl_all.astype(np.uint8))
        plt.show()

        seg = vdrift.visualize_seg(label[idx, :, :], self.inverse_color_map)
        fig, ax = plt.subplots(1)
        ax.imshow(seg.astype(np.uint8))
        plt.show()

        seg_on_image = self.visualize_seg_on_image(im_all, seg)
        fig, ax = plt.subplots(1)
        ax.imshow(seg_on_image)
        plt.show()

    def visualize_seg_on_image(self, image, seg):
        im_height, im_width = image.shape[0], image.shape[1]
        seg = cv2.resize(seg, (im_width, im_height), interpolation=cv2.INTER_NEAREST)
        if np.max(image) > 1.01:
            image = image / 255.0
        if np.max(seg) > 1.01:
            seg = seg / 255.0
        # seg = vdrift.visualize_seg(seg, self.inverse_color_map)
        seg_on_image = image * 0.5 + seg * 0.5
        return seg_on_image
