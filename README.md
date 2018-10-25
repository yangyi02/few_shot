# 10/25
**Project**: Do unsupervised low-level depth and optical flow help supervised high-level visual understanding?
- **Context**: Try to understand how potentially unsupervised learned depth and optical flow can help object recognition and detection. Either Use ground truth depth and flow or use stereo video pairs (4 frames as input) from Wang Yang's algorithm to produce the truly unsupervised signal. Then combine RGB, depth, and flow as input signals to check. Conduct experiments on MLT, VDrift, KITTI, and CityScape dataset.
- **Owner**: Yi Yang, Yang Wang, Liang Zhao
- **Progress**: 1. Submit 3D fine-grained paper to Arxiv. 2. Wrap up code

# 10/18
- **Owner**: Yi Yang, Yang Wang, Liang Zhao
- **Progress**: 1. Train unsupervised flow on CityScape dataset. 2. Use unsupervised learned optical flow as extra input, the performance does not improve yet.

# 10/11
- **Progress**: Verify that unsupervised disparity indeed helps semantic image segmentation on CityScape dataset. What is more interesting is that, using both left right image as input does not help segmentation results.
- **Todo**: 1. Testing accuracy v.s. #train data. 2. Optical flow as extra input 3. 
![图片](http://agroup-bos.cdn.bcebos.com/0cfd60bb3dc0e561bef33c36aa6e89cc2687eee6 =150x)![图片](http://agroup-bos.cdn.bcebos.com/c77d48f379243aae260c37c0b0b95d50d3c41758 =300x)

# 10/4
- **Progress**: Train semantic segmentation on CityScape dataset, using existing disparity helps.
![图片](http://agroup-bos.cdn.bcebos.com/977de16e7f2a0e8e72e3f3bc9e7d4618dc709955 =300x)![图片](http://agroup-bos.cdn.bcebos.com/9b6e230e3261b60c39869b6a23346af6c92b0448 =300x)

# 9/27
- **Progress**: Fix bugs in unsupervised training data provider for Cityscape, now training stereo disparity estimation model, EPE looks reasonable now.
![图片](http://agroup-bos.cdn.bcebos.com/268d2e650ca0d019fe462aaa7f8606182582507c =600x)
![图片](http://agroup-bos.cdn.bcebos.com/f2d82fb4a68889d8af20e7a171328470009642a6 =300x)

**Project 2**: Improved Annotation for 3D fine-grained pose dataset
- **Context**: Use image segmentation to refine the 3D pose annotation. Results look promising.
- **Owner**: Yi Yang, Feng Zhou
- **Progress**: Finish producing annotation for both StanfordCars and FGVC-Aircraft data. Working on submitting the camera-ready submission.

# 9/20
- **Progress**: Finish unsupervised training data provider for CityScape, writing testing data provider. 
- **Progress**: Finish producing annotation for StanfordCars training data and FGVC-Aricraft training/testing data, now continuing on StanfordCars testing data.

# 9/13
- **Progress**: Train semantic segmentation on VDrift dataset, using RGB 89% testing accuracy, using RGBD 91% testing accuracy. 
![图片](http://agroup-bos.cdn.bcebos.com/1c04ccbd04d182d205ec1402df6a9d80cd076b64 =300x)![图片](http://agroup-bos.cdn.bcebos.com/aa6a4d59bf2aeb51fb2cf3c271ff1cfe9b78494b =300x)
- **Progress**: Finish producing annotation for StanfordCars and FGVC-Aricraft training data, now continuing on testing data.

# 9/6
- **Progress**: Organize code for KITTI detection, able to run on AI cluster, download all required datasets to AI cluster. 
- **Progress**: Continue working on the writing of 3D fine-grained object pose estimation dataset report.

# 8/30
- **Progress**: Downloaded all CityScape video dataset, waiting for Wang yang's unsupervised code to be ready.
- **Progress**: Paper accepted at ECCV workshop. Continue working on the writing of 3D fine-grained object pose estimation for CVPR. 

# 8/23
- **Progress**: Verified: adding unsupervised depth (from Wang Yang) is helpful in lowering detection loss.
- Compare two experimental settings:
1. Kitti car detection. Use 5984 training images and 1497 testing images, train 3000 iterations, same VGG-type network structure. Adding unsupervised depth decreases the detection testing loss.
![图片](http://agroup-bos.cdn.bcebos.com/06cb125668b5ef1f4694081b4eb70d82c9eb5c84 =100x)![图片](http://agroup-bos.cdn.bcebos.com/963d5f0fc7d514abb1bbc1415d2d3c851b75505b =300x)![图片](http://agroup-bos.cdn.bcebos.com/63bf4637037a694adc4aaab4c82cadb452b68e9c =300x)
2. Kitti car few shot detection. Use 598 training images and 1497 testing images, train 300 iterations, same VGG-type network structure. Adding unsupervised depth even more significantly decreases the detection testing loss.
![图片](http://agroup-bos.cdn.bcebos.com/8319d1b2bd4f206f9beed17f6dee53a5280fdba8 =100x)![图片](http://agroup-bos.cdn.bcebos.com/f42ed0e6ba1f3cae839b78ba126e79bcc17ea991 =300x)![图片](http://agroup-bos.cdn.bcebos.com/6a1a7396bc5200a3b226727b58a8ea14d5e9d75d =300x)
|image | unsup_depth | RGB | RGB+UnsupD |
| ----- | ---- | ----- | ---- |
|![图片](http://agroup-bos.cdn.bcebos.com/b6e289ff093401c60d12fbce6ef8face8a33b21c =200x)|![图片](http://agroup-bos.cdn.bcebos.com/c0fb5a45a5de9a3799f13389af1d9114170ad059 =200x)|![图片](http://agroup-bos.cdn.bcebos.com/fb20121d038d5427f918faf67fd8676d854f1238 =200x)|![图片](http://agroup-bos.cdn.bcebos.com/b92c46bf7c2dd90fa3af0eb08df149afa9fc289b =200x)

# 8/16
- **Progress**:
	- Manage to train a preliminary RGB based object detection on KITTI car detection. 
	- Adding unsupervised depth (from Wang Yang) seems helpful in lowering detection loss but needs further verification.
	- So far both models in RGB and RGBD does not overfit on testing data.
- **Todo**
	- Will enlarge the model complexity to reach the extreme for each model.
	- Will make detection results more through so can be evaluated using mAP.

# 8/9
- After studying the dataset distribution on KITTI, decide to work with KITTI detection problem instead of VQA problem. The main reason is the class is dominated by cars which brings trouble for the model to learn a good generalization on locations to recognize other objects. 
- Modify Faster RCNN code it make it runnable on PyTorch 0.4.0. Test the network training on PASCAL VOC 2007 dataset. The prediction looks reasonable and loss converges smoothly. At the stage of writing KITTI data provider for Faster RCNN code.
![图片](http://agroup-bos.su.bcebos.com/fe9fdf8fa3787c579c520ecf30d20d26e02a94f7 =300x)![图片](http://agroup-bos.su.bcebos.com/72f2db5a7e7d42dc379af6769ee90789d33908d6 =300x)![图片](http://agroup-bos.su.bcebos.com/7aa41096a28669cf7b9bb9b02837a160754fa00b =300x)![图片](http://agroup-bos.su.bcebos.com/737fe7bad293023f6103cc15a37da59926180110 =300x)![图片](http://agroup-bos.su.bcebos.com/1c3ccb7d7ccfd4df2b127d8b8cfb9ad51530d832 =300x) ![图片](http://agroup-bos.su.bcebos.com/4b4b33d8494242e361f64b4cc70319920cf47e73 =300x)

# 8/2
- Involve top-down location guidance on KITTI attention VQA. With top-down location guidance, RGB recognition accuracy improve from 81% to 86% on testing, RGBD recognition accuracy improve from 81% to 87% on testing. There is so far 1% improvemeent adding unsupervised depth.
- However, both attention maps look not meaniful, both models need more analysis.
- The attention map also suggest that the current experimental setting needs further change. This is because the car class is dominate (75%). Although one can improve to 81% or even 87% accuracy, we shall setup a new experiment on this dataset, such as adding background class.

|image | depth | RGB | RGBD |
| ----- | ---- | ----- | ---- |
|![图片](http://agroup-bos.su.bcebos.com/83948483eff1b8561cc7e369e98c94bbc3f1b631 =200x)|![图片](http://agroup-bos.su.bcebos.com/fe7345eee545f6f8435b876bb12ae14586c73495 =200x)|![图片](http://agroup-bos.su.bcebos.com/db9c1a6d6a4c10607ea1f15a083ce91b43d4c65d =200x)|![图片](http://agroup-bos.su.bcebos.com/b78c9309260520f416997462e8428aff448a1907 =200x)|
|![图片](http://agroup-bos.su.bcebos.com/ecf1ae2cfc883deaac9d7bfd5183eb1790d3dd38 =200x)|![图片](http://agroup-bos.su.bcebos.com/e1fcd4672098a9c40b609709400bc06cbde87a87 =200x)|![图片](http://agroup-bos.su.bcebos.com/3f05dff27fcef1aaf7b5b30f7f8b6fdb007cd35f =200x)|![图片](http://agroup-bos.su.bcebos.com/78be136327b8979a7e1e61fcf63e9363062c1687 =200x)|

# 7/26
- Setup RGB baseline for KITTI object detection. Use the Faster-RCNN model pretrained on Microsoft COCO to detect KITTI car and pedestrians. The model has not been finetuned yet. And there may also be bugs there.
![图片](http://agroup-bos.su.bcebos.com/86c5133bc2f21c9601a97a9ea18d7818c04f3f09 =300x)![图片](http://agroup-bos.su.bcebos.com/f88e30d445cd6d0b780bce9f398616cef297a26d =300x)
- Setup baseline for KITTI attention VQA. Use the previous attention model on KITTI car and pedestrian VQA. Still under training. So far, using both RGB and RGBD gets 81% accuracy on testing. Adding unsupervisedly learned depth shows no improvement yet. I will need more time to analyze the results and get to more sophisticated model.

# 7/19
- Continue working on the writing of 3D fine-grained object pose estimation with Feng Zhou for another ECCV workshop. Use image segmentation to refine the initial pose annotation by our annotators. Results look promising.
- Left column is original pose annotation. Right column is refined pose annotation.
![图片](http://agroup-bos.su.bcebos.com/8676e3a9a1d8db442ce999f9154e929f455807ae =300x)![图片](http://agroup-bos.su.bcebos.com/0aefd616f46e237c7aec3bc6bcb56ba75a167081 =300x)
![图片](http://agroup-bos.su.bcebos.com/828e2022aa634123fad3df8257c4d42a307e9cc2 =300x)![图片](http://agroup-bos.su.bcebos.com/4a95aefe2f29edbe265f81c92bf889664b63a8e2 =300x)

# 7/12
- Finish the writing of zero-shot transfer VQA paper writing with Yuanpeng and submit to ECCV workshop
- Work on the writing of 3D fine-grained object pose estimation with Feng Zhou for another ECCV workshop
- Work with Wang Yang on extracting optical flow, depth and moving object masks on the KITTI object detection dataset. 

# 7/5
- Finish downloading the KITTI object detection dataset with stereo and 3 consecutive frames.
- Continue helping Yuanpeng on zero-shot transfer VQA paper writing. 

# 6/28
- Start working on KITTI object detection with depth and optical flow. So far, still preparing the data for the training and validation.
- Publish feedback neural networks on TPAMI and organize the code.
- Helping Yuanpeng on zero-shot transfer VQA paper writing. 

# 6/21
- One week off

# 6/14
- Finish the co-training pipeline, test it on a simple 2-dimensional Gaussian Mixture data, find it indeed is helpful.
- More specifically, I generate 200 training data for 2 classes with each class 100 training. I only annotate 1 data for each class, so the supervision is very limited. The baseline (blue) is the testing accuracy using only 1 data per class to supervise the classifier. The upper bound (red) is the testing accuracy using all 200 data per class to supervise the classifier. The co-training (green) use 1 labeled data per class and 99 unlabeled data per class to train.
![图片](http://agroup-bos.su.bcebos.com/1ad9f2e42f11c3258819b4cfddadb4264d8ff068 =600x)![图片](http://agroup-bos.su.bcebos.com/02bdff3a67db675bb80d8ac9e543b55c4c657be8 =200x)![图片](http://agroup-bos.su.bcebos.com/2890b7871674fa50f42db3505897f4628e33ce1c =200x)![图片](http://agroup-bos.su.bcebos.com/fd589f764a0c9f0812598257d7e540ba9e2cd42e =200x)

# 6/7
- Adding more baseline comparisons according to last week's discussion. Starting to study co-training for weakly supervised multi-class classification. In order to do co-training, the networks need to have two separate branches with each branch has the ability for classification. Hence I modify the network structure from previously using a CNN based on 4 channel input (RGBD) to two-stream CNN with one stream based on 3 channel RGB and another stream based on 1 channel depth. 
- It seems the two-stream model works even better.
![图片](http://agroup-bos.su.bcebos.com/efda6e1e53c6a7ea4aa3ba8d0d83ba41c6155cd6 =600x)

# 5/31
- Summarize the soft-attention model experiments on MLT dataset.
- Overall, Depth helps RGB for ~10% testing accuracy which is significant.
- Below is a visualization of quantitative results and qualitative results:
![图片](http://agroup-bos.su.bcebos.com/088e02c88bedae1405535673f0b5f9bccd8ee1d0 =600x)
![图片](http://agroup-bos.su.bcebos.com/85f6b2359c0eada7ac97a6963588acc19ba623dc =600x)
![图片](http://agroup-bos.su.bcebos.com/3346a09af0f3fcc95275e9a366e6f2b6878ef5c2 =600x)
![图片](http://agroup-bos.su.bcebos.com/497206b66e99de0b5a6cd18874874b7b1b4c47d7 =600x)
![图片](http://agroup-bos.su.bcebos.com/195c76019502a0f7e117c2d3a0cd8db20ebe69e2 =600x)
![图片](http://agroup-bos.su.bcebos.com/e1343aa91ba115610d2f2d471b3b41354c227ac0 =600x)

# 5/28
- Working on soft-attention model and switch back to MLT dataset because Mnist dataset is too simple.
- On MLT dataset, there are two main promising conclusions: 
1. Adding depth significantly helps image recognition. For example, the testing recognition accuracy increases from 42% to 48% by adding depth. And after adding top-down direction signal, the testing accuracy further increases from 48% to 72% which is about 30% total absolute improvement to the baseline RGB only.
2. Adding top-down direction as keyword to obtain attention significantly helps image recognition. For example, the testing recognition accuracy increases from 48% to 61% by adding top-down direction on RGB image.
- In total, the depth and direction keyword can significantly improve recognition accuracy from 42% to 72%, increasing 30%, and the training actually has not converged yet. And all curves haven't converged yet.
![图片](http://agroup-bos.su.bcebos.com/0b5775a0f23d4d1db745f4bcc24b7c0f0d305def =200x)![图片](http://agroup-bos.su.bcebos.com/f8fc486d01fe3719104026689c98375ee1b98253 =300x)![图片](http://agroup-bos.su.bcebos.com/49fed034e407f5a60f0c43e2c1d316e035e5c0a0 =300x)![图片](http://agroup-bos.su.bcebos.com/6ded1b6986cefbb260182cd2ba4350ca0c78db5b =300x)![图片](http://agroup-bos.su.bcebos.com/35c1c04f6ba43db379795bccf046951c114f05b1 =300x)
- An illustration of the soft attention model
![图片](http://agroup-bos.su.bcebos.com/204783d4733b98323d66b51c45708fd91dd6b1ec =300x)![图片](http://agroup-bos.su.bcebos.com/818b3f6e995da4998c76af0c1b22f65b66470223 =300x)![图片](http://agroup-bos.su.bcebos.com/9872c29f00468d9a547280780a50aec54af3a60f =300x)

# 5/25

- Conduct comparison between hard attention model and soft attention model on Mnist dataset. Conclusion: using soft attention model performs much more smoother and faster convergence compared to previous hard attention (spatial transformer networks). The oracle performance of soft attention model may be slightly worse than the best hard attention model on recognition, however, in reality training converges much smoother.
- Here I show the training convergence using the soft attention model and a baseline hard attention model (orange) on Mnist dataset. One can see the three (red, cyan, gray) soft-attention curves all converge much faster than the (orange) hard-attention (spatial transformer networks) curve. For more hard attention model performances, please see the last week (5/24/2018) note.
![图片](http://agroup-bos.su.bcebos.com/974b82dfb8209fd2e96d6623c3bc0b68d9f19fe5 =200x)![图片](http://agroup-bos.su.bcebos.com/febf3f9bd7396ac47e7d9181ab0a78a2d8b81b59 =300x)![图片](http://agroup-bos.su.bcebos.com/c9f9a5230793d2f0296044b10538c1611ab8a456 =300x)![图片](http://agroup-bos.su.bcebos.com/4ae43bf32a207e178a352941b6a2c4fbc94a7638 =300x)![图片](http://agroup-bos.su.bcebos.com/dde44048c76022cfd7641aafbabf371af22415b9 =300x)
- On Mnist two object dataset, using the ground truth direction lead to the best convergence which now makes a lot of sense.

# 5/24

- The attention model (spatial transformer networks) can work now on Mnist dataset, when Mnist digits are randomly uniformly located in the image. Both image-based attention and word-based attention can provide reasonable attention. Jointly train them can achieve even better and faster convergence on learning classifiers and attention models.
![图片](http://agroup-bos.su.bcebos.com/177aa3624fc93d8fb52e8b267d86e98c62987920 =200x)![图片](http://agroup-bos.su.bcebos.com/2f8ecbe0baf20a9b5f20e90538795060cc5881ba =300x)![图片](http://agroup-bos.su.bcebos.com/ee4d310b6d7507d970a64b753383e02c0a72ec2c =300x)![图片](http://agroup-bos.su.bcebos.com/de3f8bc73cf1e511a555a010ecf03af1b6ffaa5b =300x)![图片](http://agroup-bos.su.bcebos.com/a3a107023579fb7b786651092108ef083f518211 =300x)

# 5/17

- The attention model (spatial transformer networks) does not work on MLT dataset so far. 
- I find the model struggling in finding the correct object location, and is very sensitive to initialization.
- Debug attention networks on Mnist dataset-
 and find it even fails on simple Mnist data. Below is an illustration
![图片](http://agroup-bos.su.bcebos.com/669cf37528c33dbe6bfdeb17706029266d354bca =150x)![图片](http://agroup-bos.su.bcebos.com/5da9af8a06b082bbfd2b10f1280def220b1a0e00 =150x)![图片](http://agroup-bos.su.bcebos.com/4ee19733467d37c4e9cb8d5f6be85779127699c8 =195x)![图片](http://agroup-bos.su.bcebos.com/fddeede4cc4e75afff97aab60940453453631212 =195x)

# 5/10
- Start to involve attention models to recognize objects. Previously we manually crop the image patch based on the bounding box. Now we use spatial transformer networks to crop the image based on the bounding box.
- To make training/testing more stable, now use 10742 training images and 4605 testing images, 6 classes on [MLT dataset](http://robots.princeton.edu/projects/2016/PBRS/).
- Performance summarization: Random guess is 25%, brute-forcely train with only image 40%, brute-forcely train image with box 62%, suggesting a high location prior in the image, spatial transformer attention crop with gt box 69%.
![图片](http://agroup-bos.su.bcebos.com/0652334e97021dcdaad1bf2e706f32cbaf3ab34b =300x)


# 5/3

- Summarize the performance of using the bounding box to align v.s. no aligned, and using RGB + depth v.s. RGB only, on [MLT dataset](http://robots.princeton.edu/projects/2016/PBRS/), with 7000 training images and 700 testing images, 6 classes. 
- Try to train a residual network to beat the previous VGG network, but residual network obtains worse performance on all different tasks. Will figure out why.
- Start to work on depth based attention model to recognize objects in the cluttered scene (to further study the depth usage). Currently, the baseline without attention is only 40% accuracy on the testing set.
![图片](http://agroup-bos.su.bcebos.com/9f728944467f3386e05695692e515cf61f9377be =300x)![图片](http://agroup-bos.su.bcebos.com/8d2be3f3f013e2558a46cb36c2bbab29dbe2580d =300x)![图片](http://agroup-bos.su.bcebos.com/b6aa5f4e260e4dec36a92d2b851b4cb1817e621c =300x)![图片](http://agroup-bos.su.bcebos.com/2b61c7cee733430b51921b159662e5586e300fd6 =300x)


# 4/26

- Compare the effect of using bounding box to crop an object then classify (crop) v.s. directly recognize object in all possible object scales (wild). The results suggest there is more than 10% (significant) gap on recognizing both 7 classes on VIPER dataset and 6 classes on MLT dataset.
- The simplest random guess baseline on MLT testing is 29% accuracy, without any processing 44% accuracy, with bounding box processing 58% accuracy (16% improvement).
- ![图片](http://agroup-bos.su.bcebos.com/03c0c991b5846c70240aed736e3e213594c89587 =100x)![图片](http://agroup-bos.su.bcebos.com/f02d9a9b14d408dcfee6711691dd816ead0b6978 =197x)
- Using RGB and depth as input together (75%) outperforms using RGB only (58%), significantly (17% improvement).
-![图片](http://agroup-bos.su.bcebos.com/ed3a508e48ac6375c961b3aa68514ffb39c85caf =100x)![图片](http://agroup-bos.su.bcebos.com/e0e89cc702b2cbfa91b5eab2da5ab19bfe0b6887 =197x)
- When objects are not croped with bounding box, using RGB and depth as input together (55%) still outperforms using RGB only (44%), significantly (11% improvement)
-![图片](http://agroup-bos.su.bcebos.com/f9b3669b5c50c29334bec379d325d34cd9e867dc =100x)![图片](http://agroup-bos.su.bcebos.com/0b69659670546fdd340332b729fa1c94464a0221 =197x)
- Even when the number of training data decrease (from 100% to 20%), the testing accuracy on RGB + depth (60%) still outperforms RGB only with 100% training data (58%).
-![图片](http://agroup-bos.su.bcebos.com/623a76354da8dbc13c58f1a6d71d6d044db7a0f9 =100x)![图片](http://agroup-bos.su.bcebos.com/11884ae3f5665ec95af8f4ed1952d6dcdab9ece6 =197x)
- For unaligned objects, the conclusion is the same, even when the number of training data decrease (from 100% to 20%), the testing accuracy on RGB + depth (55%) still outperforms RGB only with 100% training data (44%)
-![图片](http://agroup-bos.su.bcebos.com/711c3be4b818adaa995dff562d17baf23c7200d7 =100x)![图片](http://agroup-bos.su.bcebos.com/519ded00a5d015dd38f267a60cd1bb1d13a8afbc =197x)
- The accuracy of using RGB + depth on unaligned objects (53%) is approximately equal to the accuracy of using RGB on aligned objects (55%).
-![图片](http://agroup-bos.su.bcebos.com/4c94a312c124b4c6ebba2eaf6c907b2afc7241f9 =100x)![图片](http://agroup-bos.su.bcebos.com/705223afb567e73c48ae26725fd48901a6e9c85b =197x)

# 4/5
- Compare the effect of using bounding box to crop an object then classify (crop) v.s. directly recognize object with all possible object scales (wild). The results suggest there is 10% benefit (significant) on recognizing 7 classes in VIPER dataset. 
- The baseline is 55% accuracy, without any processing 85% accuracy, with bounding box processing 95%.
- Compare the performance gap between the crop v.s. wild with the amount of training data. It seems there is no significant gap between using more than 30000 images and using only 300 images. Experiments suggest there may be a bug to fix.

![图片](http://bos.nj.bpc.baidu.com/v1/agroup/c4e4deb188ba6a31fa8a3b9f8b06d7966b79c793 =158x) ![图片](http://bos.nj.bpc.baidu.com/v1/agroup/9639b189bbca1deb2db4195b629354b9c476445f =250x)![图片](http://bos.nj.bpc.baidu.com/v1/agroup/6a13e7531e10d464120c96403007232d119d79b3 =250x)![图片](http://bos.nj.bpc.baidu.com/v1/agroup/6531af175c917377b1f56d507ae262b8b576861d =250x)![图片](http://bos.nj.bpc.baidu.com/v1/agroup/ffa47df9c08c6d9df4d74fde949ea3ef721edebc =250x)

# 3/29
- Preliminarily study the effect of noise feature on few-shot learning.
- Conclude that if we want the learner to learn fast and accurate with a few training examples, the quality of input feature is significantly important.
- The quality includes two parts: Purity and Integrity (Completeness). Purity means there is no independent noise included in the feature vector, all features are useful. Integrity means the feature is discriminative enough to provide information for classification.
- Derive the equation about bias and variance for the purity and integrity (completeness).
- Conduct experiments on the train / test loss w.r.t. the number of training data and the noise dimensionality ratio ( dim(noise) / dim(useful feature))

![图片](http://bos.nj.bpc.baidu.com/v1/agroup/21bcbc5c9dc3d9074dea4d7138968431bf49d270 =250x) ![图片](http://bos.nj.bpc.baidu.com/v1/agroup/69ff5091a5dc17fc64b1119476924dd27101b8dd =250x) ![图片](http://bos.nj.bpc.baidu.com/v1/agroup/9c811a87dffded086228cbfb9b07b44bbb2e93ef =250x)![图片](http://bos.nj.bpc.baidu.com/v1/agroup/70748e9d34de5b184c658725bcce46da85bf2d22 =250x)

# 3/22
- Investigating the reference work on illumination invariance learning, find there are two existing dataset, http://robotics.pme.duth.gr/phos2.html
- Come up a new model for modeling the optical flow on the illumination change. The overall idea is that there will be multiple motions on a single pixel.
- Investigate the new VIPER (playing for benchmarks) dataset, the dataset contains ground truth annotation for videos, including optical flow, camera pose, semantic segmentation, instance segmentation, 3D object detection.
- Will use VIPER dataset to study the lighting change for optical flow estimation and detection for one-shot classification.

![图片](http://bos.nj.bpc.baidu.com/v1/agroup/caa9dd642cbd1667145747d25a2ea00f21274c04 =250x) ![图片](http://bos.nj.bpc.baidu.com/v1/agroup/298751c077cb6cd55d089a71c5e44f764876e9f3 =250x)![图片](http://bos.nj.bpc.baidu.com/v1/agroup/e683d112e80e2c3c1f9cbb6f263596333525b895 =250x)![图片](http://bos.nj.bpc.baidu.com/v1/agroup/5593a891763218705494f820155c7729144da08b =250x)

# 3/15
- Writing 2 ECCV papers, one is about 3D object pose estimation, the other is about video hightlight extraction.

# 3/8

- ECCV writing about 3D object pose estimation with fine-grained objects
- Restart the thought of predictive learning
- Give a talk at group meeting

# 2/29

- Working on ECCV submission about 3D object pose estimation with fine-grained objects
- Discuss with Yuanpeng about zero-shot Caption-VQA transfer

# 1/25

- Read the Book An Introduction to Reinforcement Learning 2 by Richard Sutton.
- Prepare the knowledge for model-based planning.
- No experiment this week.

# 1/18

- Discuss and prepare an idea for Enrivonment model to improve Exploration

# 1/11

- Finish building the 2d push box game (preliminarily)
- Train an unsupervised super-mario game with Deep Q Learning, DQN played very poor on this game

![图片](http://bos.nj.bpc.baidu.com/v1/agroup/86a66185fcf19644c97849ffe9efa11164e22080 =250x) ![图片](http://bos.nj.bpc.baidu.com/v1/agroup/29281633bfd85a5de066743ccf8198ebd976b9f1 =250x)

# 1/4

- Start to implement push box game in python based XWorld 2D environment, expected to finish the game development in a week.
- The overall idea is to see whether long-term based planning can help decision making and whether unsupervised learning can help long-term based planning.