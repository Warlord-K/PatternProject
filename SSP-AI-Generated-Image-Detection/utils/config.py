import os

class DefaultConfigs:
    gpus = [0]
    seed = 3407
    arch = "resnet50"
    datasets = ["train"]
    datasets_test = ["test"]
    mode = "csv"
    class_bal = False
    batch_size = 64
    batchsize = 64  # For backward compatibility
    loadSize = 256
    cropSize = 224
    epoch = "latest"
    num_workers = 20
    serial_batches = False
    isTrain = True

    # data augmentation
    rz_interp = ["bilinear"]
    blur_prob = 0.0
    blur_sig = [0.5]
    jpg_prob = 0.0
    jpg_method = ["cv2"]
    jpg_qual = [75]
    gray_prob = 0.0
    aug_resize = True
    aug_crop = True
    aug_flip = True
    aug_norm = True

    # train settings
    warmup = False
    warmup_epoch = 3
    earlystop = True
    earlystop_epoch = 5
    optim = "adam"
    new_optim = False
    loss_freq = 400
    save_latest_freq = 2000
    save_epoch_freq = 20
    continue_train = False
    epoch_count = 1
    last_epoch = -1
    nepoch = 10
    beta1 = 0.9
    lr = 0.0001
    init_type = "normal"
    init_gain = 0.02
    pretrained = True
    
    # Additional properties for compatibility
    gpu_id = "0"
    image_root = None
    dataset_root = None
    save_path = "./checkpoint/"
    load = None

# Create a concrete instance for use
CONFIGCLASS = DefaultConfigs()