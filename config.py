class DefaultConfig(object):

    # trian config
    weights = 'yolov4-p5.pt' # initial weights path
    cfg = 'yolov4-p5.yaml' # model.yaml path
    data = 'data/coco128.yaml' # data.yaml path
    hyp = '' # hyperparameters path, i.e. data/hyp.scratch.yaml
    epochs = 300
    batch_size = 16 # total batch size for all GPUs
    img_size = [640, 640] # train,test sizes
    rect = True # rectangular training
    resume = False # resume from given path/last.pt, or most recent run if blank
    nosave = True # only save final checkpoint
    notest = True # only test final epoch
    noautoanchor = True # disable autoanchor check
    evolve = True # evolve hyperparameters
    bucket = '' # gsutil bucket
    cache_images = True # cache images for faster training
    name = '' # renames results.txt to results_name.txt if supplied
    device = '' # cuda device, i.e. 0 or 0,1,2,3 or cpu
    multi_scale = True # vary img-size +/- 50%%
    single_cls = True # train as single-class dataset
    adam = True # use torch.optim.Adam() optimizer
    sync_bn = True # use SyncBatchNorm, only available in DDP mode
    local_rank = -1 # DDP parameter, do not modify
    logdir = 'runs/' # logging directory

    # test config
    img_size_test = 640
    conf_thres_test = 0.001 # object confidence threshold
    iou_thres_test = 0.65 # IOU threshold for NMS
    save_json = True # save a cocoapi-compatible JSON results file
    task = 'val' # 'val', 'test', 'study'
    merge = True # use Merge NMS
    verbose = True # report mAP by class

    # Inference config
    source = 'inference/images' # file/folder, 0 for webcam
    output = 'inference/output' # output folder
    img_size_inference = 640 # inference size (pixels)
    conf_thres = 0.4 # object confidence threshold
    iou_thres = 0.5 # IOU threshold for NMS
    view_img = True # display results
    save_txt = True # save results to *.txt
    classes = 0 # filter by class: --class 0, or --class 0 2 3
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    agnostic_nms = True # class-agnostic NMS
    augment = True # augmented inference
    update = True # update all models


opt = DefaultConfig()
