# general libs
import os, sys, argparse
import random, time
import warnings

warnings.filterwarnings("ignore")
import cv2
import numpy as np
import torch
import torch.nn as nn
import datetime


from utils import *
import utils.helpers as helpers
from utils.optimizer import PolyWarmupAdamW
from models.segformer import WeTr
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.augmentations_mm import *
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp():
    # print(os.environ.keys())
    if "SLURM_PROCID" in os.environ and not "RANK" in os.environ:
        # --- multi nodes
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        gpu = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=7200),
        )
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # ---
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            "nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=7200),
        )
        dist.barrier()
    else:
        gpu = 0
    return gpu


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Full Pipeline Training")

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="nyudv2",
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="/cache/datasets/nyudv2",
        help="Path to the training set directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size to train the segmenter model.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of workers for pytorch's dataloader.",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=255,
        help="Label to ignore during training",
    )

    # General
    parser.add_argument("--name", default="", type=str, help="model name")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="If true, only validate segmentation.",
    )
    parser.add_argument(
        "--freeze-bn",
        type=bool,
        nargs="+",
        default=True,
        help="Whether to keep batch norm statistics intact.",
    )
    parser.add_argument(
        "--num-epoch",
        type=int,
        nargs="+",
        default=[100] * 3,
        help="Number of epochs to train for segmentation network.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed to provide (near-)reproducibility.",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        default="model",
        type=str,
        metavar="PATH",
        help="path to save checkpoint (default: model)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=5,
        help="How often to validate current architecture.",
    )
    parser.add_argument(
        "--print-network",
        action="store_true",
        default=False,
        help="Whether print newtork paramemters.",
    )
    parser.add_argument(
        "--print-loss",
        action="store_true",
        default=False,
        help="Whether print losses during training.",
    )
    parser.add_argument(
        "--save-image",
        type=int,
        default=100,
        help="Number to save images during evaluating, -1 to save all.",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=["rgb", "depth"],
        type=str,
        nargs="+",
        help="input type (image, depth)",
    )

    # Optimisers
    parser.add_argument("--backbone", default="mit_b3", type=str)
    parser.add_argument("--n_heads", default=8, type=int)
    parser.add_argument("--drop_rate", default=0.0, type=float)
    parser.add_argument("--dpr", default=0.4, type=float)

    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--lr_0", default=6e-5, type=float)
    parser.add_argument("--lr_1", default=3e-5, type=float)
    parser.add_argument("--lr_2", default=1.5e-5, type=float)
    parser.add_argument("--is_pretrain_finetune", action="store_true")

    return parser.parse_args()


def create_segmenter(num_classes, gpu, backbone, n_heads, dpr, drop_rate):
    segmenter = WeTr(backbone, num_classes, n_heads, dpr, drop_rate)
    param_groups = segmenter.get_param_groups()
    assert torch.cuda.is_available()
    segmenter.to("cuda:" + str(gpu))
    return segmenter, param_groups


def create_loaders(
    dataset,
    train_dir,
    val_dir,
    train_list,
    val_list,
    batch_size,
    num_workers,
    ignore_label,
):
    """
    Args:
      train_dir (str) : path to the root directory of the training set.
      val_dir (str) : path to the root directory of the validation set.
      train_list (str) : path to the training list.
      val_list (str) : path to the validation list.
      batch_size (int) : training batch size.
      num_workers (int) : number of workers to parallelise data loading operations.
      ignore_label (int) : label to pad segmentation masks with

    Returns:
      train_loader, val loader

    """
    # Torch libraries
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # Custom libraries
    from utils.datasets import SegDataset as Dataset
    from utils.transforms import ToTensor

    input_names, input_mask_idxs = ["rgb", "depth"], [0, 2, 1]

    if dataset == "nyudv2":
        input_scale = [480, 640]
    elif dataset == "sunrgbd":
        input_scale = [480, 480]

    composed_trn = transforms.Compose(
        [
            ToTensor(),
            RandomColorJitter(p=0.2),  #
            RandomHorizontalFlip(p=0.5),  #
            RandomGaussianBlur((3, 3), p=0.2),  #
            RandomResizedCrop(input_scale, scale=(0.5, 2.0), seg_fill=255),  #
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    composed_val = transforms.Compose(
        [
            ToTensor(),
            Resize(input_scale),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    # Training and validation sets
    trainset = Dataset(
        dataset=dataset,
        data_file=train_list,
        data_dir=train_dir,
        input_names=input_names,
        input_mask_idxs=input_mask_idxs,
        transform_trn=composed_trn,
        transform_val=composed_val,
        stage="train",
        ignore_label=ignore_label,
    )

    validset = Dataset(
        dataset=dataset,
        data_file=val_list,
        data_dir=val_dir,
        input_names=input_names,
        input_mask_idxs=input_mask_idxs,
        transform_trn=None,
        transform_val=composed_val,
        stage="val",
        ignore_label=ignore_label,
    )
    print_log(
        "Created train set {} examples, val set {} examples".format(
            len(trainset), len(validset)
        )
    )
    train_sampler = DistributedSampler(
        trainset, dist.get_world_size(), dist.get_rank(), shuffle=True
    )

    # Training and validation loaders
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        validset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, train_sampler


def load_ckpt(ckpt_path, ckpt_dict, is_pretrain_finetune=False):
    print("----------------")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    new_segmenter_ckpt = dict()
    if is_pretrain_finetune:
        for ckpt_k, ckpt_v in ckpt["segmenter"].items():
            if "linear_pred" in ckpt_k:
                print(ckpt_k, " is Excluded!")
            else:
                if "module." in ckpt_k:
                    new_segmenter_ckpt[ckpt_k[7:]] = ckpt_v
    else:
        for ckpt_k, ckpt_v in ckpt["segmenter"].items():
            new_segmenter_ckpt[ckpt_k] = ckpt_v
            if "module." in ckpt_k:
                new_segmenter_ckpt[ckpt_k[7:]] = ckpt_v
    ckpt["segmenter"] = new_segmenter_ckpt

    for k, v in ckpt_dict.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k], strict=False)
        else:
            print(v, " is  missed!")
    best_val = ckpt.get("best_val", 0)
    epoch_start = ckpt.get("epoch_start", 0)
    if is_pretrain_finetune:
        print_log(
            "Found [Pretrain] checkpoint at {} with best_val {:.4f} at epoch {}".format(
                ckpt_path, best_val, epoch_start
            )
        )
        return 0, 0
    else:

        print_log(
            "Found checkpoint at {} with best_val {:.4f} at epoch {}".format(
                ckpt_path, best_val, epoch_start
            )
        )
        return best_val, epoch_start


def train(
    segmenter,
    input_types,
    train_loader,
    optimizer,
    epoch,
    segm_crit,
    freeze_bn,
    print_loss=False,
):
    """Training segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      train_loader (DataLoader) : training data iterator
      optim_enc (optim) : optimiser for encoder
      optim_dec (optim) : optimiser for decoder
      epoch (int) : current epoch
      segm_crit (nn.Loss) : segmentation criterion
      freeze_bn (bool) : whether to keep BN params intact

    """
    train_loader.dataset.set_stage("train")
    segmenter.train()
    if freeze_bn:
        for module in segmenter.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()

    for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
        start = time.time()
        inputs = [sample[key].cuda().float() for key in input_types]
        target = sample["mask"].cuda().long()
        # Compute outputs
        outputs, masks = segmenter(inputs)
        loss = 0
        for output in outputs:
            output = nn.functional.interpolate(
                output, size=target.size()[1:], mode="bilinear", align_corners=False
            )
            soft_output = nn.LogSoftmax()(output)
            # Compute loss and backpropagate
            loss += segm_crit(soft_output, target)

        optimizer.zero_grad()
        loss.backward()
        if print_loss:
            print("step: %-3d: loss=%.2f" % (i, loss), flush=True)
        optimizer.step()
        losses.update(loss.item())
        batch_time.update(time.time() - start)


def validate(
    segmenter, input_types, val_loader, epoch, save_dir, num_classes=-1, save_image=0
):
    """Validate segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      val_loader (DataLoader) : training data iterator
      epoch (int) : current epoch
      num_classes (int) : number of classes to consider

    Returns:
      Mean IoU (float)
    """
    global best_iou
    val_loader.dataset.set_stage("val")
    segmenter.eval()
    conf_mat = []
    for _ in range(len(input_types) + 1):
        conf_mat.append(np.zeros((num_classes, num_classes), dtype=int))
    with torch.no_grad():
        all_times = 0
        count = 0
        for i, sample in enumerate(val_loader):
            inputs = [sample[key].float().cuda() for key in input_types]
            target = sample["mask"]
            gt = target[0].data.cpu().numpy().astype(np.uint8)
            gt_idx = (
                gt < num_classes
            )  # Ignore every class index larger than the number of classes

            """from fvcore.nn import FlopCountAnalysis, parameter_count_table

            flops = FlopCountAnalysis(segmenter, inputs)
            print("FLOPs: ", flops.total())
            print(parameter_count_table(segmenter))
            exit()"""

            start_time = time.time()

            outputs, _ = segmenter(inputs)

            end_time = time.time()
            all_times += end_time - start_time

            for idx, output in enumerate(outputs):
                output = (
                    cv2.resize(
                        output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                        target.size()[1:][::-1],
                        interpolation=cv2.INTER_CUBIC,
                    )
                    .argmax(axis=2)
                    .astype(np.uint8)
                )
                # Compute IoU
                conf_mat[idx] += confusion_matrix(
                    gt[gt_idx], output[gt_idx], num_classes
                )
                if i < save_image or save_image == -1:
                    img = make_validation_img(
                        inputs[0].data.cpu().numpy(),
                        inputs[1].data.cpu().numpy(),
                        sample["mask"].data.cpu().numpy(),
                        output[np.newaxis, :],
                    )
                    imgs_folder = os.path.join(save_dir, "imgs")
                    os.makedirs(imgs_folder, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(imgs_folder, "validate_" + str(i) + ".png"),
                        img[:, :, ::-1],
                    )
                    print("imwrite at imgs/validate_%d.png" % i)
            count += 1
        latency = all_times / count
        print("all_times:", all_times, " count:", count, " latency:", latency)

    for idx, input_type in enumerate(input_types + ["ens"]):
        glob, mean, iou = getScores(conf_mat[idx])
        best_iou_note = ""
        if iou > best_iou:
            best_iou = iou
            best_iou_note = "    (best)"
        alpha = "        "

        input_type_str = "(%s)" % input_type
        print_log(
            "Epoch %-4d %-7s   glob_acc=%-5.2f    mean_acc=%-5.2f    IoU=%-5.2f%s%s"
            % (epoch, input_type_str, glob, mean, iou, alpha, best_iou_note)
        )
    print_log("")
    return iou


def main():
    global args, best_iou
    best_iou = 0
    args = get_arguments()
    args.val_dir = args.train_dir

    if args.dataset == "nyudv2":
        args.train_list = "data/nyudv2/train.txt"
        args.val_list = "data/nyudv2/val.txt"
        args.num_classes = 40
    elif args.dataset == "sunrgbd":
        args.train_list = "data/sun/train.txt"
        args.val_list = "data/sun/test.txt"
        args.num_classes = 37

    args.num_stages = 3
    gpu = setup_ddp()
    ckpt_dir = os.path.join("ckpt", args.ckpt)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.system("cp -r *py models utils data %s" % ckpt_dir)
    helpers.logger = open(os.path.join(ckpt_dir, "log.txt"), "w+")
    print_log(" ".join(sys.argv))

    # Set random seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    # Generate Segmenter
    segmenter, param_groups = create_segmenter(
        args.num_classes,
        gpu,
        args.backbone,
        args.n_heads,
        args.dpr,
        args.drop_rate,
    )

    print_log(
        "Loaded Segmenter {}, #PARAMS={:3.2f}M".format(
            args.backbone, compute_params(segmenter) / 1e6
        )
    )
    # Restore if any
    best_val, epoch_start = 0, 0
    if args.resume:
        if os.path.isfile(args.resume):
            best_val, epoch_start = load_ckpt(
                args.resume,
                {"segmenter": segmenter},
                is_pretrain_finetune=args.is_pretrain_finetune,
            )
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume))
            return
    no_ddp_segmenter = segmenter
    segmenter = DDP(
        segmenter, device_ids=[gpu], output_device=0, find_unused_parameters=False
    )

    epoch_current = epoch_start
    # Criterion
    segm_crit = nn.NLLLoss(ignore_index=args.ignore_label).cuda()
    # Saver
    saver = Saver(
        args=vars(args),
        ckpt_dir=ckpt_dir,
        best_val=best_val,
        condition=lambda x, y: x > y,
    )  # keep checkpoint with the best validation score

    lrs = [args.lr_0, args.lr_1, args.lr_2]

    print("-------------------------Optimizer Params--------------------")
    print("weight_decay:", args.weight_decay)
    print("lrs:", lrs)
    print("----------------------------------------------------------------")

    for task_idx in range(args.num_stages):
        optimizer = PolyWarmupAdamW(
            # encoder,encoder-norm,decoder
            params=[
                {
                    "params": param_groups[0],
                    "lr": lrs[task_idx],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": param_groups[1],
                    "lr": lrs[task_idx],
                    "weight_decay": 0.0,
                },
                {
                    "params": param_groups[2],
                    "lr": lrs[task_idx] * 10,
                    "weight_decay": args.weight_decay,
                },
            ],
            lr=lrs[task_idx],
            weight_decay=args.weight_decay,
            betas=[0.9, 0.999],
            warmup_iter=1500,
            max_iter=40000,
            warmup_ratio=1e-6,
            power=1.0,
        )
        total_epoch = sum([args.num_epoch[idx] for idx in range(task_idx + 1)])
        if epoch_start >= total_epoch:
            continue
        start = time.time()
        torch.cuda.empty_cache()
        # Create dataloaders
        train_loader, val_loader, train_sampler = create_loaders(
            args.dataset,
            args.train_dir,
            args.val_dir,
            args.train_list,
            args.val_list,
            args.batch_size,
            args.num_workers,
            args.ignore_label,
        )
        if args.evaluate:
            return validate(
                no_ddp_segmenter,
                args.input,
                val_loader,
                0,
                ckpt_dir,
                num_classes=args.num_classes,
                save_image=args.save_image,
            )

        # Optimisers
        print_log("Training Stage {}".format(str(task_idx)))

        for epoch in range(min(args.num_epoch[task_idx], total_epoch - epoch_start)):
            train_sampler.set_epoch(epoch)
            train(
                segmenter,
                args.input,
                train_loader,
                optimizer,
                epoch_current,
                segm_crit,
                args.freeze_bn,
                args.print_loss,
            )
            if (epoch + 1) % (args.val_every) == 0:
                miou = validate(
                    no_ddp_segmenter,
                    args.input,
                    val_loader,
                    epoch_current,
                    ckpt_dir,
                    args.num_classes,
                )
                saver.save(
                    miou,
                    {"segmenter": segmenter.state_dict(), "epoch_start": epoch_current},
                )
            epoch_current += 1

        print_log(
            "Stage {} finished, time spent {:.3f}min\n".format(
                task_idx, (time.time() - start) / 60.0
            )
        )

    print_log("All stages are now finished. Best Val is {:.3f}".format(saver.best_val))
    helpers.logger.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
