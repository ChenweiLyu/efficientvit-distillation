import argparse
import math
import os

import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from efficientvit.apps.utils import AverageMeter
from distillation import EfficientVITFeatureExtractor
from efficientvit.cls_model_zoo import create_cls_model


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/storage2/datasets/chenwei/ImageNet/val")
    parser.add_argument("--gpu", type=str, default="all")
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=64)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_ratio", type=float, default=0.95)
    parser.add_argument("--model", type=str, default="b1-r256")
    parser.add_argument("--weight_url", type=str, default="/storage2/datasets/chenwei/code/efficientvit/outputs/student_model_vitl_b1-r256_4.pth")

    args = parser.parse_args()
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.batch_size = args.batch_size * max(len(device_list), 1)

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            args.path,
            transforms.Compose(
                [
                    transforms.Resize(
                        int(math.ceil(args.image_size / args.crop_ratio)), interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    student_model = torch.load(args.weight_url, map_location="cpu").cuda()
    student_model.eval()

    student_model_0 = create_cls_model("b1-r256", False, dropout=0).cuda()

    feature_extractor = EfficientVITFeatureExtractor(student_model_0, device="cuda")

    teacher_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitl14").cuda()
    dinov2_vits14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc').cuda()


    linear_head_state_dict = torch.load('/storage2/datasets/chenwei/code/efficientvit/outputs/dinov2_vitl14_linear_head.pth', map_location='cpu')

    # Inspect the keys and shapes
    for key, value in linear_head_state_dict.items():
        print(f"{key}: {value.shape}")
    # input_features = 768  # Feature vector size from the DINO-v2-s backbone
    input_features = 2048  # Feature vector size from the DINO-v2-l backbone
    # input_features = 3072  # Feature vector size from the DINO-v2-g backbone
    output_features = 1000  # Number of classes

    # Create the linear layer
    linear_head = torch.nn.Linear(in_features=input_features, out_features=output_features).cuda()

    # Load the state dictionary into the linear layer
    linear_head.load_state_dict(linear_head_state_dict)

    top1 = AverageMeter(is_distributed=False)
    top5 = AverageMeter(is_distributed=False)
    pool = torch.nn.AvgPool1d(256)

    model = create_cls_model(
        name="l1", weight_url="/storage2/datasets/chenwei/code/efficientvit/outputs/l1-r224.pt"
    ).cuda()



    resize_efficient_vit = transforms.Resize([224, 224])
    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc=f"Eval {args.model} on ImageNet") as t:
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()
                out1 = feature_extractor(images)
                evit_inter = teacher_model.norm(out1)

                images = resize_efficient_vit(images)
                dino_cls = teacher_model(images) # DiNOv2 cls
                dino_features = teacher_model.forward_features(images)["x_norm_patchtokens"].mean(dim=1)
                evit_any = pool(evit_inter.permute(0, 2, 1)).squeeze() # EViT features / cls
                linear_input = torch.cat([
                        dino_cls,
                        evit_any,
                    ], dim=1)
                output = linear_head(linear_input)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                t.set_postfix(
                    {
                        "top1": top1.avg,
                        "top5": top5.avg,
                        "resolution": images.shape[-1],
                    }
                )
                t.update(1)

    print(f"Top1 Acc={top1.avg:.3f}, Top5 Acc={top5.avg:.3f}")


if __name__ == "__main__":
    main()
