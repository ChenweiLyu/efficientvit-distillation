from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import torch
from typing import Literal
from functools import partial
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
import math
from torch.nn import functional as F
from efficientvit.cls_model_zoo import create_cls_model
from efficientvit.models.nn import ConvLayer, LinearLayer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", \
    "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]

class DinoV2ExtractFeatures:
    """
        Extract features from an intermediate layer in Dino-v2
    """
    def __init__(self, dino_model: _DINO_V2_MODELS, layer: int,
                 facet: _DINO_FACETS = "token", use_cls=False,
                 norm_descs=True, device: str = "cpu") -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """

        self.vit_type: str = dino_model
        self.dino_model: torch.nn.Module = torch.hub.load(
            'facebookresearch/dinov2', dino_model)
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer]. \
                register_forward_hook(
                self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer]. \
                attn.qkv.register_forward_hook(
                self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # Hook data
        self._hook_out = None

    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output

        return _forward_hook

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
            Parameters:
            - img:   The input image
        """
        with torch.no_grad():
            res = self.dino_model(img)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2 * d_len]
                else:
                    res = res[:, :, 2 * d_len:]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None  # Reset the hook
        return res

    def __del__(self):
        self.fh_handle.remove()

class EfficientVITFeatureExtractor:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self._features = None
        found_pooling = False
        # uncomment these 2 lines in distillation; comment these in evaluation
        # self.model.head.op_list[0] = torch.nn.Sequential(ConvLayer(256, 384, 1, norm="bn2d", act_func="hswish")) # match channel number
        # self.model.head.op_list[2] = torch.nn.Sequential(LinearLayer(384, 1600, False, norm="ln", act_func="hswish")) # 1600 should be changed based on the model selected. See cls.py
        self.model = model.to(device)
        # Iterate through the modules in ClsHead to find the AdaptiveAvgPool2d layer
        for module in reversed(list(self.model.head.modules())):
            if isinstance(module, torch.nn.AdaptiveAvgPool2d):
                found_pooling = True
            elif found_pooling and isinstance(module, ConvLayer):
                # Register the hook to this ConvLayer
                module.register_forward_hook(self._hook)
                break

    def _hook(self, module, input, output):
        # Store the features from the hook
        self._features = output

    def __call__(self, x):
        self._features = None  # Reset stored features
        x = x.to(self.device)
        self.model(x)

        # Return the stored features
        return self._features.permute(0, 2, 3, 1).reshape(self._features.shape[0], 256, -1)


def loss_fn(outputs, targets):
    return F.mse_loss(outputs, targets)

writer = SummaryWriter("logs")

def train():

    # img_size = 224
    # patch_size = 16
    # n_tokens = (img_size // patch_size) ** 2
    # mask_generator = MaskingGenerator(
    #     input_size=(img_size // patch_size, img_size // patch_size),
    #     max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    # )
    #
    # data_transform = DataAugmentationDINO(
    #     (0.32, 1.0),
    #     (0.05, 0.32),
    #     8,
    #     global_crops_size = 224,
    #     local_crops_size = 96,
    # )
    #
    # collate_fn = partial(
    #     collate_data_and_cast,
    #     mask_ratio_tuple=(0.1, 0.5),
    #     mask_probability=0.5,
    #     n_tokens=n_tokens,
    #     mask_generator=mask_generator,
    #     dtype=torch.half,
    # )
    #
    # # setup data loader
    #
    # dataset = make_dataset(
    #     dataset_str= "ImageNet:split=TRAIN:root=/storage2/datasets/chenwei/ImageNet:extra=/storage2/datasets/chenwei/ImageNet",
    #     transform=data_transform,
    #     target_transform=lambda _: (),
    # )
    # # sampler_type = SamplerType.INFINITE
    # sampler_type = SamplerType.SHARDED_INFINITE
    # data_loader = make_data_loader(
    #     dataset=dataset,
    #     batch_size= 64,
    #     num_workers= 0,
    #     shuffle=True,
    #     seed=0,
    #     sampler_type=sampler_type,
    #     sampler_advance=0,
    #     drop_last=True,
    #     collate_fn=collate_fn,
    # )

    device = "cuda"
    train_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            "/storage2/datasets/chenwei/ImageNet/train",
            transforms.Compose(
                [
                    transforms.Resize(
                        int(math.ceil(224 / 0.95)), interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        ),
        batch_size=64, # b1 - 64
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            "/storage2/datasets/chenwei/ImageNet/val",
            transforms.Compose(
                [
                    transforms.Resize(
                        int(math.ceil(224 / 0.95)), interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        ),
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    resize_efficient_vit = transforms.Resize([512, 512])
    last_layer = 10

    teacher_model = DinoV2ExtractFeatures(dino_model="dinov2_vits14", layer=last_layer, device="cuda")
    teacher_model.trainable_parameters = False

    student_model = create_cls_model("b1-r256", True, dropout=0) # pretrained

    # Creating the feature extractor does not modify the model's parameters
    feature_extractor = EfficientVITFeatureExtractor(student_model, device="cuda")

    student_model.trainable_parameters = True


    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)
    # Run through dataloader
    epochs = 10
    train_step = 0

    for epoch in range(epochs):
        train_loss = 0.0
        student_model.train()
        with tqdm(total=len(train_data_loader), desc=f"Training {epoch} epoch") as t:
            for i, (images, labels) in enumerate(train_data_loader):
                images, labels = images.to(device), labels.to(device)
                teacher_outputs = teacher_model(images)
                # print("teacher",teacher_outputs.shape)
                images = resize_efficient_vit(images)
                student_outputs  = feature_extractor(images)
                # print("student",student_outputs.shape)

                loss = loss_fn(student_outputs, teacher_outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_step += 1
                if not train_step % 10:
                    writer.add_scalar("train_loss", loss.item(), train_step)
                    # print(loss.item())
                if not train_step % 500:
                    scheduler.step()
                    writer.add_scalar("learning rate", optimizer.state_dict()['param_groups'][0]['lr'], train_step)
                t.update(1)

        print("Epoch:", epoch,"training loss:", train_loss)

        val_loss = 0.0
        student_model.eval()
        with tqdm(total=len(val_data_loader), desc=f"Validating {epoch} epoch") as t:
            for i, (images, labels) in enumerate(val_data_loader):
                images, labels = images.to(device), labels.to(device)
                teacher_outputs = teacher_model(images)
                images = resize_efficient_vit(images)
                student_outputs  = feature_extractor(images)

                loss = loss_fn(student_outputs, teacher_outputs)
                val_loss += loss.item()
                t.update(1)

        print("Epoch:", epoch,"validating loss:", val_loss)
        writer.add_scalar("validating loss", val_loss, epoch)

        torch.save(student_model, f"./outputs/student_model_vits_b1-r256_p_{epoch}.pth")
        print(f"model student_model_vits_b1-r256_p_{epoch}.pth is successfully saved!")
    writer.close()

if __name__ == "__main__":
    train()