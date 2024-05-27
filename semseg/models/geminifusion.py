import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.backbones.geminifusion_backbone import GeminiFusionBackbone
from semseg.models.heads import SegFormerHead
from semseg.models.layers import trunc_normal_
from fvcore.nn import flop_count_table, FlopCountAnalysis


class GeminiFusion(nn.Module):
    def __init__(
        self,
        backbone: str = "GeminiFusion-B0",
        num_classes: int = 25,
        modals: list = ["img", "depth", "event", "lidar"],
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        backbone, variant = backbone.split("-")
        self.backbone = eval(backbone)(
            variant,
            modals,
            drop_path_rate=drop_path_rate,
            num_modal=len(modals),
        )
        self.modals = modals

        self.decode_head = SegFormerHead(
            self.backbone.embed_dims,
            256 if "B0" in backbone or "B1" in backbone else 512,
            num_classes,
        )
        self.apply(self._init_weights)

        self.num_parallel = 2
        self.alpha = torch.nn.Parameter(
            torch.ones(self.num_parallel, requires_grad=True)
        )
        self.register_parameter("alpha", self.alpha)

    def forward(self, x: list) -> list:
        x_modals = self.backbone(x)
        outs = []
        for idx in range(self.num_parallel):
            out = self.decode_head(x_modals[idx])
            out = F.interpolate(
                out, size=x[0].shape[2:], mode="bilinear", align_corners=False
            )  # to original image shape
            outs.append(out)
        ens = 0
        alpha_soft = F.softmax(self.alpha, dim=0)
        for idx in range(self.num_parallel):
            ens += alpha_soft[idx] * outs[idx].detach()
        outs.append(ens)
        return outs

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            checkpoint = torch.load(pretrained, map_location="cpu")
            if "state_dict" in checkpoint.keys():
                checkpoint = checkpoint["state_dict"]
            if "model" in checkpoint.keys():
                checkpoint = checkpoint["model"]
            checkpoint.pop("head.weight")
            checkpoint.pop("head.bias")
            checkpoint = self._expand_state_dict(
                self.backbone.state_dict(), checkpoint, self.num_parallel
            )
            msg = self.backbone.load_state_dict(checkpoint, strict=True)
            print(msg)

    def _expand_state_dict(self, model_dict, state_dict, num_parallel):
        model_dict_keys = model_dict.keys()
        state_dict_keys = state_dict.keys()
        for model_dict_key in model_dict_keys:
            model_dict_key_re = model_dict_key.replace("module.", "")
            if model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
            for i in range(num_parallel):
                ln = ".ln_%d" % i
                replace = True if ln in model_dict_key_re else False
                model_dict_key_re = model_dict_key_re.replace(ln, "")
                if replace and model_dict_key_re in state_dict_keys:
                    model_dict[model_dict_key] = state_dict[model_dict_key_re]
        return model_dict


if __name__ == "__main__":
    modals = ["img"]
    # modals = ['img', 'depth', 'event', 'lidar']
    model = GeminiFusion("GeminiFusion-B2", 25, modals)
    model.init_pretrained("checkpoints/pretrained/segformer/mit_b2.pth")
    x = [torch.zeros(1, 3, 512, 512)]
    y = model(x)
    print(y.shape)
