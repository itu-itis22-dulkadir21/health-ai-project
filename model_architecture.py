import timm
import torch.nn as nn

class SwinStrokeModel(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window7_224', pretrained=True, num_classes=2):
        super().__init__()
        # Modeli doğrudan ana sınıfa ata (iç içe modül kullanma)
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.swin(x)