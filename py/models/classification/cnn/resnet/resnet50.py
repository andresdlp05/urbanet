import torch
import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    """Simplified ResNet50 with flexible head configurations."""
    
    def __init__(self, num_classes, use_gap=False, use_mlp=False):
        super(ResNet50, self).__init__()
        self.use_gap = use_gap
        self.use_mlp = use_mlp
        
        if use_gap and use_mlp:
            raise ValueError(f"Invalid combination: use_gap: {use_gap} and use_mlp: {use_mlp}, cannot both be True.")
        
        # Load pretrained ResNet50 features
        resnet_pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # ResNet50 features (agrupando todo antes del GAP y la capa Full Connected final)
        self.feature_maps = nn.Sequential(
            resnet_pretrained.conv1,
            resnet_pretrained.bn1,
            resnet_pretrained.relu,
            resnet_pretrained.maxpool,
            resnet_pretrained.layer1, # Bloque 1
            resnet_pretrained.layer2, # Bloque 2
            resnet_pretrained.layer3, # Bloque 3
            resnet_pretrained.layer4  # Bloque 4 (Último bloque)
        )
        self.model_arch = "resnet50"
        
        # Freeze feature extractor
        for param in self.feature_maps.parameters():
            param.requires_grad = False
            
        # Optional: Unfreeze last conv block (layer4) for fine-tuning
        # En nuestro Sequential, layer4 corresponde al índice 7
        for param in self.feature_maps[7].parameters():
            param.requires_grad = True
            
        # Build classifier head
        # ResNet50 arroja un tensor de profundidad 2048
        if use_gap:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = self._build_mlp_head(2048, num_classes)
            self.model_name = "ResNet50_GAP"
            
        elif use_mlp:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = self._build_mlp_head(2048 * 7 * 7, num_classes)
            self.model_name = "ResNet50_MLP"
            
        else:
            # Use pretrained classifier with avgpool
            self.avgpool = resnet_pretrained.avgpool
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, num_classes)
            )
            nn.init.xavier_uniform_(self.classifier[1].weight)
            self.model_name = "ResNet50"
            
    def _build_mlp_head(self, input_dim, num_classes):
        """Build a simple MLP classification head."""
        layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        # Initialize weights
        nn.init.xavier_uniform_(layers[1].weight)
        nn.init.xavier_uniform_(layers[4].weight)
        return layers

    def get_model_arch(self):
        return self.model_arch

    def get_model_name(self):
        return self.model_name
    
    def forward(self, x):
        x = self.feature_maps(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x