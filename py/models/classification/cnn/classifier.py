from py.models.basemodel import BaseModel

from .vgg import VGG16
from .resnet.resnet50 import ResNet50

class ConvolutionClassifier(BaseModel):
    def __init__(self, model_name, device=None, num_classes=None):
        self.num_classes = num_classes
        if num_classes is None:
            raise ValueError(f"Invalid numclasses: {num_classes}, must be Int.")
            
        super().__init__(model_name, device)
    
    def model_zoo(self):
        model_zoo = [#ImageNet
                     "VGG16_gap", "VGG16_mlp", "VGG16",
                     "ResNet50_gap", "ResNet50_mlp", "ResNet50"
                     ]
    
        print( "Model zoo:", model_zoo, "\n" )
        
    # def initialize_model(self, model_name):
        
    #     if "vgg16" in model_name.lower(): # vgg16_mlp vgg16_gap vgg16
    #         set_gap = True if "gap" in model_name.lower() else False
    #         set_mlp = True if "mlp" in model_name.lower() else False
    #         self.model = VGG16(num_classes=self.num_classes, 
    #                            use_mlp=set_mlp,  
    #                            use_gap=set_gap,
    #                            )
        
    #     else:
    #         self.model = None
            
    #     self.model_name = self.get_model_name()
    
    def initialize_model(self, model_name):
            
            # Verificamos las banderas de GAP/MLP una sola vez para simplificar
            set_gap = True if "gap" in model_name.lower() else False
            set_mlp = True if "mlp" in model_name.lower() else False
            
            if "vgg16" in model_name.lower(): 
                self.model = VGG16(num_classes=self.num_classes, 
                                use_mlp=set_mlp,  
                                use_gap=set_gap)
                                
            # 3. Añadimos la lógica de inicialización para ResNet-50
            elif "resnet50" in model_name.lower():
                self.model = ResNet50(num_classes=self.num_classes,
                                    use_mlp=set_mlp,
                                    use_gap=set_gap)
            
            else:
                self.model = None
                
            self.model_name = self.get_model_name()