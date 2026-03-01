import torch
from torchvision import models, transforms
from PIL import Image
import streamlit as st

@st.cache_resource(show_spinner="Loading pre-trained ResNet-18...")
def load_resnet_and_hooks():
    # Load model using the new weights parameter to avoid warnings
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.eval()
    
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
        
    # Register hooks on Conv1 and the 마지막 Layer (layer4)
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.layer4.register_forward_hook(get_activation('layer4'))
    
    return model, activations

def preprocess_image(image: Image.Image):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    input_tensor = preprocess(image)
    return input_tensor.unsqueeze(0) # create a mini-batch
