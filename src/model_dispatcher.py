from model import Resnet34, Resnet50, Resnet101

MODEL_DISPATCHER = {
    'resnet34': Resnet34,
    'resnet50': Resnet50, 
    'resnet101': Resnet101
}       