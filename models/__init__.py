# Bridge module for tools/inference_on_a_image.py
# It expects: from groundingdino.models import build_model

from models.GroundingDINO import build_groundingdino as build

def build_model(args):
    return build(args)
