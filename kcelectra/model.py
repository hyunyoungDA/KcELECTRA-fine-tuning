import torch
import pandas as pd
from transformers import AutoTokenizer, ElectraForSequenceClassification

def create_model():
    model = ElectraForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
    
    return model, tokenizer