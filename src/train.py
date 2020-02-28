import os 
import ast
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDataset
import torch
import torch.nn as nn
from tqdm import tqdm
from pdb import set_trace
import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEVICE = 'cuda'
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")

IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT"))
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH"))
EPOCHS = int(os.environ.get("EPOCHS"))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
VALIDATION_BATCH_SIZE = int(os.environ.get("VALIDATION_BATCH_SIZE"))

IMG_MEAN = ast.literal_eval(os.environ.get("IMG_MEAN"))
IMG_STD = ast.literal_eval(os.environ.get("IMG_STD"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))

BASE_MODEL = os.environ.get("BASE_MODEL")

def loss_func(outputs, targets):
    grapheme_root, vowel_diacritic, consonant_diacritic = outputs
    grapheme_root_, vowel_diacritic_, consonant_diacritic_ = targets 
    l1 = nn.CrossEntropyLoss()(grapheme_root, grapheme_root_)
    l2 = nn.CrossEntropyLoss()(vowel_diacritic, vowel_diacritic_)
    l3 = nn.CrossEntropyLoss()(consonant_diacritic, consonant_diacritic_)
    return (l1+l2+l3)/3 #TODO: can do weighted average which might give btr result


def train(dataset, data_loader, model, optimizer, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
        image = d['image']
        grapheme_root = d['grapheme_root']
        vowel_diacritic = d['vowel_diacritic']
        consonant_diacritic = d['consonant_diacritic']

        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def evaluate(dataset, data_loader, model):
    model.eval()
    final_loss = 0
    counter = 0
    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
        counter+=1
        image = d['image']
        grapheme_root = d['grapheme_root']
        vowel_diacritic = d['vowel_diacritic']
        consonant_diacritic = d['consonant_diacritic']

        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        with torch.no_grad():
            outputs = model(image)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss = loss_func(outputs, targets)
            final_loss += loss
    return final_loss/counter #TODO:better to use actual metric of comp and loss


def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)

    train_dataset = BengaliDataset(folds=TRAINING_FOLDS)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=TRAIN_BATCH_SIZE, 
        shuffle=True,
        num_workers=4
        )

    valid_dataset = BengaliDataset(folds=VALIDATION_FOLDS)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=VALIDATION_BATCH_SIZE, 
        shuffle=True,
        num_workers=4
        )

    # try some different parameters, or learning_rate_scheduler
    learning_rate=1e-3
    
    optimizer = torch.optim.Adam([
        {'params': model.initial_layers.parameters(), 'lr': learning_rate/20},  
        {'params': model.middle_layers.parameters(), 'lr': learning_rate/5},  
        {'params': model.later_layers.parameters(), 'lr': learning_rate},  
        {'params': model.linear_layers.parameters(), 'lr': learning_rate},  
    ], lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        steps_per_epoch=len(train_dataset)//train_dataloader.batch_size,
        max_lr=learning_rate, 
        epochs=10
        )

    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

    #TODO: implement early stopping

    for epoch in range(EPOCHS):
        logger.info(f"running epoch {epoch} of {EPOCHS}..")
        train(train_dataset, train_dataloader, model, optimizer, scheduler)
        val_score = evaluate(valid_dataset, valid_dataloader, model)
        logger.info(f"validation_loss: {val_score}")
        # scheduler.step(val_score)
        torch.save(model.state_dict(), f"./output/{BASE_MODEL}_valfold{VALIDATION_FOLDS[0]}.bin")
        
if __name__ == '__main__':
    main()