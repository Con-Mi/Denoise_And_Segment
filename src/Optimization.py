#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
############    This file is supposed to be a runnable script
############    as it is.
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torchvision

from Models.DenseLinkNet import DenseLinkModel
from Utils.Functions.ScoringFunctions import jaccard_score, dice_score
from DataLoaders.PyTorch.OxfPets import PetsDataLoader, PetsValidationDataLoader

import time
import copy
from tqdm import tqdm


# HYPERPARAMETERS
USE_CUDA_ON = torch.cuda.is_available()
if USE_CUDA_ON:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
BATCH_SZ: int = 12
NR_EPOCHS: int = 20
MOMENTUM: float = 0.95
LR_RATE: float = 0.03
MILSESTONES: list = [5, 7, 8, 10, 12, 14, 16, 17, 18]
IMG_SIZE: int = 384
GAMMA: float = 0.5

#   Model
SegmModel = DenseLinkModel(input_channels=3, pretrained=True)
SegmModel.to(device=DEVICE)


MUL_TRANFORMS: list = [ torchvision.transforms.ToTensor(), torchvision.transforms.Resize(size=(IMG_SIZE, IMG_SIZE)) ]

#optimizerSGD = torch.optim.SGD(SegmModel.parameters(), lr=LR_RATE, momentum=MOMENTUM)
optimizerSGD = torch.optim.Adam(SegmModel.parameters())
#criterion = torch.nn.BCEWithLogitsLoss().cuda() if USE_CUDA_ON else torch.nn.BCEWithLogitsLoss()
criterion = torch.nn.SmoothL1Loss().cuda() if USE_CUDA_ON else torch.nn.SmoothL1Loss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerSGD, milestones=MILSESTONES, gamma=GAMMA)

#   DATALOADER
train_ds_loader = PetsDataLoader(data_transform=torchvision.transforms.Compose(MUL_TRANFORMS), BatchSz=BATCH_SZ, worker_threads=6, shuffleOn=True)
valid_ds_loader = PetsValidationDataLoader(data_transform=torchvision.transforms.Compose(MUL_TRANFORMS), BatchSz=BATCH_SZ, worker_threads=6, shuffleOn=True)

dict_loaders = {"train": train_ds_loader, "valid": valid_ds_loader}


def train_model(cust_model, dataloaders, criterion, optimizer, num_epochs, scheduler=None):
    start_time = time.time()
    val_acc_history = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(cust_model.state_dict())

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("_" * 15)
        for phase in ["train", "valid"]:
            if phase == "train":
                cust_model.train()
            if phase == "valid":
                cust_model.eval()
            running_loss = 0.0
            jaccard_acc = 0.0
            dice_loss = 0.0

            for input_img, noisy_input_img, labels,  in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                input_img = input_img.to(device=DEVICE, dtype=torch.float)
                labels = labels.to(device=DEVICE, dtype=torch.float)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    out = cust_model(input_img)
                    preds = torch.sigmoid(out)
                    loss = criterion(preds, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input_img.size(0)
                jaccard_acc += jaccard_score(labels, preds)
                # dice_acc += dice(labels, preds)

            epoch_loss = running_loss / len(dataloaders[phase])
            aver_jaccard = jaccard_acc / len(dataloaders[phase])
            # aver_dice = dice_acc / len(dataloaders[phase])

            print("| {} Loss: {:.4f} | Jaccard Average Acc: {:.4f} |".format(phase, epoch_loss, aver_jaccard))
            print("_" * 15)
            if phase == "valid" and aver_jaccard > best_acc:
                best_acc = aver_jaccard
                best_model_wts = copy.deepcopy(cust_model.state_dict)
            if phase == "valid":
                val_acc_history.append(aver_jaccard)
        print("^" * 15)
        print(" ")
        scheduler.step()
    time_elapsed = time.time() - start_time
    print("Training Complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best Validation Accuracy: {:.4f}".format(best_acc))
    best_model_wts = copy.deepcopy(cust_model.state_dict())
    cust_model.load_state_dict(best_model_wts)
    return cust_model, val_acc_history


segm_model, acc = train_model(SegmModel, dict_loaders, criterion, optimizerSGD, NR_EPOCHS, scheduler=scheduler)
#save_model(segm_model, name="dense_linknet_20.pt")