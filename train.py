# ----------------------------------------------------------------------
# Title:    train.py
# Details:  train Flower classification models,
#           save and predict
#
# Author:   Ruihan Xu
# Created:  2019/12/25
# Modified: 2020/01/17
# ----------------------------------------------------------------------

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import models
import time
import FC_io


# run model on validation data and return accuracy
def val_model(model, loader):
    corrects = 0.0
    total = len(loader.dataset)

    model.eval()
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs.data, 1)
            corrects += torch.sum(preds == labels.data).to(torch.float32)
    val_acc = corrects / total
    print('Valid Acc: {:.4f}'.format(val_acc))
    return val_acc


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                num_epochs=40):
    total = len(train_loader.dataset)
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        if epoch == 20:
            for parma in model.parameters():
                parma.requires_grad = True
            # Observe that all parameters are being optimized
            optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
            # Decay LR by a factor of 0.1 every 5 epochs
            scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0.0

        # Iterate over data.
        for data in train_loader:
            # get the inputs
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data
            running_corrects += torch.sum(preds == labels.data).to(
                torch.float32)

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        val_acc = val_model(model, val_loader)
        if best_acc < val_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Valid Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def create_resnet50(device="cpu"):
    # get model and replace the original fc layer with my own fc layer
    model = models.resnet50(pretrained=True)  # load pretrained model
    for parma in model.parameters():
        parma.requires_grad = False
    num_ftrs = model.fc.in_features  # the number of features
    model.fc = nn.Linear(num_ftrs, 5)  #
    return model.to(device)


def create_vgg16(device="cpu"):
    model = models.vgg16(pretrained=True)  # load pretrained model
    for parma in model.parameters():
        parma.requires_grad = False
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 5))
    return model.to(device)


def create_mymodel(device="cpu"):
    from my_model import ResNet18
    model = ResNet18()
    print(model)
    return model.to(device)


def gen_model(model, train_loader, val_loader):
    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=0.001,
                             momentum=0.9)

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                           step_size=5,
                                           gamma=0.1)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler,
    )
    return model


if __name__ == "__main__":
    # find if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running on gpu" if torch.cuda.is_available() else "running on cpu")

    # vgg16 = gen_model(create_vgg16(device), train_loader, val_loader)
    # FC_io.save_model(vgg16, "vgg16")
    # FC_io.predict(vgg16, FC_io.test_loader())

    # my_model = gen_model(
    #     create_mymodel(device), FC_io.train_loader(1), FC_io.test_loader()
    # )
    # FC_io.save_model(my_model, "my_model")
    # FC_io.predict(my_model, FC_io.test_loader())

    ensemble_models = []
    k_folder = 10
    for i in range(k_folder):
        ensemble_models.append(FC_io.load_model("resnet50-{}".format(i)))
    FC_io.predict(ensemble_models, FC_io.test_loader())
"""
    ensemble_models = []
    k_folder = 10
    loader = FC_io.train_loader(k_folder)
    for i in range(k_folder):
        train_loader, val_loader = loader[i]
        resnet50 = gen_model(create_resnet50(device), train_loader, val_loader)
        FC_io.save_model(resnet50, "resnet50-{}".format(i))
        ensemble_models.append(resnet50)
    FC_io.predict(ensemble_models, FC_io.test_loader())
"""
