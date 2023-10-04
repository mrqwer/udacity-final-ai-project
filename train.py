from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image
import json
import seaborn as sb
import argparse
from collections import OrderedDict
from predict import set_device

def parse():
    parser = argparse.ArgumentParser(description = "Parser for training the model")
    parser.add_argument('data_dir', nargs="+", help = 'Dataset director|Required', type = str)
    parser.add_argument('--save_dir',help = 'Directory to save | optional', default = './' , type = str)
    parser.add_argument('--arch', help = 'Architecture | Options = [vgg16, densenet121]', default='vgg16', type = str)
    parser.add_argument('--lr', help ='Learning rate', default = 0.001, type = float)
    parser.add_argument('--hidden_units', help = "A hidden layer's units | 4000", default = 4000, type = int)
    parser.add_argument('--epochs', help = 'Epochs', default = 5, type = int)
    parser.add_argument('--gpu', help = "'GPU' usage | (yes/no)", default = 'yes', type = str)
    parser.add_argument('--dropout', help = 'Dropout rate', default = 0.2, type=float)

    args = parser.parse_args()

    return args

def transform_and_load(args):
    data_dir = args["data_dir"][0]
    train_dir, valid_dir, test_dir = [data_dir+s for s in ["/train", "/valid", "/test"]]

    normalizer = transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
    data_transforms = {
        'train' : transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalizer]),

        'test' : transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalizer]),

        'validation' : transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalizer])}

    data = {
        'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }

    loaders = {
        "train": torch.utils.data.DataLoader(data["train"], batch_size=32, shuffle=True),
        "test": torch.utils.data.DataLoader(data["test"], batch_size=32, shuffle=True),
        "validation": torch.utils.data.DataLoader(data["validation"], batch_size=32, shuffle=True)
    }

    return data, loaders

def build_model(args):
    if args["arch"] == 'vgg16':
        model = models.vgg16(pretrained=True)
        args["input_size"] = 25088
    elif args["arch"] == 'densenet121':
        model = models.densenet121(pretrained=True)
        args["input_size"] = 1024
    for param in model.parameters():
        param.requires_grad = False

    dropout_rate = args["dropout"]
    hidden_size = args["hidden_units"]
    args["output_size"] = 102
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(args["input_size"], hidden_size)),
                            ('relu1', nn.ReLU()),
                            ('dropout', nn.Dropout(dropout_rate)),
                            ('fc3', nn.Linear(hidden_size, args["output_size"])),
                            ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    return model

def optimizer_criterion(model,args):
    optimizer = optim.Adam(model.classifier.parameters(),lr=args["lr"])
    criterion = nn.NLLLoss()
    return optimizer, criterion

def train_model(model, args, loaders, criterion, optimizer):
    model.to(args["device"])
    epochs = args["epochs"]
    running_loss = 0
    steps = 0
    print_every = 10
    device = args["device"]
    with open("log.txt", "w") as f:
        for epoch in range(epochs):
            running_loss = 0
            running_corrects = 0
            total_samples = 0
            model.train()

            for index, (inputs, labels) in enumerate(loaders["train"]):
                steps += 1

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

                if steps % print_every == 0:
                    model.eval()
                    validation_loss = 0
                    validation_accuracy = 0

                    with torch.no_grad():
                        for i, (inputs, labels) in enumerate(loaders["validation"]):
                            inputs, labels = inputs.to(device), labels.to(device)

                            output = model.forward(inputs)

                            validation_loss += criterion(output, labels).item()

                            ps = torch.exp(output)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    train_accuracy = running_corrects / total_samples
                    validloader = loaders["validation"]
                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Step {steps}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Train accuracy: {train_accuracy:.4f}.. "
                        f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                        f"Validation accuracy: {validation_accuracy/len(validloader)}")

                    f.write(f"Epoch {epoch+1}/{epochs}.. "
                        f"Step {steps}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Train accuracy: {train_accuracy:.4f}.. "
                        f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                        f"Validation accuracy: {validation_accuracy/len(validloader)}")
                    f.write('\n')
                    running_loss = 0
                    model.train()
    f.close()
    return model

def test_model(model, args, testloader, criterion):
    model.eval()
    accuracy = 0
    test_loss = 0
    correct = 0
    total = 0
    device = args["device"]

    with torch.no_grad():
        for inputs, labels in testloader:
            model.to(device)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_test_loss = test_loss / len(testloader)

    test_accuracy = correct / total
    print(f"Test Loss: {average_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

def save_model(model, args, optimizer, data_sets):
    model.class_to_idx = data_sets['train'].class_to_idx

    checkpoint = {
        'arch': args["arch"],
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': args["epochs"],
        'input_size': args["input_size"],
        'learning_rate': args["lr"],
        'output_size': args["output_size"],
    }
    torch.save(checkpoint, str(args["save_dir"] + 'checkpoint1.pth'))
    return

def main():
    torch.cuda.empty_cache()
    args = vars(parse())
    print(args)
    data, loaders  = transform_and_load(args)
    set_device(args)
    model = build_model(args)
    optimizer, criterion = optimizer_criterion(model, args)
    model = train_model(model, args, loaders, criterion, optimizer)
    test_model(model, args, loaders["test"], criterion)
    save_model(model, args, optimizer, data)

if __name__ == '__main__':
    main() 
