import torch
import time
from torch import nn
from torchvision import models, transforms, datasets


def create_model(hidden_units=1024, arch='vgg13'):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise Exception("You can only choose either vgg13 or alexnet")
    for param in model.parameters:
        param.requires_grad = False
    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier
    return model


def create_dataloader(train_datadir, valid_datadir):
    colors_mean = [0.485, 0.456, 0.406]
    colors_std = [0.229, 0.224, 0.225]

    train_transforms = datasets.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(colors_mean,
                                                            colors_std)])
    valid_transforms = datasets.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(colors_mean,
                                                            colors_std)])
    
    train_dataset = datasets.ImageFolder(train_datadir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_datadir, transform=valid_transforms)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    return train_dataloader, valid_dataloader, train_dataset.class_to_idx


def train(epochs=3, learning_rate=0.001, gpu=False, train_datadir, valid_datadir, model):
    train_dataloader, valid_dataloader, class_to_idx = create_dataloader(train_datadir, valid_datadir)
    model.class_to_idx = class_to_idx
    if gpu==True and cude.is_available() == False:
        raise Exception("GPU training is not available")
    elif gpu==True:
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.to(device)
    for epoch in range(epochs):
        start = time.time()
        running_loss = 0
        valid_loss = 0
        accuracy = 0
        
        # Training pass
        model.train()
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            images, lables = images.to(device), labels.to(device)
            # Feedforward
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            running_loss += loss
            # Backpropagation
            loss.backward()
            optimizer.step()
        
        # Validation pass
        model.eval()
        for images, labels in valid_dataloader:
            with torch.no_grad():
                # Feedforward
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                valid_loss += loss
                # Accuracy Calculations
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        # Time taken to train this epoch
        time_elapsed = time.time() - start
        # Print Training Results
        print(f"Epoch:{epoch+1}/{epochs}",
            f"Training Loss:{running_loss/len(trainloader):.3f}",
            f"Validation Loss:{valid_loss/len(validationloader):.3f}",
            f"Accuracy:{accuracy*100/len(validationloader):.3f}%",
            f"Epoch Total Time: {time_elapsed//60}m {time_elapsed%60:.0f}s")
    # saving the model right after training
    checkpoint = {"epochs": epochs,
                  "model_state": model.state_dict(),
                  "optimizer_state": optimizer.state_dict(),
                  "classes_to_indices": train_dataset.class_to_idx,
                  "criterion": criterion,
                  "classifier" : classifier}
    # return checkpoint dict
    return checkpoint
    