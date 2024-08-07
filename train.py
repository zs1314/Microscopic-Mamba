import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.tensorboard import SummaryWriter
from utils import create_lr_scheduler,EarlyStopping,seed
from microscopicmamba import VSSM as microscopicmamba
# set random seed
seed(0)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     # transforms.RandomHorizontalFlip(p=0.5),
                                     # transforms.RandomVerticalFlip(p=0.5),
                                     # transforms.RandomRotation(degrees=(45,90)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.7016, 0.5726, 0.7400), (0.1949, 0.2388, 0.1536))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.7016, 0.5726, 0.7400), (0.1949, 0.2388, 0.1536))])}

    # You need to run getmean.py to obtain the means and standard deviations for better training.

    # replace your dataset path
    train_set_path = r"/root/MedMfc/train"
    val_set_path = r"/root/MedMfc/val"

    if not os.path.exists(train_set_path):
        print(f"Error: Train set path '{train_set_path}' does not exist.")
        return

    if not os.path.exists(val_set_path):
        print(f"Error: Validation set path '{val_set_path}' does not exist.")
        return

    batch_size = 58
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # train_data_init
    train_dataset = datasets.ImageFolder(root=train_set_path, transform=data_transform["train"])
    train_num = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, worker_init_fn=np.random.seed(0))

    # val_data_init
    validate_dataset = datasets.ImageFolder(root=val_set_path, transform=data_transform["val"])

    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, worker_init_fn=np.random.seed(0))

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    num_classes = len(train_dataset.classes)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    model_name = 'MedMf_microscopicmamba_t1'
    net = microscopicmamba(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=num_classes)   # microscopicmamba_t
    # net = microscopicmamba(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=num_classes)   # microscopicmamba_s
    # net = microscopicmamba(depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024], num_classes=num_classes)   # microscopicmamba_b

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    epochs = 200
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs,
                                       warmup=True, warmup_epochs=10)

    best_acc = 0.0
    save_path = './{}.pth'.format(model_name)
    train_steps = len(train_loader)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir='./runs/{}'.format(model_name))

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=30)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            # update lr
            lr_scheduler.step()
            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # log the training loss
        writer.add_scalar('Training Loss', running_loss / train_steps, epoch)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_loss = 0.0  # accumulate validation loss
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        val_loss /= len(validate_loader)

        # Calculate metrics
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        """multiple classes"""
        # # Binarize the labels for AUC calculation
        # all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
        # auc = roc_auc_score(all_labels_bin, all_probs, average='weighted', multi_class='ovr')
        """"""

        """binary classes"""
        all_probs = np.array(all_probs)
        all_probs = all_probs[:, 1]  # 只取正类的概率
        auc = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
        """"""

        print(
            '[epoch %d] train_loss: %.3f  val_loss: %.3f  val_accuracy: %.3f  val_precision: %.3f  val_recall: %.3f  '
            'val_f1: %.3f  val_auc: %.3f' %
            (epoch + 1, running_loss / train_steps, val_loss, val_accurate, precision, recall, f1, auc))

        # Log validation metrics
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_accurate, epoch)
        writer.add_scalar('Validation Precision', precision, epoch)
        writer.add_scalar('Validation Recall', recall, epoch)
        writer.add_scalar('Validation F1 Score', f1, epoch)
        writer.add_scalar('Validation AUC', auc, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)

        # Save the best model
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print("save new best model")

        # Check early stopping
        early_stopping(val_loss, net, save_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    writer.close()
    print('Finished Training')


if __name__ == '__main__':
    main()
