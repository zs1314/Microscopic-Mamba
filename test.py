import os
import sys
import numpy as np
import torch
from sklearn.preprocessing import label_binarize
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score
from utils import seed
from microscopicmamba import VSSM as microscopicmamba
seed(0)


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.7016, 0.5726, 0.7400), (0.1949, 0.2388, 0.1536))])

    test_set_path = r"/root/MedMfc/test"
    if not os.path.exists(test_set_path):
        print(f"Error: Test set path '{test_set_path}' does not exist.")
        return

    test_dataset = datasets.ImageFolder(root=test_set_path,
                                        transform=data_transform)
    test_num = len(test_dataset)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw,
                                              worker_init_fn=np.random.seed(0))
    print("using {} images for testing.".format(test_num))

    model_name = 'MedMf_microscopicmamba_t1'
    num_classes = len(test_dataset.classes)
    net = microscopicmamba(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=num_classes)   # microscopicmamba_t
    # net = microscopicmamba(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=num_classes)   # microscopicmamba_s
    # net = microscopicmamba(depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024], num_classes=num_classes)   # microscopicmamba_b
    net.to(device)

    # Load model weights
    weight_path = './{}.pth'.format(model_name)
    if not os.path.exists(weight_path):
        print(f"Error: Model weight file '{weight_path}' does not exist.")
        return

    try:
        net.load_state_dict(torch.load(weight_path))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    net.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    try:
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for data in test_bar:
                images, labels = data
                outputs = net(images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    """ multiple classes"""
    # # Binarize the output for multi-class AUC calculation
    # all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
    # # Calculate AUC for each class
    # auc_per_class = roc_auc_score(all_labels_bin, all_probs, average=None)
    # auc_weighted = roc_auc_score(all_labels_bin, all_probs, average='weighted', multi_class='ovr')
    """"""

    """binary classes"""
    all_probs = np.array(all_probs)
    all_probs = all_probs[:, 1]  # 只取正类的概率
    auc_weighted = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
    """"""

    report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
    print('Test Accuracy: {:.4f}'.format(acc))
    print('Test Precision: {:.4f}'.format(precision))
    print('Test Recall: {:.4f}'.format(recall))
    print('Test F1 Score: {:.4f}'.format(f1))
    print('Test Weighted AUC: {:.4f}'.format(auc_weighted))

    """multiple classes"""
    # print("\nAUC per class:")
    # for idx, class_name in enumerate(test_dataset.classes):
    #     print(f'Class {class_name}: AUC: {auc_per_class[idx]:.3f}')
    """"""

    print("\nDetailed Classification Report:")
    print(report)

    # Save results to file
    result_file = 'test_results.txt'.format()
    with open(result_file, 'w') as f:
        f.write('Test Accuracy: {:.4}\n'.format(acc))
        f.write('Test Precision: {:.4f}\n'.format(precision))
        f.write('Test Recall: {:.4f}\n'.format(recall))
        f.write('Test F1 Score: {:.4f}\n'.format(f1))
        f.write('Test Weighted AUC: {:.4f}\n'.format(auc_weighted))
        f.write('\nAUC per class:\n')

        """multiple classes"""
        # for idx, class_name in enumerate(test_dataset.classes):
        #     f.write(f'Class {class_name}: AUC: {auc_per_class[idx]:.3f}\n')
        """"""

        f.write('\nDetailed Classification Report:\n')
        f.write(report)

    print(f"Results saved to {result_file}")


if __name__ == '__main__':
    test()
