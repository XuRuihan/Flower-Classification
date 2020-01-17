# ----------------------------------------------------------------------
# Title:    FC_io.py
# Details:  Flower classification input and output
#
# Author:   Ruihan Xu
# Created:  2020/01/16
# Modified: 2020/01/17
# ----------------------------------------------------------------------

from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision import transforms, datasets
from PIL import Image
from scipy.stats import mode
import os
import csv
import torch

class2label = {
    "daisy": 0,
    "dandelion": 1,
    "rose": 2,
    "sunflower": 3,
    "tulip": 4
}


# 默认读取图像的函数
def default_loader(path):
    return Image.open(path).convert('RGB')


# 自定义读取测试集图像的类
class ImageLabels(Dataset):
    def __init__(self, root, label_csv, transform=None, loader=default_loader):
        file = open(label_csv)
        csv_reader = csv.reader(file)
        # 从csv文件中读取标签。标签是手动标注的，不完全正确，只用于测试，不用于训练
        labels = [(line[0], line[1]) for line in csv_reader]
        file.close()
        self.root = root
        self.labels = labels[1:]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.labels[index]
        img = self.loader(os.path.join(self.root, fn + '.jpg'))
        if self.transform is not None:
            img = self.transform(img)
        return img, class2label[label]

    def __len__(self):
        return len(self.labels)


# 加载训练数据集，并划分验证集；如果k_folder == 1，则不划分验证集
def train_loader(k_folder):
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        # transforms.TenCrop(224),
        # transforms.Lambda(
        #     lambda crops: torch.stack(
        #         [transforms.ToTensor()(crop) for crop in crops]
        #     )
        # ),
        transforms.Normalize([0.4591, 0.4197, 0.3009],
                             [0.2960, 0.2657, 0.2885])
    ])
    full_dataset = datasets.ImageFolder('data/train', train_transforms)
    if k_folder == 1:
        return DataLoader(full_dataset,
                          batch_size=8,
                          shuffle=True,
                          num_workers=4)
    full_size = len(full_dataset)
    k_sizes = [int(1.0 / k_folder * full_size) for i in range(k_folder - 1)]
    k_sizes.append(full_size - int(1.0 / k_folder * full_size) *
                   (k_folder - 1))
    k_sets = random_split(full_dataset, k_sizes)
    k_loaders = []
    for i in range(k_folder):
        train_datasets = []
        for j in range(k_folder):
            if j != i:
                train_datasets.append(k_sets[j])
            else:
                val_datasets = k_sets[j]
        k_loaders.append([
            DataLoader(ConcatDataset(train_datasets),
                       batch_size=8,
                       shuffle=True,
                       num_workers=4),
            DataLoader(val_datasets, batch_size=8, shuffle=True, num_workers=4)
        ])
    return k_loaders


# 加载测试数据集
def test_loader():
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        # transforms.TenCrop(224),
        # transforms.Lambda(
        #     lambda crops: torch.stack(
        #         [transforms.ToTensor()(crop) for crop in crops]
        #     )
        # ),
        transforms.Normalize([0.4504, 0.4161, 0.2915],
                             [0.2966, 0.2652, 0.2840])
    ])  # 测试集的transform
    test_datasets = ImageLabels('data/test',
                                'data/label.csv',
                                transform=test_transforms)
    return DataLoader(test_datasets,
                      batch_size=4,
                      shuffle=False,
                      num_workers=4)


# 在测试集上测试（集成）模型
def predict(ensemble_models, test_dataloaders):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble_preds = []
    # 每个模型在测试集上测试
    for model in ensemble_models:
        model.eval()

        single_preds = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloaders):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs, preds = torch.max(outputs, 1)
                # 保存测试结果
                single_preds = single_preds + [
                    preds[j].item() for j in range(inputs.size()[0])
                ]
        ensemble_preds.append(single_preds)
    # 转置保存结果的二维数组，方便 Ostracism
    ensemble_preds = list(map(list, zip(*ensemble_preds)))

    final_preds = [  # Ostracism
        mode(pred)[0][0] for pred in ensemble_preds
    ]
    out2csv(final_preds)


# 保存模型
def save_model(model, filename):
    model_path = 'model/' + filename + '.pkl'
    torch.save(model, model_path)


# 加载模型
def load_model(filename):
    model_path = 'model/' + filename + '.pkl'
    return torch.load(model_path)


# 将预测结果输出到csv文件
def out2csv(final_preds):
    # 定义数字特征到标签的转换
    label2class = {
        0: "daisy",
        1: "dandelion",
        2: "rose",
        3: "sunflower",
        4: "tulip"
    }
    # 将预测的结果写入csv文件中
    file = open('data/res.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(('Id', 'Expected'))
    for i, label in enumerate(final_preds):
        writer.writerow(i, label2class[label])
    file.close()


if __name__ == "__main__":
    test_dataloader = test_loader()
    for inputs, labels in test_dataloader:
        print(len(inputs))
        print(len(labels))
