import torch
import pandas as pd
import albumentations as A
import numpy as np
from tqdm import tqdm
import os
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from dataset import CustomDataLoader
from utils import collate_fn


def inference():
    dataset_path = './input/data'
    test_path = os.path.join(dataset_path,'test.json')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34", 
        encoder_weights="imagenet",
        in_channels=3,
        classes=11
    )
    test_transform = A.Compose([
                           ToTensorV2()
                           ])
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=16,
                                          num_workers=4,
                                          collate_fn=collate_fn)
    model_path = './saved/best_mIoU.pt'
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    submission = pd.read_csv('sample_submission.csv', index_col=None)
    
    size = 256
    transform = A.Compose([A.Resize(size,size)])
    print('Start prediction')

    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            model = model.to(device)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
            
            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i['file_name'] for i in image_infos])

        print("End prediction")
        file_names = [y for x in file_name_list for y in x]

        for file_name, string in zip(file_names, preds_array):
            submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

        if not os.path.isdir('./submission'):
            os.mkdir('./submission')
        submission.to_csv("./submission/submission.csv", index=False)


if __name__ == '__main__':
    inference()

    
   

