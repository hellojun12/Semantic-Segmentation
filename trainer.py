import numpy as np
import torch
from torch.cuda import device_count
import os.path as osp
import wandb
from wandb_img import wb_mask

from utils import label_accuracy_score, add_hist

class Trainer(object):

    def __init__(self, num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, batch_size, device):

        self.num_epochs = num_epochs
        self.model = model
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.saved_dir = saved_dir
        self.val_every = val_every
        self.batch_size = batch_size
        self.device = device

        self.n_class = 11
        self.best_loss = 9999999
        self.best_mIou = 0.0
        self.log_headers = [
            'epoch',
            'step',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc'
        ]

        with open(osp.join(self.saved_dir, 'log.csv'),'w') as f:
            f.write(','.join(self.log_headers) +'\n')
        

    def train(self):
        print('Start training..')
        model = self.model
        best_loss = 9999999
        best_mIoU = 0.0

        for epoch in range(self.num_epochs):
            model.train()

            hist = np.zeros((self.n_class, self.n_class))
            for step, (images, masks, _) in enumerate(self.data_loader):
                images = torch.stack(images)
                masks = torch.stack(masks).long()

                images, masks = images.to(self.device).float(), masks.to(self.device)

                model = model.to(self.device)

                outputs = model(images)

                loss = self.criterion(outputs, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()

                metrics = []
                hist = add_hist(hist, masks, outputs, n_class=self.n_class)
                acc, acc_cls, mIoU, fwavacc, _ = label_accuracy_score(hist)
                metrics.append((acc, acc_cls, mIoU, fwavacc))
                metrics = np.mean(metrics, axis=0)
                metrics = list(map(lambda x : round(x,4), metrics))
                
                if (step + 1) % 25 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{step+1}/{len(self.data_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                    with open(osp.join(self.saved_dir,'log.csv'),'a') as f:
                        log = [epoch + 1, step+1] + [round(loss.item(),4)] + \
                            metrics + [''] * 5 
                        log = map(str,log)
                        f.write(','.join(log) + '\n')

                    wandb.log({"Train Epoch":epoch+1,"Train Loss":round(loss.item(),4), 'Train mIoU': round(mIoU,4)})
            if self.val_every != 0:
                if (epoch + 1) % self.val_every == 0:
                    avrg_loss, avrg_mIoU = self._validation(epoch + 1, model)
                    if avrg_mIoU > best_mIoU:
                        print(f"Best performance at epoch: {epoch + 1}")
                        print(f"Save model in {self.saved_dir}")
                        best_mIoU = avrg_mIoU
                        check_point = {'net': model.state_dict()}
                        output_path = osp.join(self.saved_dir, 'best_mIoU.pt')
                        torch.save(model, output_path)
                    if avrg_loss < best_loss:
                        print(f"Best performance at epoch: {epoch + 1}")
                        print(f"Save model in {self.saved_dir}")
                        best_loss = avrg_loss
                        check_point = {'net': model.state_dict()}
                        output_path = osp.join(self.saved_dir, 'best_loss.pt')
                        torch.save(model, output_path)

    def _validation(self, epoch, model):
        category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

        print(f'Start validation #{epoch}')
        model.eval()

        with torch.no_grad():
            total_loss = 0
            cnt = 0
            mask_list = []
            hist = np.zeros((self.n_class, self.n_class))
            for step, (images, masks, _) in enumerate(self.val_loader):

                images = torch.stack(images)
                masks = torch.stack(masks).long()

                images, masks = images.to(self.device).float(), masks.to(self.device)

                model = model.to(self.device)

                outputs = model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss
                cnt += 1

                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()

                #wandb image save
                if epoch%5 == 0 and step < 2:
                    
                    images= images.detach().cpu().numpy().transpose([0,2,3,1])
                    for i in range(self.batch_size):
                        
                        mask_list.append(wb_mask(images[i,:,:,:], outputs[i,:,:], masks[i,:,:], category_names))

                hist = add_hist(hist, masks, outputs, n_class=self.n_class)

            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            metrics = [acc, acc_cls, mIoU, fwavacc]
            metrics = list(map(lambda x: round(x,4), metrics))
            IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU, category_names)]

            avrg_loss = total_loss /cnt
            print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                    mIoU: {round(mIoU, 4)}')
            print(f'IoU by class : {IoU_by_class}')

            category_dict = {}
            for cl in IoU_by_class:
                category_dict.update(cl)
                
            wandb.log({"Category_IoU/": category_dict}) 
            wandb.log({"Validation_epoch":epoch, "Average Loss":round(avrg_loss.item(), 4), "Validation Accuracy" : round(acc, 4), 'Validation mIoU': round(mIoU,4)})
            wandb.log({"Examples":mask_list})
            with open(osp.join(self.saved_dir, 'log.csv'),'a') as f:
                log = [epoch, 'Val'] + [''] * 5 + \
                    [round(avrg_loss.item(),4)] + metrics 
                log = map(str, log)
                f.write(','.join(log) + '\n')
        return avrg_loss, round(mIoU, 4)