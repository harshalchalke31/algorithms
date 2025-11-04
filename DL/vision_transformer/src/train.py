import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import yaml
from models import ModelParams
from .dataloader import ViTDataset
from .model import VisionTransformer
import os
import csv
from tqdm import tqdm



class ViTModel():
    def __init__(self):
        # set seed
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # load params
        with open('config.yaml','r') as f:
            config = yaml.safe_load(f)

        self.params = ModelParams(**config['ModelParams'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        # load dataset
        dataset = ViTDataset(BATCH_SIZE=self.params.BATCH_SIZE,
                             IMG_SIZE=self.params.IMAGE_SIZE)
        train_loader = dataset.train_loader()
        test_loader = dataset.test_loader()

        # load model
        model = VisionTransformer(EMBED_DIM=self.params.EMBED_DIMENSION,
                                  NUM_HEADS=self.params.NUM_HEADS,
                                  NUM_CLASSES=self.params.NUM_CLASSES,
                                  DEPTH=self.params.DEPTH,
                                  DROPOUT_RATE=self.params.DROPOUT_RATE,
                                  IMG_SIZE=self.params.IMAGE_SIZE,
                                  IN_CHANNELS=self.params.IN_CHANNELS,
                                  PATCH_SIZE=self.params.PATCH_SIZE,
                                  MLP_RATIO=self.params.MLP_RATIO
                                  ).to(self.device)
        # loss and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15) # best - 0.10
        optimizer = optim.AdamW(params=model.parameters(),
                               lr = self.params.LR,
                               weight_decay=0.15) # best - 0.05
        
        total_steps = int(len(train_loader)* self.params.EPOCHS)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.params.LR,
            total_steps=total_steps,
            pct_start=0.05,  # 5% warmup
            div_factor=10,
            final_div_factor=100,
            anneal_strategy='cos'
        )
        patience_counter =0
        best_loss = float('inf')


        os.makedirs('logs',exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)

        with open('logs/train_logs.csv',mode='w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Valid loss", "Train Accuracy", "Valid Accuracy", "Learning Rate"])

        for epoch in range(self.params.EPOCHS):
            
            model.train()
            train_loss, correct, total = 0,0,0

            for images, labels in tqdm(train_loader, desc=f'Epoch: {epoch +1}/{self.params.EPOCHS} [Training], Patience: {patience_counter}'):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs,labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            avg_train_loss = train_loss/total
            train_acc = 100. * correct/total

            # Test
            model.eval()
            test_loss, test_total, test_correct = 0,0,0
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc=f'Epoch: {epoch +1}/{self.params.EPOCHS} [Validation]'):
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs= model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() *images.size(0)
                    _, predicted = outputs.max(1)
                    test_correct += predicted.eq(labels).sum().item()
                    test_total += labels.size(0)
            
            avg_test_loss = test_loss/test_total
            test_acc = 100. * test_correct / test_total
            current_lr = optimizer.param_groups[0]['lr']
            

            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save(model.state_dict(), 'checkpoints/best_model.pt')
                patience_counter = 0
            else:
                patience_counter+=1

            with open('logs/train_logs.csv',mode='a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_train_loss, avg_test_loss, train_acc, test_acc, current_lr])
        
            if patience_counter >= self.params.PATIENCE:
                print('Early Stopping Triggered')
                break
        




