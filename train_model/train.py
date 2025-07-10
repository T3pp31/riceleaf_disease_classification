import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
import os
import wandb

import config
import model
from dataset import get_dataloaders, get_transforms

def get_lr_scheduler(optimizer, scheduler_type='cosine'):
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=config.NUM_EPOCHS, 
            eta_min=config.MIN_LR if hasattr(config, 'MIN_LR') else 1e-6
        )
    elif scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    else:
        scheduler = None
    
    return scheduler

def train_epoch(model, train_loader, criterion, optimizer, device, mixup_fn=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        original_labels = labels
        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += original_labels.size(0)
        if mixup_fn is not None:
            # When using Mixup, the labels are one-hot encoded and smoothed.
            # We need to convert them back to class indices to calculate accuracy.
            _, original_labels_idx = torch.max(labels, 1)
            correct += (predicted == original_labels_idx).sum().item()
        else:
            correct += (predicted == original_labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total if total > 0 else 0
    
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def main():
    if config.USE_WANDB:
        wandb.init(
            project=config.WANDB_PROJECT,
            config={
                "learning_rate": config.LEARNING_RATE,
                "epochs": config.NUM_EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "model_name": config.MODEL_NAME,
                "img_size": config.IMG_SIZE,
                "lr_scheduler": config.LR_SCHEDULER,
                "use_timm_augmentation": config.USE_TIMM_AUGMENTATION if hasattr(config, 'USE_TIMM_AUGMENTATION') else False,
                "use_mixup": config.USE_MIXUP if hasattr(config, 'USE_MIXUP') else False,
            },
            dir=config.LOG_DIR
        )

    train_loader, test_loader = get_dataloaders()
    
    net = model.build_model(
        model_name=config.MODEL_NAME,
        pretrained=True,
        freeze_base=config.FREEZE_BASE if hasattr(config, 'FREEZE_BASE') else True
    ).to(config.DEVICE)

    if config.USE_WANDB:
        wandb.watch(net)
    
    mixup_fn = None
    if hasattr(config, 'USE_MIXUP') and config.USE_MIXUP:
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            cutmix_minmax=None,
            prob=0.5,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=0.1,
            num_classes=config.NUM_CLASSES
        )
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY if hasattr(config, 'WEIGHT_DECAY') else 1e-4
    )
    
    scheduler = get_lr_scheduler(
        optimizer, 
        config.LR_SCHEDULER if hasattr(config, 'LR_SCHEDULER') else 'cosine'
    )
    
    best_accuracy = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer, config.DEVICE, mixup_fn)
        val_loss, val_acc = evaluate(net, test_loader, torch.nn.CrossEntropyLoss(), config.DEVICE)
        
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = config.LEARNING_RATE
        
        print(f'Epoch {epoch + 1}/{config.NUM_EPOCHS}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {current_lr:.6f}')

        if config.USE_WANDB:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr,
                "epoch": epoch
            })
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            model_filename = f"{wandb.run.name if config.USE_WANDB and wandb.run else 'best_model'}-epoch{epoch+1}-acc{best_accuracy:.2f}.pth"
            model_save_path = os.path.join(config.MODEL_DIR, model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'model_name': config.MODEL_NAME
            }, model_save_path)
            print(f'  Best model saved to {model_save_path} with accuracy: {best_accuracy:.2f}%')
    
    print('\nTraining completed!')
    print(f'Best validation accuracy: {best_accuracy:.2f}%')

    if config.USE_WANDB:
        wandb.finish()

if __name__ == '__main__':
    main()
