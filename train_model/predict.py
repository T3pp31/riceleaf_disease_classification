import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Union, List, Dict, Optional
import os
from pathlib import Path

import config
import model
from dataset import RiceLeafDataset

class RiceLeafPredictor:
    def __init__(
        self, 
        model_path: str = None,
        model_name: str = None,
        device: str = None
    ):
        self.device = device or config.DEVICE
        self.model_path = model_path or config.MODEL_PATH
        self.class_names = config.CLASS_NAMES
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            self.model_name = checkpoint.get('model_name', config.MODEL_NAME)
        else:
            state_dict = checkpoint
            self.model_name = model_name or config.MODEL_NAME
        
        self.model = model.build_model(
            model_name=self.model_name,
            pretrained=False,
            freeze_base=False
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = self._get_inference_transform()
    
    def _get_inference_transform(self):
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE[0] + 32, config.IMG_SIZE[1] + 32)),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def predict_single(
        self, 
        image: Union[str, Path, Image.Image, np.ndarray],
        return_probs: bool = False
    ) -> Dict[str, Union[str, float, Dict[str, float]]]:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        prediction = {
            'class': self.class_names[predicted.item()],
            'confidence': confidence.item()
        }
        
        if return_probs:
            all_probs = {}
            for idx, class_name in enumerate(self.class_names):
                all_probs[class_name] = probs[0, idx].item()
            prediction['probabilities'] = all_probs
        
        return prediction
    
    def predict_batch(
        self, 
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 32,
        return_probs: bool = False
    ) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        predictions = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_tensors = []
            
            for img in batch_images:
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert('RGB')
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                
                tensor = self.transform(img)
                batch_tensors.append(tensor)
            
            batch_input = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_input)
                probs = F.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probs, 1)
            
            for j in range(len(batch_images)):
                prediction = {
                    'class': self.class_names[predicted[j].item()],
                    'confidence': confidences[j].item()
                }
                
                if return_probs:
                    all_probs = {}
                    for idx, class_name in enumerate(self.class_names):
                        all_probs[class_name] = probs[j, idx].item()
                    prediction['probabilities'] = all_probs
                
                predictions.append(prediction)
        
        return predictions
    
    def predict_directory(
        self, 
        directory: Union[str, Path],
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
        batch_size: int = 32,
        return_probs: bool = False
    ) -> Dict[str, Dict[str, Union[str, float, Dict[str, float]]]]:
        directory = Path(directory)
        image_files = []
        
        for ext in extensions:
            image_files.extend(directory.glob(f'*{ext}'))
            image_files.extend(directory.glob(f'*{ext.upper()}'))
        
        predictions = self.predict_batch(
            [str(f) for f in image_files], 
            batch_size=batch_size,
            return_probs=return_probs
        )
        
        results = {}
        for img_file, pred in zip(image_files, predictions):
            results[str(img_file)] = pred
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Rice Leaf Disease Prediction')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('--model', default=None, help='Path to model checkpoint')
    parser.add_argument('--model-name', default=None, help='Model architecture name')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for batch prediction')
    parser.add_argument('--show-probs', action='store_true', help='Show all class probabilities')
    parser.add_argument('--device', default=None, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    predictor = RiceLeafPredictor(
        model_path=args.model,
        model_name=args.model_name,
        device=args.device
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = predictor.predict_single(str(input_path), return_probs=args.show_probs)
        print(f"\nPrediction for {input_path.name}:")
        print(f"Class: {result['class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        if args.show_probs:
            print("\nClass Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
    
    elif input_path.is_dir():
        results = predictor.predict_directory(
            input_path,
            batch_size=args.batch_size,
            return_probs=args.show_probs
        )
        
        print(f"\nPredictions for {len(results)} images in {input_path}:")
        for img_path, result in results.items():
            print(f"\n{Path(img_path).name}:")
            print(f"  Class: {result['class']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            
            if args.show_probs:
                print("  Probabilities:")
                for class_name, prob in result['probabilities'].items():
                    print(f"    {class_name}: {prob:.4f}")
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")

if __name__ == '__main__':
    main()