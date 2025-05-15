# IMLO_Coursework

This project trains and evaluates deep learning models (VGG16 and ResNet) on the CIFAR-10 dataset using PyTorch. It supports flexible hyperparameter tuning, modular training, testing with detailed evaluation reports, and clean visualization.

---

## 1. Create Conda Environment
```bash
conda env create -f environment.yml -n IMLO-Coursework
conda activate IMLO-Coursework
```
---

## 2. Hyperparameter Tuning (Optional)
Run a grid search across multiple configurations:
```bash
python3 tune.py
```
This explores combinations of:

- Model type: vgg16, resnet
- Activation: ReLU, LeakyReLU
- Optimizer: adam, sgd, adamw
- Learning rates: 0.001, 0.0005, 0.001
- Batch sizes: 64, 128, 256
- Loss: cross_entropy, focal

Results are saved to tuning_results.csv.

---

## 3. Training

Train the model using
```bash
python3 train.py
```
You can customize the training configuration in train.py:

```bash
config = {
    "model": "vgg16",          
    "activation": nn.ReLU(),      
    "loss": "cross_entropy",      
    "optimizer": "adam",          
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 60,
    "patience": 10,
    "save_path": "model.pth"
}
```
Outputs:

Trained model saved to model.pth

Learning curves saved as training_metrics.png

---

## 4. Test
Evaluate the trained model using:

```bash
python3 test.py
```

This will:

Report loss and accuracy on the test set

Print and save a classification report (test_report.csv)

Save a confusion matrix (test_confusion_matrix.png)

Visualize correct predictions (test_correct.png)

Visualize incorrect predictions (test_incorrect.png)

