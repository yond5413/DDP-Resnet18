# Lab 4: Neural Network Implementation

This project implements a neural network using PyTorch and TorchVision. It includes training and evaluation of a ResNet-18 model on the CIFAR-10 dataset. The project allows customization of hyperparameters such as learning rate, optimizer, and computing device. In this instance the goal was to compare the performance of the model utilizing various GPU configurations.

---

## **Requirements**
- Python 3.x
- PyTorch
- TorchVision

Install the required libraries using:
```bash
pip install torch torchvision
```

## **Usage**
Run with Default Parameters
```
python lab4.py
```
## **Run with Custom Parameters**
```
python lab4.py --lr 0.01 --device cuda --opt adam --c7
```

### Experimenting with GPU config
```bash
python3 lab4.py --world_config <val>
```


## **Parameters Desctiption**
| Parameter      | Description                                                                 | Default Value | Options/Constraints                     |
|----------------|-----------------------------------------------------------------------------|---------------|-----------------------------------------|
| `--lr`         | Learning rate                                                               | 0.1           | Must be a float < 1                     |
| `--device`     | Computing device                                                            | `cpu`         | `cpu`, `cuda`                           |
| `--num_workers`| Number of I/O workers                                                       | 2             | Must be an integer                      |
| `--data_path`  | Dataset download directory                                                  | `./data`      | Path to directory                       |
| `--opt`        | Optimizer selection                                                         | `sgd`         | `sgd`, `nesterov`, `adam`, `adagrad`, `adadelta` |
| `--c7`         | Disable batch normalization in ResNet-18                                    | `False`       | Set to `True` to disable batch normalization |
|--world_config	 | GPU configuration for DDP                                                   | 0             |	0  (1 GPU), 1 (2 GPUs), 2 (4 GPUs) |
---

## Project Structure
```lab4/
├── lab4.py               # Main script for training and evaluation
├── README.md             # Project documentation
└── data/                 # Directory for storing the dataset (default)
Additional Files
plotting.py: Generates figures for the report, such as training loss and test accuracy plots.
```

## Example Commands
Train the model with a learning rate of 0.01 on a GPU using the Adam optimizer:

```python lab4.py --lr 0.01 --device cuda --opt adam```
## ***Train the model without batch normalization***:

```python lab4.py --c7```
Output
The script will print the training loss and test accuracy for each epoch. Example output:

```
Epoch 1, Train Loss: 1.2345, Test Accuracy: 0.5678
Epoch 2, Train Loss: 1.1234, Test Accuracy: 0.6789
```

## Notes
Ensure the dataset is downloaded to the correct directory (./data by default).

Experiment with different hyperparameters to observe their impact on performance.

In order to test various GPU configurations I would suggest utilzing a vm from a cloud service provider like AWS/GCP, but keep in mind the cost be very expensive. In my case I had access to a cluster for a course, so I could request resources via shell scripts and send jobs that way.



