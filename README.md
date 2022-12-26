# InstanceSegmentation using fasterrcnn model with mobilenet_v2 as a backbone and MultiScaleRoIAlign

### Create virtual environment
```python
conda create -n <ENV_NAME> python=3.9
conda activate <ENV_NAME>
pip install -r requirements.txt
```

### Run training 
```python
python train.py --batch_size=64 --lr=3e-4 --model_name="efficientnet_b3a"
```
