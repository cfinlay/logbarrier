This is an example use of the LogBarrier attack on a pre-trained (undefended) CIFAR10 model. Run the script with

```python run_attack.py --data-dir path/to/cifar10-data/```

Adversarial distances will be saved to a npz file in a new folder `attack0/`. 
To see all options, pass the `--help` flag. Adversarial images may be optionally saved as well.
