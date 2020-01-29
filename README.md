# LogBarrier adversarial attack
This repository implements the logarithmic barrier adversarial attack, described in
[The LogBarrier adversarial attack: making effective use of decision boundary information](https://arxiv.org/abs/1903.10396).

The logarithmic barrier adversarial attack begins with a mis-classified image.
It then iteratively minimizes distance to a clean unperturbed image, while
enforcing model mis-classification, using the logarithmic barrier function.
The strength of the LogBarrier attack is its ability to adversarially perturb *all*
images with a relatively small norm, compared to other attack methods (say for
example PGD).  The LogBarrier attack works well in both the Euclidean and
maximum norms.

**Update 2019-08-21** We have updated the LogBarrier method for any non-smooth metric (including state-of-the-art L0 attacks). Please see the repo [ProxLogBarrierAttack](https://github.com/APooladian/ProxLogBarrierAttack), and accompanying [preprint](https://arxiv.org/abs/1908.01667). 

## Details
Written in Python 3 and PyTorch 1.0.

An example attack, on a pretrained CIFAR10 model, is provided in
`cifar10-example/run_attack.py`. The attack itself is implemented as a class in
`logbarrier.py`.

### Citation
If you find the LogBarrier attack useful in your scientific work, please cite as
```
@InProceedings{Finlay_2019_ICCV,
  author = {Finlay, Chris and Pooladian, Aram-Alexandre and Oberman, Adam},
  title = {The LogBarrier Adversarial Attack: Making Effective Use of Decision Boundary Information},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}
} 
```
