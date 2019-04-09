# LogBarrier adversarial attack
This repository implements the logarithmic barrier adversarial attack, described in
[The LogBarrier adversarial attack: making effective use of decision boundary information](https://arxiv.org/abs/1903.10396).

The logarithmic barrier adversarial attack begins with a mis-classified image.
It then and iteratively minimizes distance to a clean unperturbed image, while
enforcing model mis-classification, using the logarithmic barrier function.
The LogBarrier attack strength is its ability to adversarially perturb *all*
images with a relatively small norm, compared to other attack methods (like for
example PGD).  The LogBarrier attack works well in both the Euclidean and
maximum norms.

## Details
Written in Python 3 and PyTorch 1.0.

An example attack, on a pretrained CIFAR10 model, is provided in
`cifar10-example/run_attack.py`. The attack itself is implemented as a class in
`logbarrier.py`.

### Citation
If you find the LogBarrier attack useful in your scientific work, please cite as
```
@article{finlay2019logbarrier,
  title={The {L}og{B}arrier adversarial attack: making effective use of decision boundary information},
  author={Finlay, Chris and Pooladian, Aram-Alexandre and Oberman, Adam M.},
  journal={arXiv preprint arXiv:1903.10396},
  year={2019},
  url={http://arxiv.org/abs/1903.10396},
  archivePrefix={arXiv},
  eprint={1903.10396},
}
```
