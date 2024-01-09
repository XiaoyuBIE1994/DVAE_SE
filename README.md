# Speech Enhancement based on DVAE

This repository is the official code for the paper:

**Unsupervised Speech Enhancement using Dynamical Variational Auto-Encoders, TASLP, 2022**  
[Xiaoyu Bie](https://team.inria.fr/perception/team-members/xiaoyu-bie/), [Simon Leglaive](https://sleglaive.github.io/index.html), [Xavier Alameda-Pineda](http://xavirema.eu/), [Laurent Girin](http://www.gipsa-lab.grenoble-inp.fr/~laurent.girin/cv_en.html)

**[[Paper](https://arxiv.org/abs/2106.12271)]**


## Dataset
For HumanEva-I and Human3.6M, we follow the instructions in gsps, then put all data into data directory:

* **WSJ-QUT** (paid): it comes from the the subset of original WSJ dataset, which was used in the [CHiME-3 Challenge](https://www.chimechallenge.org/challenges/chime3/data), corrupted by the [QUT noise](https://github.com/qutsaivt/QUT-NOISE)  
* **VoiceBank-DEMAND** (free): download data from their [website](https://datashare.ed.ac.uk/handle/10283/2791), downsample to 16kHz and select p226 & p287 as validation dataset (following [MetricGanU](https://arxiv.org/abs/2110.05866))

## Train


## Test



## Bibtex
If you find this code useful, please star the project and consider citing:

```
@article{bie2022unsupervised,
  title={Unsupervised speech enhancement using dynamical variational autoencoders},
  author={Bie, Xiaoyu and Leglaive, Simon and Alameda-Pineda, Xavier and Girin, Laurent},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={30},
  pages={2993--3007},
  year={2022},
  publisher={IEEE}
}

```
