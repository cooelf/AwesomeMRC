# AwesomeMRC

*working in progress

This repo is our summary and playground for MRC.  More features are coming.

In this work, MRC model is regarded as two-stage Encoder-Decoder architecture. Our main attempts are shared in this repo. 

![](figures/overview.png)

### Decoder:

The implementation is based on [Transformers](https://github.com/huggingface/transformers) v2.3.0. 

#### Answer Verification

**1）Multitask-style verification**

We evaluate different loss functions 

*cross-entropy* (`run_squad_av.py`)

*binary cross-entropy* (`run_squad_av_bce.py`)

*mse regression*  (`run_squad_avreg.py`)

**2) External verification**

Train an external verifier (`run_cls.py`)

#### Matching Attention

*Cross Attention* (`run_squad_seq_trm.py`)

*Matching Attention* (`run_squad_seq_sc.py`)

#### Answer Dependency

Model answer dependency (start + seq -> end) (`run_squad_dep.py`)

#### Retrospective Reader

1) train a sketchy reader (`sh_albert_cls.sh`)

2) train an intensive reader (`sh_albert_av.sh`)

3) rear verification: merge the prediction for final answer (`run_verifier.py`)

### Citation

```
@article{zhang2020retrospective,
  title={Retrospective reader for machine reading comprehension},
  author={Zhang, Zhuosheng and Yang, Junjie and Zhao, Hai},
  journal={arXiv preprint arXiv:2001.09694},
  year={2020}
}
```
### Contact

Feel free to email zhangzs [at] sjtu.edu.cn if you have any questions.

