FastReID is a research platform that implements state-of-the-art re-identification algorithms. It is a ground-up rewrite of the previous version, [reid strong baseline](https://github.com/michuanhaohao/reid-strong-baseline).

## What's New

- [Sep 2021] [DG-ReID](https://github.com/xiaomingzhid/sskd) is updated, you can check the [paper](https://arxiv.org/pdf/2108.05045.pdf).
- [June 2021] [Contiguous parameters](https://github.com/PhilJd/contiguous_pytorch_params) is supported, now it can
  accelerate ~20%.
- [May 2021] Vision Transformer backbone supported, see `configs/Market1501/bagtricks_vit.yml`.
- [Apr 2021] Partial FC supported in [FastFace](projects/FastFace)!
- [Jan 2021] TRT network definition APIs in [FastRT](projects/FastRT) has been released! 
Thanks for [Darren](https://github.com/TCHeish)'s contribution.
- [Jan 2021] NAIC20(reid track) [1-st solution](projects/NAIC20) based on fastreid has been released！
- [Jan 2021] FastReID V1.0 has been released！🎉
  Support many tasks beyond reid, such image retrieval and face recognition. See [release notes](https://github.com/JDAI-CV/fast-reid/releases/tag/v1.0.0).
- [Oct 2020] Added the [Hyper-Parameter Optimization](projects/FastTune) based on fastreid. See `projects/FastTune`.
- [Sep 2020] Added the [person attribute recognition](projects/FastAttr) based on fastreid. See `projects/FastAttr`.
- [Sep 2020] Automatic Mixed Precision training is supported with `apex`. Set `cfg.SOLVER.FP16_ENABLED=True` to switch it on.
- [Aug 2020] [Model Distillation](projects/FastDistill) is supported, thanks for [guan'an wang](https://github.com/wangguanan)'s contribution.
- [Aug 2020] ONNX/TensorRT converter is supported.
- [Jul 2020] Distributed training with multiple GPUs, it trains much faster.
- Includes more features such as circle loss, abundant visualization methods and evaluation metrics, SoTA results on conventional, cross-domain, partial and vehicle re-id, testing on multi-datasets simultaneously, etc.
- Can be used as a library to support [different projects](projects) on top of it. We'll open source more research projects in this way.
- Remove [ignite](https://github.com/pytorch/ignite)(a high-level library) dependency and powered by [PyTorch](https://pytorch.org/).

We write a [fastreid intro](https://l1aoxingyu.github.io/blogpages/reid/fastreid/2020/05/29/fastreid.html) 
and [fastreid v1.0](https://l1aoxingyu.github.io/blogpages/reid/fastreid/2021/04/28/fastreid-v1.html) about this toolbox.

## Changelog

Please refer to [changelog.md](CHANGELOG.md) for details and release history.

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

See [GETTING_STARTED.md](GETTING_STARTED.md).

Learn more at out [documentation](https://fast-reid.readthedocs.io/). And see [projects/](projects) for some projects that are build on top of fastreid.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Fastreid Model Zoo](MODEL_ZOO.md).

## Deployment

We provide some examples and scripts to convert fastreid model to Caffe, ONNX and TensorRT format in [Fastreid deploy](tools/deploy).

## License

Fastreid is released under the [Apache 2.0 license](LICENSE).

## Citing FastReID

If you use FastReID in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```BibTeX
@article{he2020fastreid,
  title={FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
  journal={arXiv preprint arXiv:2006.02631},
  year={2020}
}
```

# 프로젝트 목적: 기업에서 제시한 세가지 Fast-reid base 모델 중 서비스 사용에 적합한 모델을 비교분석하여 선정
  
## Dataset: 차량 외관 영상 데이터 ****- AI_Hub****
**Data information**
- 차량 외관(차종, 연식, 색상, 트림)과 14개 파트(프론트범퍼, 리어범퍼, 타이어, A필러, C필러, 사이드미러, 앞도어, 뒷도어, 라디에이터그릴, 헤드램프, 리어램프, 보닛, 트렁크, 루프)를 식별할 수 있는 AI 학습용 데이터셋.
- 데이터 형식은 jpg와 이미지에 대한 json이 있음.
- 학습은 진행하지 않았기 때문에 validation set만을 사용함

## 비교 분석 방법
- 색깔 (흰색, 검은색, 회색)
- 방향 (앞, 뒤, 옆)

# 결과: veri-wild 모델을 선정
## 용량 문제로 데이터셋은 올리지 못함
## 해당 모델로 YOLOv4-Deepsort를 이용해 두 CCTV 영상의 차량들을 매핑하는 것이 
