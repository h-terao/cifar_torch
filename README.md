# Guide to implement

ここから，div align="center" の一つ上までは実装後削除してください．

## model_kwargs

LightningDataModuleにmodel_kwargsという名前の変数を辞書として登録すると，LightningModuleにその値をキーワード変数として渡すことができます．この機能は，クラス数や入力次元，データの種類などを渡すときに便利です．

```python
class ExampleDataModule(pl.LightningDataModule):
    def __init__(self, ...):
        super().__init__()
        model_kwargs = {
            "n_classes": 10,
            "size": 32,
        }

class ExampleModule(pl.LightningModule):
    def __init__(self, n_classes, size, ...):
        # n_classes and size are given from datamodule, not from config!
        ...
```

## trainer_kwargs

model_kwargsと似たような機能として，trainer_kwargsという名前の辞書を変数としてLightningDataModuleやLightningModuleに登録しておくと，Trainerにその値を渡すことができます．例えば，multiple_trainloader_modeを指定する必要がある時はこの機能が便利です．

## Logging correct metrics

正確にメトリクスを記録するときは，`self.log(..., sync_dist=True, batch_size=batch_size)`のようにしてロギングすると良いです．Torchmetricsを使うよりもこっちのほうが楽な気がします．

## Augmentation

Korniaを使い，GPU上でデータ水増しを行うことで学習全体にかかる時間を短縮できます．
この時，LightningModuleにモデルとして水増し操作を登録すると，どのGPU上にいるのかなど考えなくて良いので楽です．
ただし，バッチ化するためにはデータのサイズが揃っている必要があるのでクロッピング操作やリサイズ操作だけは
データローダに実装する必要があります．

---

<div align="center">

# Your Project Name

</div>

One paragraph of project description goes here.

# Getting Started

## How to run

This project provides the singularity definition file in singularity/. I recommend to build a singularity container from the definition file to reproduce my experiments. You can build a singularity container as follows:

```bash
sudo singularity build env.sif singularity/env.def
```

Then, you can execute any codes in the container. For example, you can train a model with default parameters on a GPU as follows:

```bash
singularity exec --nv python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
singularity exec --nv python run.py experiment=experiment_name
```

You can override any parameter from command line like this

```bash
singularity exec --nv python run.py trainer.max_epochs=20 datamodule.batch_size=64
```

Finally, in default, run.py automatically load the previous experiment and restart training if you previously execute run.py with the exactly same parameters. You can stop this behaviour by overwriting autoload=False (see configs/config.yaml).

# Acknowledgments

This project implementation is based on [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

<br>
