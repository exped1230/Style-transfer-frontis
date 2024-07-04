## 第一步：下载所需模型
配置TokenFlow所需模型：
```python
cd TokenFlow
huggingface-cli download --resume-download runwayml/stable-diffusion-v1-5 --local-dir SD-v1-5
```
注意，用以上命令在huggingface镜像下载模型过程中，如果下载过慢，可以终止后，重新输入以上命令，会继续下载

配置Track-Anything所需模型：
下载[SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)、[E2FGVI](https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view)、[XMem](https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth)模型后，放置在```Track-Anything/checkpoints/```目录下

## 第二步：局部跟踪与填充
```python
cd Track-Anything
python app.py
```
接下来，即可在基于gradio的本地网页上进行交互```localhost:12212```。

注意：一定要跟踪(Tracking)后再进行填充(Inpainting)，否则会因为缺少视频中每一帧的掩码而导致没有填充目标。

## 第三步：风格迁移
```python
cd ../TokenFlow
python preprocess.py
python run_tokenflow_pnp.py
```
如果要修改风格迁移的提示文本(prompt)，可以修改```TokenFlow/configs/config_custom.yaml```文件中第16行的prompt，如果是局部调整，例如银色的建筑，可以尝试将第16行的pnp_attn_t值调整为0.5。

## 第四步：完善局部替换
```python
cd ../Track-Anything
python replace.py
```

新生成的new_video.mp4是最终的迁移视频。
