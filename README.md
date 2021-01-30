# Bilinear Attentional Transforms (BAT) for Image Classification

This is the official code of [Non-Local Neural Networks With Grouped Bilinear Attentional Transforms](http://openaccess.thecvf.com/content_CVPR_2020/html/Chi_Non-Local_Neural_Networks_With_Grouped_Bilinear_Attentional_Transforms_CVPR_2020_paper.html) for image classification on ImageNet.

## Pretrained models
Here we provide some of the pretrained models.

| Method | GFLOPs |#Params |Top-1 Acc| Link |
| :--: | :--: | :--: | :--: | :--: |
| ResNet-50 + BAT | 5.4 | 30.2M | 78.3% | [GoogleDrive](https://drive.google.com/file/d/1prEX0xhrwlqLfyMTPXztf0EmqeO0sxgn/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/15PJ2L3RbRvquLzJJZKiqUQ)(Access Code: h587) |
| ResNet-101 + BAT | 9.2 | 49.2M | 79.1% | [GoogleDrive](https://drive.google.com/file/d/1OpUN7_4C2XzvXwS6-kvn5Juzp5IE7yoe/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/1oHExy1xw6srbzOohrmh1zA)(Access Code: h2bo) |
| ResNet-152 + BAT | 12.9 | 64.9M | 79.4% | [GoogleDrive](https://drive.google.com/file/d/1DgKbFjPTxkzTAuPYQI0RxLHa8htoNrUz/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/1H9LbbmCdM83vKTn9yhSixw)(Access Code: eh6f) |

**Note:** The models provided here are trained on [MXNet’s recordIO files](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/dataloading_recordio.html), so the performance will drop significantly while evaluated on JPEG images. But they can be used as good training initalization for other tasks, such as [video classification]().

## Quick starts
### Requirements

- pip install -r requirements.txt

### Data preparation
You can follow the Pytorch implementation:
https://github.com/pytorch/examples/tree/master/imagenet

### Training

To train a model, run [main.py](main.py) with the desired model architecture and other super-paremeters:

```bash
python main.py -a nlnet50 --nltype bat --nl_mod 2 2 1000 --dropout 0.2 [imagenet-folder with train and val folders]
```
**Note:** For models deeper than ResNet-34, dropout=0.2 is adopted at residual connection for BAT-Block to reduce over-fitting. 
We use "nl_mod" to control how many BAT-Blocks to be added at Res3, Res4 and Res5. It's set to "2 2 1000" for ResNet-50, "2 7 1000" for ResNet-101, "4 12 1000" for ResNet-152.

### Testing
```bash
python main.py -a nlnet50 --nltype bat --nl_mod 2 2 1000 --evaluate --resume PATH/TO/CHECKPOINT [imagenet-folder with train and val folders]
```

## Other applications of BAT
* [Video Classification]()

## Citation
If you find this work or code is helpful in your research, please cite:
````
@InProceedings{Chi_2020_CVPR,
  author = {Chi, Lu and Yuan, Zehuan and Mu, Yadong and Wang, Changhu},
  title = {Non-Local Neural Networks With Grouped Bilinear Attentional Transforms},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
````