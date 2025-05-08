BMSegNet
===
The data is publicly available on quark: Link: [https://pan.baidu.com/s/1i7QjmVVNadfPHQDwxVxcxg](https://pan.quark.cn/s/4ef1d3bd4609) Extract code: U9nv<br>
In the data, 0-149 corresponds to BMHS, 150-299 corresponds to BMTO, and 300-450 corresponds to BMS.
The model is built based on mmsegmentation. The python version used is 3.8. RepViT, ResNet, Swin, PCPVT, MobileNetV2 are built based on mmseg==0.30.0, mmcv-full==1.7.1;<br>
EfficientFormer and EfficientFormerV2 are built on mmseg==0.19.0, mmcv-full==1.3.17;<br>
The rest of the models were built based on mmseg==1.2.2, mmcv-full==1.7.2, where EfficientNet requires mmpretrain to be installed.<br>

Train
-------
python ./tools/train.py --work-dir ./work/M2-3 ./configs\sem_fpn\repvit_fpn_dsaspp_m2_3.py

Test
-------
python ./tools/test.py ./configs/sem_fpn/repvit_fpn_daspp_m2_3.py ./work/M2-3/.pth --eval mIoU mFscore --show-dir ./Prediction/RepViT/Image
