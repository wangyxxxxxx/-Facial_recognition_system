本项目是 ArcFace CPU 本地推理版本。

目录说明：
backbones/              ArcFace 网络结构代码
weights/model.pt         训练好的 ArcFace 权重
galleries/fei_gallery.pt 普通注册人脸 gallery
galleries/fei_gallery_wm_theta090.pt 水印 gallery
watermark_keys/watermark_key_theta090.pt 水印检测 key
scripts/predict_one.py   单图普通识别
scripts/detect_embedding_watermark.py 单图水印检测

本地运行时请使用 --device cpu。
