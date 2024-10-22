1、运行step1_extract_raw_data.ipynb
这个代码和开源代码https://www.kaggle.com/code/lucasrr/rsna-generate-1024x1024-data完全一致，其它数据请参考改文件。

2、运行step2_crop_data.ipynb
这个代码里面需要对step1处理后的图片进行剪切

3、运行step3_tfrecord.ipynb
这个代码是把step2中的数据转为tfrecord

4、运行step4_train.ipynb
这个代码是对step3中的tfrecord进行训练

5、运行step5_infer.ipynb进行提交

