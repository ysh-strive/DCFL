# DCFL
Diverse Co-saliency Feature Learning for Text-Based Person Retrieval
# Pipeline

![network-2025-4-7](https://github.com/user-attachments/assets/f0c1fed5-f2b3-447c-a1c9-2d495cddac6c)


## Usage
### Requirements
we use single RTX3090 24G GPU for training and evaluation. 
```
Python 3.9
Pytorch 2.0.0 & torchvision 0.15.0
```
#### Modify the ./anaconda3/envs/xxx/lib/python3.9/site-packages/torch/nn/functional.py file.
##### Before the modification is as follows:
```
  if attn_mask is not None:
      attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
  else:
      attn = torch.bmm(q, k.transpose(-2, -1))     
  attn = softmax(attn, dim=-1)
```
##### After the modification is as follows:
```
  if attn_mask is not None:
      #attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
      attn = torch.bmm(q, k.transpose(-2, -1))
      attn = softmax(attn,dim=-1)
      attn = attn*(attn_mask)
  else:
      attn = torch.bmm(q, k.transpose(-2, -1))
      attn = softmax(attn, dim=-1)
  #attn = softmax(attn, dim=-1)
```


### Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```

## Training
```python
python train.py
```
## Testing

```python
python test.py
```
#### Comparison with other methods on three datasets (CUHK-PEDES, ICFG-PEDES, and RSTPReid). Rank-1, Rank-5, and Rank-10 represent the accuracy (%), with higher values indicating better performance.
![](images/res.png)




[Model & log for CUHK-PEDES](https://drive.google.com/file/d/1OBhFhpZpltRMZ88K6ceNUv4vZgevsFCW/view?usp=share_link)

[Model & log for ICFG-PEDES](https://drive.google.com/file/d/1Y3D7zZsKPpuEHWJ9nVecUW-HaKdjDI9g/view?usp=share_link)

[Model & log for RSTPReid](https://drive.google.com/file/d/1LpUHkLErEWkJiXyWYxWwiK-8Fz1_1QGY/view?usp=share_link)



## Acknowledgments
Some components of this code implementation are adopted from [PFM-EKFP](https://github.com/lhf12278/PFM-EKFP) and [IRRA](https://github.com/BrandonHanx/TextReID). We sincerely appreciate for their contributions.
