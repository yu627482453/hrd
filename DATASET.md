# DATASET
We adopt the same protocol with BGNN for Visual Genome and Openimage datasets.

## Visual Genome
The following is adapted from BGNN by following the same protocal of [Unbiased Scene Graph Generation from Biased Training](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) 
You can download the annotation directly by following steps.

### Download:
1. Download the VG images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `/path/to/vg/VG_100K`. 

2. Download the [scene graphs annotations](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/bipher_shanghaitech_edu_cn/EfI9vkdunDpCqp8ooxoHhloBE6KDuztZDWQM_Sbsw_1x5A?e=N8gWIS) and extract them to `/path/to/vg/vg_motif_anno`.

3. Link the image into the project folder
```
ln -s /path-to-vg datasets/vg
```
