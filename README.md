# <p align=center> `SEE`  <a href='https://arxiv.org/pdf/2501.18783'><img src='https://img.shields.io/badge/TPAMI-2025-red'></a> </p>

**Segment concealed object with incomplete supervision, TPAMI, 2025**

[Chunming He](https://chunminghe.github.io/), [Kai Li](https://scholar.google.com/citations?user=YsROc4UAAAAJ&hl=en), [Yachao Zhang](https://scholar.google.com/citations?user=a-I8c8EAAAAJ&hl=en), [Ziyun Yang](https://scholar.google.com/citations?user=G-AAVZEAAAAJ&hl=en), [Youwei Pang](https://scholar.google.com/citations?user=jdo9_goAAAAJ&hl=en), [Longxiang Tang](https://scholar.google.com/citations?user=3oMQsq8AAAAJ&hl=en), [Chengyu Fang](https://cnyvfang.github.io/), [Yulun Zhang](https://yulunzhang.com), [Linghe Kong](https://scholar.google.com/citations?hl=en&user=-wm2X-8AAAAJ), [Xiu Li](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en) and [Sina Farsiu](https://scholar.google.com/citations?user=mzcr92sAAAAJ&hl=en) 

>**Abstract:** Existing concealed object segmentation (COS) methods frequently utilize reversible strategies to address uncertain regions. However, these approaches are typically restricted to the mask domain, leaving the potential of the RGB domain underexplored. To address this, we propose the Reversible Unfolding Network (RUN), which applies reversible strategies across both mask and RGB domains through a theoretically grounded framework, enabling accurate segmentation. RUN first formulates a novel COS model by incorporating an extra residual sparsity constraint to minimize segmentation uncertainties. The iterative optimization steps of the proposed model are then unfolded into a multistage network, with each step corresponding to a stage. Each stage of RUN consists of two reversible modules: the Segmentation-Oriented Foreground Separation (SOFS) module and the Reconstruction-Oriented Background Extraction (ROBE) module. SOFS applies the reversible strategy at the mask level and introduces Reversible State Space to capture non-local information. ROBE extends this to the RGB domain, employing a reconstruction network to address conflicting foreground and background regions identified as distortion-prone areas, which arise from their separate estimation by independent modules. As the stages progress, RUN gradually facilitates reversible modeling of foreground and background in both the mask and RGB domains, directing the network's attention to uncertain regions and mitigating false-positive and false-negative results. Extensive experiments demonstrate the superior performance of RUN and highlight the potential of unfolding-based frameworks for COS and other high-level vision tasks.   

![](feature.png)













