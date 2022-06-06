Repo for "Neural Architecture Search for Compressed Sensing Magnetic Resonance Image Reconstruction".  

The original project borrowed codes from https://github.com/facebookresearch/fastMRI (old version) and https://github.com/quark0/darts . 

The current version was based on pytorch=0.4.1=py36_cuda8.0.61_cudnn7.1.2_1 with torchvision=0.2.1=py36_0 without maintenance to adapt new pytorch version.

To run:
python3 ./NAS-for-CSMRI/models/nas/search_mri.py --name 'setting_1' for searching the architecture of cells.

This project is for research purpose and not approved for clinical use.

Our previous article is published in Computerized Medical Imaging And Graphics, or refer to researchgate revised version(https://www.researchgate.net/publication/339471747_Neural_Architecture_Search_for_Compressed_Sensing_Magnetic_Resonance_Image_Reconstruction).  

Please cite:  

@article{yan2020neural,
  title={Neural Architecture Search for compressed sensing Magnetic Resonance image reconstruction},
  author={Yan, Jiangpeng and Chen, Shou and Zhang, Yongbing and Li, Xiu},
  journal={Computerized Medical Imaging and Graphics},
  volume={85},
  pages={101784},
  year={2020},
  publisher={Elsevier}
}