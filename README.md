# LineRWKV

Code for ["Onboard deep lossless and near-lossless predictive coding of hyperspectral images with line-based attention"](https://arxiv.org/abs/2403.17677) paper.

LineRWKV is a method for lossless and lossy compression of hyperspectral images. The compression algorithm is based on predictive coding where a neural network performs prediction of each pixel based on a causal spatial and spectral context, followed by entropy coding of the prediction residual. The neural network predictor processes the image line-by-line using a novel hybrid attentive-recursive operation that combines the representational advantages of Transformers with the linear complexity and recursive implementation of recurrent neural networks. This allows significant savings in memory and computational complexity while reaching state-of-the-art rate-distortion performance.

BibTex reference:
```
@article{valsesia2024linerwkv,
  title={Onboard deep lossless and near-lossless predictive coding of hyperspectral images with line-based attention},
  author={Valsesia, Diego and Bianchi, Tiziano and Magli, Enrico},
  journal={arXiv preprint arXiv:2403.17677},
  year={2024}
}
```

## Code
Code coming soon.
