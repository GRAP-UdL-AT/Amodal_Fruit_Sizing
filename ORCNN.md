# ORCNN
In this method, we use the visible and the amodal masks from ORCNN to estimate the diameter. From the contour of the amodal masks, an enclosing circle is fitted. Then, a pixel-to-mm conversion factor is calculated from the filtered XYZ-coordinates of the visible mask (using histogram filtering). The real-world diameter is estimated by multiplying the diameter of the enclosing circle (in pixels) with the pixel-to-mm conversion factor. <br/> <br/>
To use this method, please follow this procedure: <br/>
1. Annotate the dataset, see [ANNOTATE.md](ANNOTATE.md)
2. Train ORCNN, see [ORCNN_Train_and_Evaluate_AmodalVisibleMasks.ipynb](ORCNN_Train_and_Evaluate_AmodalVisibleMasks.ipynb)
3. Estimate the diameter from the visible and the amodal masks, see [Diameter_estimation_AmodalVisibleMasks.ipynb](Diameter_estimation_AmodalVisibleMasks.ipynb)
