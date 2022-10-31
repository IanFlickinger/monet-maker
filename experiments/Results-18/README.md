Key Alterations
=======================================================================================
 - Spatial Dropout at rate of 0.5 at beginning of down stack, base stack, up stack, and
   immediately prior to final convolution

Training
=======================================================================================
 - 15 epochs of 300 steps
 - Training time 1:27:43.8

Analysis
=======================================================================================
 - Although the only alteration between v17 and v18 was the dropout, the generators 
   lost all color information. As predicted, spatial dropout applied immediately prior 
   to the final convolution could be encouraging the model to learn grayscale 
   interpolation at the last stage to minimize the l1 id and cycle losses

Next Steps
=======================================================================================
 - 