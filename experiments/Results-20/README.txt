Key Alterations
=======================================================================================
 - Spatial Dropout at rate of 0.5 at beginning of each level in down stack, beginning
   of each residual block in base stack, and beginning of each upsample in up stack

Training
=======================================================================================
 - 20 epochs of 300 steps
 - Execution Time 2:11:11.3s

Analysis
=======================================================================================
 - The results here are similar to the results of experiment 17, which was identical to
   this experiment, except with no dropout. This leads me to question the effectiveness
   of adding spatial dropout, especially given the likely increased cost of training
   with dropout regularization.

Next Steps
=======================================================================================
 - After noticing that LayerNormalization was still being used in the discriminator,
   I've decided the next experiment will be updating that to Instance Normalization and
   running a training session identical to experiment 17