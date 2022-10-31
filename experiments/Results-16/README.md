Key Alterations
=======================================================================================
 - Replaced Dropout in generator with SpatialDropout at rate of 0.5
 - Increased depth of down and up stacks in generator to 5
 - Doubled number of filters at each level in generator, capped at 512
 - Decreased depth of residual base stack in generator from 9 to 4
 - Doubled number of filters at each level in discriminator

Training
=======================================================================================
 - 15 epochs of 300 steps
 - Time elapse 1:35:31

Analysis
=======================================================================================
 - 

Next Steps
=======================================================================================
 - Normalization?
 - Augmentation Agent?
 - Initialization?
 - Dropout?
    - Specify not training when predicting?
 - SpatialDropout?
 - Multiple Optimizers?
 - Different Loss Functions?
 - Activation Functions?
 - More channels!?