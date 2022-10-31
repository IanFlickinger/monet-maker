Key Alterations
=======================================================================================
 - The discriminator loss weight in the overall computation of generator loss was 
   increased by a factor of four to encourage focus on "fooling" the discriminator

Training
======================================================================================
 - Alternating generator/discriminator updates
 - 15 epochs, 836 steps per epoch

Analysis
=======================================================================================
 - Increasing the weight of the discriminator loss did very little to change the 
   development of consistent artifacts out of the discriminator that simply seem to be
   overlayed on the input image. This further supports my previous hypothesis that the
   generator's optimization on account of the discriminator loss is primarily focused
   on a small portion of the pixels, the strength of which may be eradicated by dropout
   regularization.

Next Steps
=======================================================================================
