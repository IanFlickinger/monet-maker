Key Alterations
=======================================================================================
 - PatchGAN stack depth increased
 - PatchGAN filter depths changed from contracting (3, 128, ..., 32, 1) to expanding
   (3, 32, ..., 256, 1)
 - More rigorous training period

Training
======================================================================================
 - Alternating generator/discriminator updates
 - 15 epochs, 836 steps per epoch

Analysis
=======================================================================================
 - Most images are coming out identical to their input no matter the class. I 
   hypothesize that the generators learned to trick their corresponding discrimators
   with some small alteration (hence the fuzzy green vertical line running through the
   left half of photos turned monet), and then primarily received gradient updates
   relevant to the id and cycle loss due to the discriminator's reliance on the green 
   line.

Next Steps
=======================================================================================
 - Some form of regularization may help oppose this issue. For example: dropout could 
   force the networks to attend to every pixel.