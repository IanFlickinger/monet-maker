Key Alterations
=======================================================================================
 - Learning rate reduced to 2e-4
 - Training updates are alternated between generators and discriminators
 - PatchGAN discriminator with receptive field of 71
 - Generator has skip connection which reintroduces the coloring of the original input

Training
======================================================================================
 - Alternating generator/discriminator updates
 - 10 epochs, 300 steps per epoch

Analysis
=======================================================================================
 - Discoloration is still occurring. though the discoloration appears to be uniform
   throughout each class. Perhaps this is a learned aspect? 
 - Each monet has a bright spot added into it. The odd part is that it seems almost as
   though the generator has learned to add a sun (sometimes very convincingly) despite
   the lack of suns in most monets.

Next Steps
=======================================================================================
Since I have very few ideas for going forward at this point, I think I'll just run the
same model for a longer training period and compare the results. I'd like to find a way
to reduce the discoloration effect... I'm also not a fan of the immense amount of 
parameters dedicated solely to normalization.

I've prolonged the implementation of dropout and differentiable augmentation for long
enough. Perhaps if I find no more immediately pressing architecture changes, I'll work
on those goals next.