Key Alterations
=========================================================================================
 - Added differentiable augmentations to the CycleGAN training procedure
 - Reintroduced spatial dropout to down stack, base stack, and up stack of unet

Training
=========================================================================================
 - **Epochs** 16
 - **Steps** 300
 - **Learning Rate** 2e-3
 - **Generator Dropout** 0.5, 0.5, 0.5
 - **Discriminator Dropout** 0.5
 - **Loss Weights** 1, 1, 1
 - **Alternated** No
 - **Augmentations** Brightness, Saturation, Color, Contrast
 - **Duration** 1hr 10min

Analysis
=========================================================================================
 - The generators have lost all sense of color again. It seems most likely to me that
   this loss of color information is due to the spatial dropout. Sure enough, upon
   review of the generator model, there is spatial dropout after the concatentation of
   the input image with the reconstructed image. This would encourage the generator to
   learn to interpolate a greyscale image, as this experiment shows.
 - There are still some small spots in the images. Close inspection shows that these
   spots appear to be identical in size and shape, and nearly identical in color - 
   though the color does seem to be correlated with the surrounding image pixels. It is
   unclear what causes these spots to appear where they do. If it were not for the surely
   determinant nature of the network, I might claim it was random.
 - The brightest spots in the images seem to have been maximized in the monet generator,
   and the areas around these spots seem to have contracted the same magnitude. This
   brightness creep seems to even ignore sharp edges into dark areas - likely an artifact
   of the convolutional architecture of the generator.

Next Steps
=========================================================================================
 - Translation and cutout augmentations are also encouraged by the original DiffAugment
   paper. I hope that these augmentations will prevent the discriminator from 
   memorizing the content of Monets, as that is not the point of style transfer. It is
   for this reason that I am most interested in cutout as an augmentation.
 - Decrease the capacity of the discriminator. Since the discriminator need only learn
   basic features in order to determine style, it seems fitting to reduce capacity. 
 - Discriminator validation. It would be nice to have some form of measurement of 
   whether or not the discriminator is overfitting.
 - Remove spatial dropout from upstack. At the very least, the final spatial dropout
   should be removed.