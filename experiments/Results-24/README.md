Key Alterations
=======================================================================================
 - Replaced Concat(upstack, skip) with Add(upstack, skip) in the generator upstack
   architecture.

Training
=========================================================================================
 - **Epochs** 16
 - **Steps** 300
 - **Learning Rate** 2e-4
 - **Generator Dropout** 0.5
 - **Discriminator Dropout** 0.5
 - **Loss Weights** 10, 1, 1
 - **Alternated** No

Outcome
=========================================================================================
 - **Duration** 1hr 36min

![Monet Generator Evolution](./monet-cycle-gan-evolution.png)
![Photo Generator Evolution](./photo-cycle-gan-evolution.png)
![Example Monets](./final-test.png)

Analysis
=======================================================================================
 - 

Next Steps
=======================================================================================
 - 