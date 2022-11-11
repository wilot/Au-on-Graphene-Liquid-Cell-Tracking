# Au on Graphene Liquid Cell Tracking

Similar to the Pt on MoS2 liquid cell tracking project, where I was trying to make U-Net based atom-finders for adatom and lattice tracking of HSSDF STEM videos.

This data contains atomic resolution HAADF & BF STEM videos of gold atoms on a graphene sheet within a liquid cell.

The hope is to train U-Net based CNNs for lattice atom and adatom tracking.

## Contents

- data/ - The data given to me by Nick. Data collected from EPSIC in their format.

- synthesisers/ - Training data synthesis programs and scripts

- trainers/ - scripts to train CNNs from the training data

- trackers/ - Contains trained models and scripts to apply them to the sourced data

- results/ - Results from the trackers in the form of CNN label videos as well as atomic coordinates. 