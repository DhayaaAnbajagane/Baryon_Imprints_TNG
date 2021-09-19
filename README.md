# Baryon Imprints on Dark Matter Scaling Relations from TNG

This repository hosts both the data and corresponding interpolation scripts for the scaling relation parameters of halos in the TNG50/100/300 runs (both dark matter only and full physics variants) from redshifts 0 < z < 11.98. The properties and their corresponding scaling parameters are computed as described in [Anbajagane, Evrard & Farahi 2021](https://arxiv.org/abs/2109.02713). **Citation to this publication is requested if these scaling parameters are used.**

The following properties are available:

1. sigma_DM_3D: The 3D DM velocity dispersion
2. c200c: The NFW concentration
3. a_form: The formation time of the halo
4. s_DM: The density-space shape, defined as the major-to-minor axis.
5. s_DM_vel: Same as above, but for the velocity-space
6. M_acc_dyn: The mass accretion rate, dlnM/dlna, over one dynamical time
7. E_s_DM_scaled: The dimensionless DM surface pressure energy.


## QUICKSTART

```

import sys
sys.path.append("<path to 'Baryon_Imprints_TNG'>/Baryon_Imprints_TNG")

import Interpolator
import numpy as np

print("List of available properties is " + str(Interpolator.avail_properties))
print("List of available sims is " + str(Interpolator.avail_sims))
print("List of available scaling parameters is " + str(Interpolator.avail_params))

M200c = 10**np.linspace(9, 14.5, 100)

mean = Interpolator.Scaling_Relation(M200c, property = 'c200c', 
                                     parameter = 'mean', sim = 'TNG300', z = 0.45)
                                     
corr = Interpolator.Correlation(M200c, property1 = 'c200c', property2 = 'sigma_DM_3D',  
                                sim = 'TNG100', z = 0.45)
                                
cov  = Interpolator.Covariance(M200c,  property1 = 'c200c', property2 = 'sigma_DM_3D',  
                               sim = 'TNG50',  z = 0.45)

```

### Caveats

1. a_form is only available for the z = 0 catalog.
2. The params for TNG300 are available only up to z < 9 due to limited sample size above the mass threshold at higher redshifts.
3. The c200c-related params for TNG100 are only available up to z < 10.98. For c200c we have a slightly higher mass threshold. Other params are available through 0 < z < 11.98


If you find any errors/bugs in the code, please reach out to dhayaa@uchicago.edu
