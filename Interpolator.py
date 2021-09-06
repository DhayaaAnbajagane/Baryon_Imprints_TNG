import numpy as np
from scipy import interpolate
import os

avail_properties = ['a_form', 'c200c', 'E_s_DM_scaled', 'M_acc_dyn',
                    's_DM', 's_DM_vel', 'sigma_DM_3D']

avail_sims = ['TNG300', 'TNG300_DMO', 'TNG100', 'TNG100_DMO', 'TNG50', 'TNG50_DMO']

avail_params = ['mean', 'slope', 'scatter', 'skew']

parameter_to_index = {'mean': 1, 'slope': 8, 'scatter': 16, 'skew': 24}

cwd_path = os.path.dirname(__file__)

z_list = ['z0p00', 'z0p10', 'z0p20', 'z0p30', 'z0p40',
          'z0p50', 'z0p70', 'z1p00', 'z1p50', 'z2p00',
          'z3p01', 'z4p01', 'z5p00', 'z6p01', 'z7p01',
          'z8p01', 'z9p00', 'z10p00', 'z10p98', 'z11p98']

def Scaling_Relation(M200c, property, parameter, sim, z):
    '''
    Convenience function that interpolates from a precomputed table
    to provide scaling relation parameters of different halo properties
    with halo mass, M200c, at redshifts 0 <= z <= 11.98.
    The interpolation is done linearly over log(M200c), and log(a), where
    a = 1/(1 + z) is the scale factor.


    ---------
    Params
    ---------

    M200c:  (float, int) or (list, numpy array)
        The halo mass to extract parameters at. In units
        of Msun

    property: str
        The halo property whose scaling relation parameters
        should be extracted. See Interpolator.avail_properties
        for the list of available properties. The halo
        formation time, "a_form", is not available for z > 0.

    parameter: str
        The scaling relation parameter to extract. Can take
        values of 'mean' (linear quantity, not log),
        'slope', 'scatter' (in natural log), and 'skew'.

    sim: str
        The specific simulation run the scaling relation should
        be extracted from. See Interpolator.avail_sims for the list of
        available simulations

    z: int, float
        The redshift that the parameters should be extracted at.
        When needed the function will interpolate between available
        data to estimate parameters at the exact input redshift.

    --------
    Output
    --------

    numpy array:

        Array of dimension (M200c.size, 7). The first column gives the
        mean value for the parameter while the columns after that provide
        the upper and lower bounds at 1/2/3 sigma. If a requested M200c or z value
        is outside the interpolation range, the corresponding row of entries
        in the output will contain np.NaN values.

    '''

    if property not in avail_properties:

        raise ValueError("Requested property, %s, is not available. Choose one of "%property + str(avail_properties))

    elif (property == 'a_form') & (z != 0):

        raise ValueError("a_form is available only for z = 0. You requested z = %0.2f"%z)

    if sim not in avail_sims:

        raise ValueError("Requested sim, %s, is not available. Choose one of "%sim + str(avail_sims))

    if parameter not in avail_params:

        raise ValueError("Requested parameter, %s, is not available. Choosen one of "%parameter + str(avail_params))


    M200c = np.atleast_1d(M200c)[:, None]
    z     = np.ones_like(M200c) * z
    a     = 1/(1 + z)

    if property != 'a_form':

        input  = np.concatenate([M200c, a], axis = 1)
        input  = np.log10(input)

        if 'TNG300' in sim:
            z_list_temp = z_list[:-3]
        elif ('TNG100' in sim) and ('c200c' in property):
            z_list_temp = z_list[:-1]
        else:
            z_list_temp = z_list

        data = [0]*len(z_list_temp)

        #Load data from all redshifts
        for i, z_label in enumerate(z_list_temp):

            z_val   = float(z_label[1:].replace('p','.'))
            a_val   = 1/(1 + z_val)

            data[i] = np.loadtxt(cwd_path +'/Data/KLLR_Params_%s_%s_%s.txt'%(sim, property, z_label))
            data[i] = np.concatenate([data[i], np.ones([data[i].shape[0], 1])*a_val], axis = 1)

        data   = np.concatenate(data, axis = 0)
        index  = parameter_to_index[parameter]
        output = np.zeros([M200c.size, 7])

        for i in range(7):

            if (parameter is 'mean') & (property is not 'M_acc_dyn'):
                data_modified = np.log10(data[:, 1 + i])

            else:
                data_modified = data[:, index + i]

            interp       = interpolate.LinearNDInterpolator(np.log10(data[:, [0, -1]]), data_modified)
            output[:, i] = interp(input)


            if (parameter is 'mean') & (property is not 'M_acc_dyn'):
                output[:, i] = 10**output[:, i]

    elif property == 'a_form':

        input  = np.log10(M200c.flatten())

        data = np.loadtxt(cwd_path +'/Data/KLLR_Params_%s_%s_z0p00.txt'%(sim, property))
        data = np.concatenate([data, np.ones([data.shape[0], 1])], axis = 1)

        index  = parameter_to_index[parameter]
        output = np.zeros([M200c.size, 7])

        for i in range(7):

            interp       = interpolate.interp1d(np.log10(data[:, 0]), data[:, index + i], bounds_error = False)
            output[:, i] = interp(input)

    return output


def Correlation(M200c, property1, property2, sim, z):
    '''
    Convenience function that interpolates from a precomputed table
    to provide correlations between different halo properties
    as a function of halo mass, M200c, at redshifts 0 <= z <= 11.98.
    The interpolation is done linearly over log(M200c), and log(a), where
    a = 1/(1 + z) is the scale factor.


    The uncertainties only include those coming from the correlation
    coefficient alone and are thus lower bounds.

    ---------
    Params
    ---------

    M200c:  (float, int) or (list, numpy array)
        The halo mass to extract parameters at. In units
        of Msun

    property1, property2: str
        The halo properties whose correlation should be extracted.
        See Interpolator.avail_properties for the list of available properties.
        The halo formation time, "a_form", is not available for z > 0.

    sim: str
        The specific simulation run the correlation should
        be extracted from. See Interpolator.avail_sims for the list of
        available simulations

    z: int, float
        The redshift that the correlation should be extracted at.
        When needed the function will interpolate between available
        data to estimate parameters at the exact input redshift.

    --------
    Output
    --------

    numpy array:

        Array of dimension (M200c.size, 7). The first column gives the
        mean value for the correlation while the columns after that provide
        the upper and lower bounds at 1/2/3 sigma. If a requested M200c or z value
        is outside the interpolation range, the corresponding row of entries
        in the output will contain np.NaN values.

    '''

    if sim not in avail_sims:

        raise ValueError("Requested sim, %s, is not available. Choose one of "%sim + str(avail_sims))

    elif property1 not in avail_properties:

        raise ValueError("Requested property, %s, is not available. Choose one of "%property1 + str(avail_properties))

    elif property2 not in avail_properties:

        raise ValueError("Requested property, %s, is not available. Choose one of "%property2 + str(avail_properties))

    elif ('a_form' in [property1, property2]) & (z != 0):

        raise ValueError("a_form is available only for z = 0. You chose z = %0.2f"%z)

    elif (property1 is property2):

        raise ValueError("Property1 is the same as Property2. Correlation is r = 1 by construction.")


    #Check which label comes first "alphabetically".
    #Need to do this since filenames have certain format
    if avail_properties.index(property1) > avail_properties.index(property2):
        property1, property2 = property2, property1

    M200c = np.atleast_1d(M200c)[:, None]
    z     = np.ones_like(M200c) * z
    a     = 1/(1 + z)

    if 'a_form' not in [property1, property2]:

        input  = np.concatenate([M200c, a], axis = 1)
        input  = np.log10(input)

        if 'TNG300' in sim:
            z_list_temp = z_list[:-3]
        elif ('TNG100' in sim) and ('c200c' in [property1, property2]):
            z_list_temp = z_list[:-1]
        else:
            z_list_temp = z_list

        data = [0]*len(z_list_temp)

        #Load data from all redshifts
        for i, z_label in enumerate(z_list_temp):

            z_val   = float(z_label[1:].replace('p','.'))
            a_val   = 1/(1 + z_val)

            data[i] = np.loadtxt(cwd_path +'/Data/KLLR_Corr_%s_%s_%s_%s.txt'%(sim, property1, property2, z_label))
            data[i] = np.concatenate([data[i], np.ones([data[i].shape[0], 1])*a_val], axis = 1)

        data   = np.concatenate(data, axis = 0)
        output = np.zeros([M200c.size, 7])

        for i in range(7):

            interp       = interpolate.LinearNDInterpolator(np.log10(data[:, [0, -1]]), data[:, 1 + i])
            output[:, i] = interp(input)

    elif 'a_form' in [property1, property2]:

        input  = np.log10(M200c.flatten())

        data = np.loadtxt(cwd_path +'/Data/KLLR_Corr_%s_%s_%s_z0p00.txt'%(sim, property1, property2))
        data = np.concatenate([data, np.ones([data.shape[0], 1])], axis = 1)

        output = np.zeros([M200c.size, 7])

        for i in range(7):

            interp       = interpolate.interp1d(np.log10(data[:, 0]), data[:, 1 + i], bounds_error = False)
            output[:, i] = interp(input)

    return output


def Covariance(M200c, property1, property2, sim, z):

    '''
    Convenience function that interpolates from a precomputed table
    to provide covariances between different halo properties
    as a function of halo mass, M200c, at redshifts 0 <= z <= 11.98.
    The interpolation is done linearly over log(M200c), and log(a),
    where a = 1/(1 + z) is the scale factor.



    ---------
    Params
    ---------

    M200c:  (float, int) or (list, numpy array)
        The halo mass to extract parameters at. In units
        of Msun

    property1, property2: str
        The halo properties whose covariance should be extracted.
        See Interpolator.avail_properties for the list of available properties.
        The halo formation time, "a_form", is not available for z > 0.

    sim: str
        The specific simulation run the covariance should
        be extracted from. See Interpolator.avail_sims for the list of
        available simulations

    z: int, float
        The redshift that the covariance should be extracted at.
        When needed the function will interpolate between available
        data to estimate parameters at the exact input redshift.

    --------
    Output
    --------

    numpy array:

        Array of dimension (M200c.size, 7). The first column gives the
        mean value for the covariance while the three pairs of columns
        after that provide the lower and upper bounds for 1/2/3-sigma.
        If a requested M200c value is outside the interpolation range,
        the corresponding row of entries in the output will contain np.NaN values.

    '''

    #Get intrinsic scatter of each property, and their correlation
    params1 = Scaling_Relation(M200c, property1, 'scatter', sim, z)

    if property1 is not property2:

        params2 = Scaling_Relation(M200c, property2, 'scatter', sim, z)
        corr    = Correlation(M200c, property1, property2, sim, z)

        #Multiply correlation by mean scatter of each property
        #cov = corr * sigma1 * sigma2
        output = corr * params1 * params2

    else:

        output = params1**2

    return output
