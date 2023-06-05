import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.models import load_model
import math
import numpy as np
import pickle

def main():
    df_testing= pd.read_csv('..\data\database.csv')
    X_test = df_testing.loc[:, ['wavelength', 'fractal_dimension', 'fraction_of_coating', 'primary_particle_size',
                   'number_of_primary_particles',
                   'vol_equi_radius_outer', 'vol_equi_radius_inner', 'equi_mobility_dia']]
    # Normalizaing Min max
    #scaling_x = MinMaxScaler()
    #scaling_y = MinMaxScaler()
    #X_train = scaling_x.fit_transform(X_train)
    #X_test = scaling_x.fit_transform(X_test)
    #Y_train = scaling_y.fit_transform(Y_train)
    scaling_x = pickle.load(open('..\data\scaler_x.sav', 'rb'))
    scaling_y = pickle.load(open('..\data\scaler_y.sav', 'rb'))
    X_test= scaling_x.transform(X_test)

    model = load_model('../data/best_model_forward.hdf5')
    Y_test = model.predict(X_test)
    Y_test = scaling_y.inverse_transform(Y_test)
    #print(Y_test)

    # Computing others
    Y_test = pd.DataFrame(data=Y_test, columns=["q_abs", "q_sca", "g"])

    wavelength = df_testing['wavelength']
    fractal_dimension = df_testing['fractal_dimension']
    fraction_of_coating = df_testing['fraction_of_coating']
    primary_particle_size = df_testing['primary_particle_size']
    number_of_primary_particles = df_testing['number_of_primary_particles']
    vol_equi_radius_inner = df_testing['vol_equi_radius_inner']
    vol_equi_radius_outer = df_testing['vol_equi_radius_outer']
    equi_mobility_dia = df_testing['equi_mobility_dia']

    mie_epsilon = np.zeros_like(wavelength) + 2
    length_scale_factor = 2 * math.pi / wavelength

    m_real_bc=np.empty_like(wavelength)
    for i in range(0,len(wavelength)):
        if wavelength[i]==467:
            m_real_bc[i]=1.92
        elif wavelength[i]==530:
            m_real_bc[i]=1.96
        elif wavelength[i]==660:
            m_real_bc[i]=2
        else:
            m_real_bc[i]=np.nan

    m_im_bc = np.empty_like(wavelength)
    for i in range(0, len(wavelength)):
        if wavelength[i] == 467:
            m_im_bc[i] = 0.67
        elif wavelength[i] == 530:
            m_im_bc[i] = 0.65
        elif wavelength[i] == 660:
            m_im_bc[i] = 0.63
        else:
            m_im_bc[i] = np.nan

    m_real_organics = np.empty_like(wavelength)
    for i in range(0, len(wavelength)):
        if wavelength[i] == 467:
            m_real_organics[i] = 1.59
        elif wavelength[i] == 530:
            m_real_organics[i] = 1.47
        elif wavelength[i] == 660:
            m_real_organics[i] = 1.47
        else:
            m_real_organics[i] = np.nan

    m_im_organics = np.empty_like(wavelength)
    for i in range(0, len(wavelength)):
        if wavelength[i] == 467:
            m_im_organics[i] = 0.11
        elif wavelength[i] == 530:
            m_im_organics[i] = 0.04
        elif wavelength[i] == 660:
            m_im_organics[i] = 0
        else:
            m_im_organics[i] = np.nan

    volume_total = (4 * math.pi * (vol_equi_radius_outer ** 3)) / 3
    volume_bc = (4 * math.pi * (vol_equi_radius_inner ** 3)) / 3
    volume_organics = volume_total - volume_bc

    density_bc = np.zeros_like(wavelength) + 1.5 #Check
    density_organics = np.zeros_like(wavelength) + 1.1 #Check

    mass_bc = volume_bc * density_bc * (1 / 1000000000000000000000)
    mass_organics = volume_organics * density_organics * (1 / 1000000000000000000000)
    mass_total = mass_bc + mass_organics
    mr_total_bc = mass_total / mass_bc
    mr_nonbc_bc = mass_organics / mass_bc

    q_abs = Y_test['q_abs']
    q_sca = Y_test['q_sca']
    q_ext = q_abs + q_sca
    g = Y_test['g']
    c_geo = (math.pi) * ((vol_equi_radius_outer) ** 2)
    c_ext = (q_ext * c_geo) / (float(1000000))
    c_abs = q_abs * c_geo / (1000000)
    c_sca = q_sca * c_geo / (1000000)
    ssa = q_sca / q_ext
    mac_total = (c_abs) / (mass_total * 1000000000000)
    mac_bc = c_abs / (mass_bc * (1000000000000))
    mac_organics = c_abs / (mass_organics * (1000000000000))

    final = np.stack((wavelength, fractal_dimension, fraction_of_coating, primary_particle_size,
                      number_of_primary_particles, vol_equi_radius_inner, vol_equi_radius_outer, equi_mobility_dia,
                      mie_epsilon, length_scale_factor, m_real_bc, m_im_bc, m_real_organics, m_im_organics,
                      volume_total, volume_bc, volume_organics, density_bc, density_organics, mass_total, mass_organics,
                      mass_bc, mr_total_bc, mr_nonbc_bc, q_ext, q_abs, q_sca, g, c_geo, c_ext, c_abs, c_sca, ssa,
                      mac_total, mac_bc, mac_organics), axis=1)

    final_dataset = pd.DataFrame(data=final, columns=['wavelength', 'fractal_dimension', 'fraction_of_coating', 'primary_particle_size',
                      'number_of_primary_particles', 'vol_equi_radius_inner', 'vol_equi_radius_outer', 'equi_mobility_dia',
                      'mie_epsilon', 'length_scale_factor', 'm_real_bc', 'm_im_bc', 'm_real_organics', 'm_im_organics',
                      'volume_total', 'volume_bc', 'volume_organics', 'density_bc', 'density_organics', 'mass_total', 'mass_organics',
                      'mass_bc', 'mr_total_bc', 'mr_nonbc_bc', 'q_ext', 'q_abs', 'q_sca', 'g', 'c_geo', 'c_ext', 'c_abs', 'c_sca', 'ssa',
                      'mac_total', 'mac_bc', 'mac_organics'])
    final_dataset.to_csv('..\data\predicted_forward_dataset.csv', index=False)

if __name__ == "__main__":
    main()