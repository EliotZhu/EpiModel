import os
import pickle
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from model.SIR_SMC import SDE_run


"""
    Data preparation
"""

df_country = pd.read_csv('data/df_country.csv')
country_set = df_country.groupby('country')[['test', 'death', 'active', 'S6', 'S7']].sum().reset_index()
country_set = country_set.loc[:,'country']

########################################################################################################################
"""
    Stage 1, fit SEIRD+ to observation
"""

country_i = 'Germany'
df_country_i = df_country[df_country.country == country_i]
init_date = np.array(df_country_i.loc[df_country_i.active >= 1, 'date'])[7]
df_country_i = df_country_i[(df_country_i.date >= init_date) & (df_country_i.date <= '2021-01-10')]
if not  (os.path.exists(os.path.join('results/coutry_profile', country_i) + '.pickle')):
    if len(df_country_i) >= 50:
        idx = np.where(df_country_i.infected > 0)[0]
        for i in range(max(idx)):
            if df_country_i.infected.iloc[i] == 0:
                if i > 0:
                    dist = min(idx[idx > i]) - i
                    step = (df_country_i.infected.iloc[min(idx[idx > i])] - df_country_i.infected.iloc[
                        i - 1]) / dist
                    df_country_i.infected.iloc[i] = df_country_i.infected.iloc[i - 1] + step
                else:
                    df_country_i.infected.iloc[i] = 0

        for i in range(len(df_country_i.infected)):
            i = len(df_country_i.infected) - i - 1
            if df_country_i.infected.iloc[i] == 0:
                idx = np.where(df_country_i.infected > 0)[0]
                if len(idx[idx < i]) > 0:
                    df_country_i.infected.iloc[i] = df_country_i.infected.iloc[max(idx[idx < i])]

        for i in range(len(df_country_i.S6)):
            i = len(df_country_i.S6) - i - 1
            if df_country_i.S6.iloc[i] == 0:
                idx = np.where(df_country_i.S6 > 0)[0]
                if len(idx[idx < i]) > 0:
                    df_country_i.S6.iloc[i] = df_country_i.S6.iloc[max(idx[idx < i])]
                    df_country_i.StringencyIndex.iloc[i] = df_country_i.StringencyIndex.iloc[
                        max(idx[idx < i])]
                    df_country_i.S7.iloc[i] = df_country_i.S7.iloc[max(idx[idx < i])]

    out_put = SDE_run(df_country_i)
    with open(os.path.join('results/coutry_profile', country_i) + '.pickle', 'wb') as f:
        pickle.dump(out_put, f)