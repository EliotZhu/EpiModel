import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.utils.extmath import cartesian
from model.SMC_helpler import sirp, assignweight,sir_f


def simulatep(data_dict, para, threshold= 0.55):
    t_length = len(data_dict['infection'])
    resample_size = min(para['nn'], 6000)

    # Set up geometric Brownian motion
    motion_beta = np.array(np.random.normal(0., para['betavol'], para['nn'] * t_length)).reshape(t_length, -1)
    motion_beta[0, :] = np.exp(motion_beta[0, :]) * para["beta"]

    motion_panel = [cartesian((motion_beta[t, :], np.array([0.00001]), np.array([1]) )) for t in (range(t_length))]
    motion_panel = np.array(motion_panel)

    re_sample = np.random.choice(a=motion_panel.shape[1], size=resample_size, replace=False).astype('int32')
    motion_panel = motion_panel[:, re_sample, :]
    # motion_panel = motion_beta.reshape((-1, resample_size,1))

    tp = np.arange(0, t_length, 1)
    S_traj = np.zeros(t_length)
    E_traj = np.zeros(t_length)
    I_traj = np.zeros(t_length)
    R_traj = np.zeros(t_length)
    D_traj = np.zeros(t_length)
    Report_traj = np.zeros(t_length)
    Test_traj = np.zeros(t_length)

    beta_traj = np.zeros(t_length)
    para2_traj = np.zeros(t_length)
    test_prop_traj = np.zeros(t_length)

    lik_values = np.zeros(t_length)
    latent_sample = np.zeros(t_length).astype('int32')

    w = np.zeros((resample_size, t_length))
    w[:, 0] = 1  # weights
    W = np.zeros((resample_size, t_length))
    A = np.zeros((resample_size, t_length))  # particle parent matrix

    storeL = np.zeros((resample_size, t_length, 9))  # store the squence by particles, t_length and counts to match
    storeL[:, 0, 0] = data_dict['population'][0] - data_dict['infection'][0]  # suspected
    storeL[:, 0, 1] = data_dict['test'][0]
    storeL[:, 0, 2] = data_dict['infection'][0]
    storeL[:, 0, 3] = data_dict['recovery'][0]
    storeL[:, 0, 4] = data_dict['death'][0]
    # 1: exposed initialisation; 5: case_pending;  6:case_report; 7: case test
    storeLm = np.mean(storeL, axis=0)

    '''
    Particle Filter
    '''
    data_dict['infection']
    for tt in (range(1, t_length)):

        if tt == max(1, min(np.where(data_dict['death'] > 0)[0])):
            motion_temp = np.array(np.random.normal(0., 2, resample_size*(t_length-tt+1))).reshape(t_length-tt+1, -1)
            motion_panel[tt-1:, :, 1] = np.exp(motion_temp) * para["xi"]
        if tt ==1:
            motion_test = np.array(np.random.normal(0., para['testvol'], resample_size*(t_length-tt+1))).reshape(t_length-tt+1, -1)
            motion_panel[:, :, 2] = np.exp(motion_test)
        motion_panel[tt, :, :] = motion_panel[tt - 1, :, :] * np.exp(motion_panel[tt, :, :])

        traj = sirp(storeTab=storeL[:, tt - 1, :], motion_panel_t=motion_panel[tt, :, :], para=para)
        storeL[:, tt, :] = np.array(traj).T

        # normalise particle weights
        w[:, tt] = assignweight(data_dict, storeL, tt)
        sum_weights = sum(w[:, tt])
        W[:, tt] = w[:, tt] / sum_weights

        # resample particles by sampling parent particles according to weights:
        A[:, tt] = np.random.choice(a=range(0, resample_size), size=resample_size, p=W[:, tt], replace=True)
        storeL[:, tt, :] = storeL[A[:, tt].astype('int32'), tt, :]
        motion_panel[tt, :, :] = motion_panel[tt, A[:, tt].astype('int32'), :]

        storeLm[tt,:] = np.mean(storeL[:, tt, :], axis=0)


    check1 = abs(storeLm[tt, 5] - data_dict['infection'][tt])/data_dict['infection'][tt]
    for tt in range(1, t_length):
        lik_values[tt] = np.log(sum(w[:, tt]))


    if check1 <=threshold:

        likelihood_overall = -t_length * np.log(resample_size) + sum(lik_values)  # full averaged log-likelihoods

        # Sample latent variables
        loc = np.random.choice(a=range(0, resample_size), size=resample_size, p=W[:, tt], replace=True)
        t_end = t_length - 1
        latent_sample[t_end] = loc[0]
        S_traj[t_end] = storeL[latent_sample[t_end], t_end, 0]
        E_traj[t_end] = storeL[latent_sample[t_end], t_end, 1]
        I_traj[t_end] = storeL[latent_sample[t_end], t_end, 2]
        R_traj[t_end] = storeL[latent_sample[t_end], t_end, 3]
        D_traj[t_end] = storeL[latent_sample[t_end], t_end, 4]
        Report_traj[t_end] = storeL[latent_sample[t_end], t_end, 5]
        Test_traj[t_end] = storeL[latent_sample[t_end], t_end, 6]

        beta_traj[t_end] = motion_panel[t_end, latent_sample[t_end], 0]
        para2_traj[t_end] = motion_panel[t_end, latent_sample[t_end], 1]
        test_prop_traj[t_end] = motion_panel[t_end, latent_sample[t_end], 2]

        for ii in np.arange(t_end, 0, -1):
            latent_sample[ii - 1] = A[latent_sample[ii], ii]  # find the corresponding particle sequence
            S_traj[ii - 1] = storeL[latent_sample[ii - 1], ii - 1, 0]
            E_traj[ii - 1] = storeL[latent_sample[ii - 1], ii - 1, 1]
            I_traj[ii - 1] = storeL[latent_sample[ii - 1], ii - 1, 2]
            R_traj[ii - 1] = storeL[latent_sample[ii - 1], ii - 1, 3]
            D_traj[ii - 1] = storeL[latent_sample[ii - 1], ii - 1, 4]
            Report_traj[ii - 1] = storeL[latent_sample[ii - 1], ii - 1, 5]
            Test_traj[ii - 1] = storeL[latent_sample[ii - 1], ii - 1, 6]

            beta_traj[ii - 1] = motion_panel[ii - 1, latent_sample[ii - 1], 0]
            para2_traj[ii - 1] = motion_panel[ii - 1, latent_sample[ii - 1], 1]
            test_prop_traj[ii - 1] = motion_panel[ii - 1, latent_sample[ii - 1], 2]
    else:
        likelihood_overall = -10000

    return {'t': tp, 'S': S_traj, 'E': E_traj, 'I': I_traj, 'T': Test_traj, 'D': D_traj, 'Report': Report_traj,
            "beta": beta_traj, "xi": para2_traj,"test_prop": test_prop_traj, "liklihood": likelihood_overall, "lik": lik_values}


def dat_loader(df_country_i, t_start=0):
    t_length = len(df_country_i)
    data_dict = {}
    data_dict['infection'] = np.array(df_country_i.infected, dtype=float)[t_start:t_start + t_length]
    data_dict['recovery'] = np.array(df_country_i.recovered, dtype=float)[t_start:t_start + t_length]
    data_dict['death'] = np.array(df_country_i.death, dtype=float)[t_start:t_start + t_length]
    data_dict['population'] = np.array(df_country_i.population, dtype=float)[t_start:t_start + t_length]
    data_dict['passenger'] = np.array(df_country_i.passenger, dtype=float)[t_start:t_start + t_length]
    data_dict['test'] =  np.array(df_country_i.test, dtype=float)[t_start:t_start + t_length]
    data_dict['S7'] = np.array(df_country_i.S7, dtype=float)[t_start:t_start + t_length]

    para = {}
    #parameters to be estimated
    para['dt'] = 1
    para['beta'] = 0.7
    para['xi'] = np.mean(((data_dict['death'][14:]- data_dict['death'][13:-1])/ data_dict['infection'][0:-14]) *
                         (data_dict['infection'][0:-14]/sum(data_dict['infection'][0:-14])))
    para['gamm'] = 1 / 2.9  # mean infectious period: Transmission dynamics of 2019 novel coronavirus (2019-nCoV)
    para['sigma'] = 1 / 5.2  # mean incubation period: early Transmission Dynamics in Wuhan, China, of Novel Coronavirus–Infected Pneumonia
    #para['day_to_report'] = 1/6.1 # mean days to from onset to confirmation

    para['dt'] = 1
    para['resample'] = 3000
    para['N'] = data_dict['population'][0]
    if data_dict['passenger'][0]>0:
        para['Travel'] = data_dict['passenger'][0] / data_dict['population'][0] / 365
    else:
        para['Travel'] =  0.00001
    if len(np.where(data_dict['S7'] >= 2)[0])> 0:
        para['Ban'] = np.where(data_dict['S7'] >= 2)[0][0]
    else :
        para['Ban'] = 1000

    return para, data_dict

def grid_search(para,data_dict, nn=1000):
    print('grid searching...')
    days_to_report_choice = 1/np.arange(1,25,2)
    betavol_choice = np.arange(0.3,1.5,0.2)
    test_init_choice = np.arange(0.5,1,0.1)
    def loop(days_to_report):
        store_lik = []
        for ii in range(len(test_init_choice)):
            for jj in range(len(betavol_choice)):
                    para['nn'] = int(nn)  # paricles for beta
                    para["day_to_report"] = days_to_report
                    para['betavol'] =  betavol_choice[jj]
                    para['testvol'] = test_init_choice[ii]
                    # Run SMC to check likelihood
                    sir_out = simulatep(data_dict, para, threshold=1)
                    store_lik.append([para['day_to_report'],para['betavol'], para["testvol"], sir_out['liklihood']])
        return store_lik

    store_lik = Parallel(n_jobs=-1)(delayed(loop)(i) for i in (days_to_report_choice))
    store_lik = np.concatenate(store_lik)
    return store_lik

def SDE_run(df_country_i):
    para, data_dict = dat_loader(df_country_i)
    store_lik = grid_search(para,data_dict)
    grid_optimal = np.array(store_lik)[np.where(np.array(store_lik)[:,3] == max(np.array(store_lik)[:,3]))[0][0],:]

    para['nn'] = int(8000)
    para['day_to_report'],para['betavol'],para["testvol"] =  grid_optimal[0:3]
    #para['day_to_report'],para['betavol'],para["testvol"] = [0.1, 0.8, 0.4]
    para['betavol'] = 2
    sir_out = simulatep(data_dict, para, 0.8)
    plt.plot(sir_out['t'], sir_out['Report'], '--')
    plt.plot(sir_out['t'], data_dict['infection'])

    plt.plot(sir_out['t'], sir_out['D'], '--')
    plt.plot(sir_out['t'], data_dict['death'])
    plt.show()


    sir_out = Parallel(n_jobs=-1)(delayed(simulatep)(data_dict, para) for i in (range(200)))
    sir_out_a = np.array(sir_out)

    beta = np.array([out['beta']  for out in sir_out_a if out['liklihood']!=-10000] )
    xi = np.array([out['xi'] for out in sir_out_a if out['liklihood']!=-10000])
    I = np.array([out['Report'] for out in sir_out_a if out['liklihood']!=-10000])
    T = np.array([out['T'] for out in sir_out_a if out['liklihood']!=-10000])
    D = np.array([out['D'] for out in sir_out_a if out['liklihood']!=-10000])
    like = np.array([out['liklihood'] for out in sir_out_a if out['liklihood']!=-10000])
    TP = np.array([out['test_prop'] for out in sir_out_a if out['liklihood']!=-10000])


    like = np.quantile(like, [0.025,0.25,0.5,0.75,0.975])
    like = np.tile([like],D.shape[1]).reshape(-1,5)
    R0_quantile = np.array([np.quantile(beta[:,t]/para['gamm'], [0.025,0.25,0.5,0.75,0.975]) for t in range(beta.shape[1])])
    xi_quantile = np.array([np.quantile(xi[:,t], [0.025,0.25,0.5,0.75,0.975]) for t in range(xi.shape[1])])
    I_quantile = np.array([np.quantile(I[:,t], [0.025,0.25,0.5,0.75,0.975]) for t in range(I.shape[1])])
    T_quantile = np.array([np.quantile(T[:,t], [0.025,0.25,0.5,0.75,0.975]) for t in range(T.shape[1])])
    D_quantile = np.array([np.quantile(D[:,t], [0.025,0.25,0.5,0.75,0.975]) for t in range(D.shape[1])])
    TP_quantile = np.array([np.quantile(TP[:,t], [0.025,0.25,0.5,0.75,0.975]) for t in range(TP.shape[1])])


    Bias_I = (np.abs(np.mean(I, 0)-data_dict['infection'])/data_dict['infection']).reshape(-1,1)
    Bias_T = (np.abs(np.mean(T, 0)-data_dict['test']+1)/(data_dict['test']+1)).reshape(-1,1)
    Bias_D = (np.abs(np.mean(D, 0)-data_dict['death']+1)/(data_dict['death']+1)).reshape(-1,1)
    Bias_T50 = (np.abs(T_quantile[:,2]-data_dict['test']+1)/(data_dict['test']+1)).reshape(-1,1)
    Bias_I50 = (np.abs(I_quantile[:,2]-data_dict['infection'])/data_dict['infection']).reshape(-1,1)
    Bias_D50 = (np.abs(D_quantile[:,2]-data_dict['death']+1)/(data_dict['death']+1)).reshape(-1,1)

    popt = np.concatenate([R0_quantile,xi_quantile,I_quantile, D_quantile,like, Bias_I,Bias_T,Bias_D, Bias_I50,Bias_T50, Bias_D50, TP_quantile], axis=1)

    return data_dict, para, popt

def dat_loader2(df_country_i,para_map_i, t_start=-1, offset = -1):
    t_length = len(df_country_i)
    data_dict = {}
    data_dict['infection'] = np.array(df_country_i.infected, dtype=float)[t_start + t_length]
    data_dict['recovery'] = np.array(df_country_i.recovered, dtype=float)[t_start + t_length]
    data_dict['death'] = np.array(df_country_i.death, dtype=float)[t_start + t_length]
    data_dict['population'] = np.array(df_country_i.population, dtype=float)[t_start + t_length]
    data_dict['passenger'] = np.array(df_country_i.passenger, dtype=float)[t_start + t_length]
    data_dict['test'] =  np.array(df_country_i.test, dtype=float)[t_start + t_length]
    data_dict['S7'] = np.array(df_country_i.S7, dtype=float)[t_start + t_length]

    para = {}
    #parameters to be estimated
    para['dt'] = 1
    para['beta'] = 0.7
    para['xi'] = para_map_i.xi50.iloc[offset]
    para['rt'] = para_map_i.R050.iloc[offset]

    para['rt25'] = para_map_i.R025.iloc[offset]
    para['rt75'] = para_map_i.R075.iloc[offset]
    para['xi25'] = para_map_i.xi25.iloc[offset]
    para['xi75'] = para_map_i.xi75.iloc[offset]


    para['day_to_report'] = para_map_i.day_to_report.iloc[offset]
    para['gamm'] = 1 / 2.9  # mean infectious period: Transmission dynamics of 2019 novel coronavirus (2019-nCoV)
    para['sigma'] = 1 / 5.2  # mean incubation period: early Transmission Dynamics in Wuhan, China, of Novel Coronavirus–Infected Pneumonia
    #para['day_to_report'] = 1/6.1 # mean days to from onset to confirmation
    para['dt'] = 1
    para['resample'] = 3000
    para['N'] = data_dict['population']
    para['SI'] = para_map_i.I50.iloc[-1]
    para['SD'] = para_map_i.D50.iloc[-1]

    if data_dict['passenger']>0:
        para['Travel'] = data_dict['passenger'] / data_dict['population'] / 365
    else:
        para['Travel'] =  0.00001
    if len(np.where(data_dict['S7'] >= 2)[0])> 0:
        para['Ban'] = np.where(data_dict['S7'] >= 2)[0][0]
    else :
        para['Ban'] = 1000

    return para, data_dict


def SDE_predict(df_country_i,future_df,prediction_date,prediction_date2, pred_interval, offset = -1):
    para_map_i = future_df[future_df.countryname == df_country_i.country.iloc[0]]
    para, data_dict = dat_loader2(df_country_i[df_country_i.date <= prediction_date],para_map_i,offset)

    choice = [para['SI']/data_dict['population'],para['SI']/data_dict['population']*2,
                     para['SI']/data_dict['population']*2.5,para['SI']/data_dict['population']*3,
                     para['SI']/data_dict['population']*3.5,para['SI']/data_dict['population']*4,
                     para['SI']/data_dict['population']*4.5,para['SI']/data_dict['population']*5,
                     para['SI']/data_dict['population']*7,para['SI']/data_dict['population']*9,
                     para['SI']/data_dict['population']*10, para['SI']/data_dict['population']*0.5]
    # 1: exposed in itialisation; 5: case_pending;  6:case_report; 7: case test
    bias = []
    for jj in range(len(choice)):
        para['eta'] = np.array(choice)[jj]
        storeL = np.zeros((1, pred_interval, 7))  # store the squence by particles, t_length and counts to match
        storeL[:, 0, 0] = data_dict['population'] - para['SI']  # suspected
        storeL[:, 0, 1] = 0
        storeL[:, 0, 2] = para['SI']
        storeL[:, 0, 3] = data_dict['recovery']
        storeL[:, 0, 4] = para['SD']
        storeL[:, 0, 5] = data_dict['infection']
        storeL[:, 0, 6] = 0
        for tt in (range(1, pred_interval)):
            traj = sir_f(storeTab=storeL[:, tt - 1, :], para=para)
            storeL[:, tt, :] = np.array(traj).T
        I = storeL[:, :, 5]
        true_i = np.array(df_country_i[(df_country_i.date> prediction_date)&(df_country_i.date< prediction_date2)]['infected'])
        bias.append(np.array(abs(I - true_i)/(true_i))[-1][-1])

    para['eta'] = np.array(choice)[np.where(bias == np.min(bias))][0]
    para25 = para.copy()
    para25['rt'] =  para25['rt25']
    para25['xi'] =  para25['xi25']
    para75 = para.copy()
    para75['rt'] = para75['rt75']
    para75['xi'] = para75['xi75']

    storeL = np.zeros((1, pred_interval, 7))
    storeL[:, 0, 0] = data_dict['population'] - para['SI']  # suspected
    storeL[:, 0, 1] = 0
    storeL[:, 0, 2] = para['SI']
    storeL[:, 0, 3] = data_dict['recovery']
    storeL[:, 0, 4] = para['SD']
    storeL[:, 0, 5] = data_dict['infection']
    storeL[:, 0, 6] = 0

    storeL25 = np.array(storeL, copy=True)

    storeL75 = np.array(storeL, copy=True)

    for tt in (range(1, pred_interval)):
        traj = sir_f(storeTab=storeL[:, tt - 1, :], para=para)
        traj25 = sir_f(storeTab=storeL25[:, tt - 1, :], para=para25)
        traj75 = sir_f(storeTab=storeL75[:, tt - 1, :], para=para75)
        storeL[:, tt, :] = np.array(traj).T
        storeL25[:, tt, :] = np.array(traj25).T
        storeL75[:, tt, :] = np.array(traj75).T

    I = storeL[:, :, 5][0].T
    I25 = storeL25[:, :, 5][0].T
    I75 = storeL75[:, :, 5][0].T

    return I,I25,I75, para['eta']



