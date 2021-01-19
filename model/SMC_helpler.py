import numpy as np
from scipy.stats import norm, nbinom


def sirp(storeTab, motion_panel_t, para):
    # importing the pervious states
    S = storeTab[:, 0]
    E = storeTab[:, 1]
    I = storeTab[:, 2]
    R = storeTab[:, 3]
    D = storeTab[:, 4]
    case_report = storeTab[:, 5]
    case_test = storeTab[:, 6]
    p_case_report = storeTab[:, 7]
    p_case_test = storeTab[:, 8]

    for i in range(storeTab.shape[1]):
        storeTab[:, i] = np.clip(storeTab[:, i], a_min=0, a_max=None)
    dt = para['dt']
    N = para['N']

    # setting parameters
    beta = np.clip(motion_panel_t[:, 0] * I / N, 0, 1)  # the beta para
    xi = np.clip(motion_panel_t[:, 1], 0, 1)
    eta = np.clip(motion_panel_t[:, 2], 0, 1)  # the testing para

    # clinical import
    gamm = para['gamm']
    sigma = para['sigma']
    rho = para["day_to_report"]

    # Binomial increments with exponential decaying probability
    sfrac = 1.0 - np.exp(-beta * dt)
    eta = 1.0 - np.exp(-eta * dt)
    dfrac = 1.0 - np.exp(-xi * dt)


    S_to_E = S * sfrac
    E_to_I = E * sigma
    I_to_R = I * gamm
    I_to_D = I * dfrac
    p_test_to_rp = p_case_test * rho
    p_case_to_rp = p_case_report * rho

    # Process model for SEIRD+
    St = S - S_to_E
    Et = E + S_to_E - E_to_I
    It = I + E_to_I - I_to_R - I_to_D
    Rt = R + I_to_R
    Dt = D + I_to_D

    # Case tracking model for SEIRD+
    p_case_test_t = p_case_test - p_test_to_rp + (S_to_E ) * eta
    case_test_t = case_test + p_test_to_rp

    p_case_report_t = p_case_report - p_case_to_rp + (E_to_I - I_to_R - I_to_D) * eta
    case_report_t = case_report + p_case_to_rp

    return St, Et, It, Rt, Dt, case_report_t, case_test_t, p_case_report_t, p_case_test_t


def assignweight(data_dict, storeL, tt):
    # Get observations
    report_true_diff_tt = data_dict['infection'][tt] - data_dict['infection'][tt - 1]
    report_true_tt = data_dict['infection'][tt]
    d_true_diff_tt = data_dict['death'][tt] - data_dict['death'][tt - 1]
    test_true_tt = data_dict['test'][tt]
    test_diff_tt = data_dict['test'][tt] - data_dict['test'][tt - 1]

    # Get estimations
    death = np.clip(storeL[:, tt, 4] - storeL[:, tt - 1, 4], a_min=0, a_max=None)
    report_diff = np.clip(storeL[:, tt, 5] - storeL[:, tt - 1, 5], a_min=0, a_max=None)
    report = storeL[:, tt, 5]
    test = np.clip(storeL[:, tt, 6] , a_min=0, a_max=None)

    # Get the log likelihood from poisson distribution

    loglikSum_report_diff = nbinom.logpmf(k=report_true_diff_tt, n=1, p=1 / (1 + report_diff))
    loglik_report = norm.logpdf(report_true_tt, loc=report, scale=report_true_tt)


    if (test_diff_tt > 0) & (data_dict['death'][tt] > 0):
        loglik_test = norm.logpdf(test_true_tt, loc=test, scale= test_true_tt)
        loglik_d = nbinom.logpmf(k=d_true_diff_tt, n=1, p=1 / (1 + death))
        loglikSum = (loglikSum_report_diff + loglik_report + loglik_test + loglik_d) / 4
        loglikSum = np.clip(loglikSum, a_min=-500, a_max=None)
    elif (data_dict['death'][tt] > 0):
        loglik_d = nbinom.logpmf(k=d_true_diff_tt, n=1, p=1 / (1 + death))
        loglikSum = (loglikSum_report_diff + loglik_report + loglik_d) / 3
        loglikSum = np.clip(loglikSum, a_min=-500, a_max=None)
    elif (test_diff_tt > 0):
        loglik_test =norm.logpdf(test_true_tt, loc=test, scale= test_true_tt)
        loglikSum = (loglikSum_report_diff + loglik_report + loglik_test) / 3
        loglikSum = np.clip(loglikSum, a_min=-500, a_max=None)
    else:
        loglikSum = (loglikSum_report_diff + loglik_report) / 2
        loglikSum = np.clip(loglikSum, a_min=-500, a_max=None)



    return np.exp(loglikSum)


def sir_f(storeTab, para):
    # importing the pervious states
    S = storeTab[:, 0]
    E = storeTab[:, 1]
    I = storeTab[:, 2]
    R = storeTab[:, 3]
    D = storeTab[:, 4]
    case_report = storeTab[:, 5]
    p_case_report = storeTab[:, 6]

    for i in range(storeTab.shape[1]):
        storeTab[:, i] = np.clip(storeTab[:, i], a_min=0, a_max=None)
    dt = para['dt']
    N = para['N']

    # setting parameters
    beta = np.clip(  para['rt'] * I / N, 0, 1)  # the beta para
    xi = np.clip(    para['xi'], 0, 1)
    eta = para['eta']  # the testing para

    # clinical import
    gamm = para['gamm']
    sigma = para['sigma']
    rho = para["day_to_report"]

    # Binomial increments with exponential decaying probability
    sfrac = 1.0 - np.exp(-beta * dt)
    eta = 1.0 - np.exp(-eta * dt)
    dfrac = 1.0 - np.exp(-xi * dt)


    S_to_E = S * sfrac
    E_to_I = E * sigma
    I_to_R = I * gamm
    I_to_D = I * dfrac
    p_case_to_rp = p_case_report * rho

    # Process model for SEIRD+
    St = S - S_to_E
    Et = E + S_to_E - E_to_I
    It = I + E_to_I - I_to_R - I_to_D
    Rt = R + I_to_R
    Dt = D + I_to_D


    p_case_report_t = p_case_report - p_case_to_rp + (E_to_I - I_to_R - I_to_D) * eta
    case_report_t = case_report + p_case_to_rp

    return St, Et, It, Rt, Dt, case_report_t, p_case_report_t
