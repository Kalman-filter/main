import pandas as pd
import numpy as np
import os.path
from scipy.stats import norm
from scipy.stats._continuous_distns import _norm_cdf
from scipy.stats._continuous_distns import _norm_pdf
from scipy.stats import logistic
from scipy.stats import hypsecant
from scipy.optimize import minimize
from scipy.special import ive   ## exponentially modified first order Bessel function


########################################################################
def entropy(p):

    y = -p *np.log(p) - (1-p)*np.log(p)
    return y

########################################################################
def ghL_func(zz, y, PAR, only_L = False ):
    # zz: the difference in the skills
    # y: output: ordinal variable (integer)
    # PAR: definition of the model
    # only_L=True if we need only the output L (this is done to spare calculation)
    #
    # Output:
    # g: derivative of the log-likelihood ( ATTENTION: must be negated if we want to obtain derivative of negated log-likelihood)
    # h: negated second derivative of the log-likelihood (must be positive because the log-likelihood is concave)
    # L: likelihood (probabilities for all ordinal outputs)

    def vw_func(z):
        v = np.zeros_like(z)
        ii = z > -20
        # v[ii] = norm.pdf(z[ii]) / norm.cdf(z[ii])
        v[ii] = _norm_pdf(z[ii]) / _norm_cdf(z[ii])
        ii = z <= -20
        v[ii] = abs(z[ii]) + 1 / abs(z[ii])  # limiting case (avoid division by zero)
        w = v * (z + v)
        return v, w

    model = PAR["rating_model"]

    N = zz.size
    a = np.log(10)
    g = np.zeros_like(zz)
    h = np.zeros_like(zz)

    if model == "Thurston":
        # L = norm.cdf(zz * sign_y)
        if N == 1:
            L = np.zeros(2)
        else:
            L = np.zeros((2, N))

        z = _norm_cdf(-zz)
        L[0,] = z
        L[1,] = 1-z
        sign_y = 2 * y - 1
        (v, w) = vw_func(zz * sign_y)
        g = v * sign_y
        h = w
    elif model == "Bradley-Terry":
        if N == 1:
            L = np.zeros(2)
        else:
            L = np.zeros((2, N))

        pow_zn = 10 ** (-zz)
        FL = 1 / (1 + pow_zn)
        L[0,] = 1 - FL
        L[1,] = FL
        g = a * (y - FL)
        h = (a ** 2) * FL * (1 - FL)
    elif model == "Gauss":
        var = PAR["std.dev"]**2
        g = (y - zz) / var
        h = np.ones_like(zz).astype(float) / var
        L = 0
    elif model == "Davidson":
        kappa = PAR["draw"]
        pow_zn = 10 ** (-zz)
        pow_zp = 10 ** (zz)
        den = pow_zp + kappa + pow_zn
        if N == 1:
            L = np.zeros(3)
        else:
            L = np.zeros((3, N))

        L[0,] = pow_zn / den
        L[1,] = kappa / den
        L[2,] = pow_zp / den

        g = 2 * a *( y/2 - (pow_zp+0.5 * kappa)/den )
        h = (a ** 2) * (kappa*(pow_zp+pow_zn) + 4)/(den ** 2)
    elif model == "MOV":
        alpha = PAR["alpha"]
        delta = PAR["delta"]
        num = 10 ** (alpha + 2*delta * zz)
        den = num.sum(axis=0)

        if N == 1:
            L = (num / den).ravel()
        else:
            L = (num / den)
        F = delta.transpose() @ num / den
        g = 2 * a * (delta[y].transpose() - F).ravel()
        h = 4 * (a**2) * ( (delta.transpose()**2) @ num / den - F**2).ravel()
    elif model == "Skellam":
        exp_c = np.exp(PAR["offset"])
        mean_h = np.exp(zz)
        L = np.exp(-exp_c * (mean_h + 1/mean_h) + y * zz + 2*exp_c + np.log(ive(np.abs(y), 2*exp_c)))
        g = y - exp_c*(mean_h - 1/mean_h)
        h = exp_c*(mean_h + 1/mean_h)
    else:
        raise Exception("wrong model")

    return (g, h, L)


########################################################################
def p_hat_GH(z, omega, PAR):
    if PAR["rating_model"] == "Thurston":
        #  use analytical integration formula
        # sign_y = 2 * y - 1
        # p = norm.cdf(sign_y * m / np.sqrt(1 + omega))
        p_hat = [_norm_cdf(-z / np.sqrt(1 + omega)), _norm_cdf(z / np.sqrt(1 + omega))]
    else:
        y_dummy = 0
        if omega == 0:
            (g, h, L) = ghL_func(z, y_dummy, PAR, only_L=True)
            p_hat = L
        else:   # Outcome probability calculated using Gauss-Hermite quadrature
            # raise Exception("this has to be checked")
            K = 10
            (x, w) = np.polynomial.hermite.hermgauss(K)  # Gauss-Hermite quadrature points
            x = x.reshape((1, K))
            (g, h, L) = ghL_func(x * np.sqrt(2 * omega) + z, y_dummy, PAR, only_L=True)
            p_hat = (L @ w.reshape((K, 1)) ) / np.sqrt(np.pi)

    return p_hat


########################################################################
def forecast_metric(p_hat, y, PAR, p1_true=0):
    ## $p_hat$ is a vector of probabilities associated with different outputs; we observe the output $y$

    if len(p_hat.shape) ==  1:
        p_hat = p_hat[:, None]   # becomes the column vector
    (M, N) = p_hat.shape
    if M < 2:
        raise Exception("distribution should have at least two elements")
    # if y.size != N:
        # raise Exception("p_hat and y should have the same number of elements")

    p_th = 1.e-10
    # first determine how to map the probabilities to the outcomes (important for multi-level outcomes)
    if PAR["rating_model"] in {"Thurston", "Bradley-Terry"}:
        # jj = [0,1]   # the outcome is the index
        y_ADH = y
        pp_ADH = p_hat
        score_set = np.array([0, 1])
        # score = y
    elif PAR["rating_model"] == "Davidson":
        # jj = [0, 1, 2]  # the outcome is the index
        y_ADH = y
        pp_ADH = p_hat
        score_set = np.array([0, 0.5, 1])
        # score = score_set[y]
    elif PAR["rating_model"] == "MOV":
        J = len(PAR["delta"])
        j_draw = int((J - 1) / 2)
        y_ADH = (y == j_draw)*1 + (y > j_draw)*2  # transform the result y into a ternary result y_ADH
        ADH_matrix = np.zeros((3,M))
        ADH_matrix[0, np.arange(0, j_draw)] = 1
        ADH_matrix[1, j_draw] = 1
        ADH_matrix[2, np.arange(j_draw + 1, J)] = 1
        pp_ADH = ADH_matrix @ p_hat
        score_set = PAR["delta"].ravel()
        # score = score_set[y]
    elif PAR["rating_model"] == "Skellam":
        J = len(PAR["delta"])
        j_draw = int((J - 1) / 2)
        y_ADH = (y == 0)*1 + (y > 0)*2  # transform the result y into a ternary result y_ADH
        ADH_matrix = np.zeros((3, M))
        ADH_matrix[0, np.arange(0, j_draw)] = 1
        ADH_matrix[1, j_draw] = 1
        ADH_matrix[2, np.arange(j_draw + 1, J)] = 1
        pp_ADH = ADH_matrix @ p_hat
        score_set = PAR["delta"].ravel()
    # remove very small value (less than p_th)
    pp_ADH *= (pp_ADH >= p_th)
    pp_ADH += p_th * (pp_ADH < p_th)

    #####
    if PAR["metric_type"] == "LS_exact":
        raise Exception("This works only for binary outcomes")
        p_hat *= (p_hat >= p_th)
        p_hat += p_th * (p_hat < p_th)
        LS = - (1 - p1_true) * np.log(p0_hat) - p1_true * np.log(p1_hat)
    if PAR["metric_type"] == "DKL":
        raise Exception("This works only for binary outcomes")
        pp = p_hat_GH(zz, 0, 1, PAR)  # use point estimate of skills: omega <- 0
        p_hat *= (p_hat >= p_th)
        p_hat += p_th * (p_hat < p_th)
        if p1_true < p_th or (p1_true > 1-p_th):
            LS = 0
        else:
            LS = (1 - p1_true) * np.log(1 - p1_true) + p1_true * np.log(p1_true)

        LS += - (1 - p1_true) * np.log(p0_hat) - p1_true * np.log(p1_hat)
    elif PAR["metric_type"] == "LS":
        LS = -np.log(pp_ADH[y_ADH, np.arange(N)])  # log-score
    elif PAR["metric_type"] == "LS_0":
        LS = -np.log(pp_ADH[y_ADH, np.arange(N)])  # log-score
    elif PAR["metric_type"] == "RPS_0":
        is_A = (y_ADH == 0)*1
        is_AorD = (y_ADH <= 1)*1

        RPS = 0.5*( (pp_ADH[0, :]-is_A)**2 + (pp_ADH[0:2, :].sum(axis=0)-is_AorD)**2 )
        LS = RPS
    elif PAR["metric_type"] == "ACC":
        i_max = np.argmax(pp_ADH, axis=0)

        AC = (i_max == y_ADH)*1
        LS = -AC
    elif PAR["metric_type"] == "empirical_avg":
        # score averaged using Gauss-Hermite quadrature
        K = 10
        (x, w) = np.polynomial.hermite.hermgauss(K)  # Gauss-Hermite quadrature points
        (g, h, L) = ghL_func(x * np.sqrt(2 * omega) + zz, y, PAR, only_L=True)
        ll = -np.log(L)
        LS = np.dot(ll, w) / np.sqrt(np.pi)
    elif PAR["metric_type"] == "MSE":
        Ex_score = np.dot(score_set, p_hat)
        LS = (Ex_score - y_ADH/2)**2

    else:
        raise Exception("wrong metric calculation type: " + PAR["metric_type"] + " unspecified" )

    return LS

##############
def forecast_from_bookies(results_in, bookie, PAR_in):

    results = results_in.copy()
    PAR = PAR_in.copy()

    if bookie == "Bet365":
        pH = 1/results["B365H"].values
        pD = 1/results["B365D"].values
        pA = 1/results["B365A"].values
        tot = pH+pA+pD
        pH /=tot
        pA /=tot
        pD /=tot

        N= pA.shape[0]
        p_hat = np.concatenate((pA.reshape((1,N)),pD.reshape((1,N)),pH.reshape((1,N))), axis=0)

        PAR["rating_model"] = "Davidson"
        results = filter_results(results, PAR)  # adds the column of the "game_results"
        LS = forecast_metric(p_hat, results["game_result"].values, PAR)

    return LS


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

####################################################################################################
def Kalman(results_in, PAR, rating_0=[], variance_0=[]):

    # PAR: dictionary
    # results_in: Pandas frame with columns {"team_home","team_away","game_result","time_stamp"}
    # we expect the input variable rating_0, when not empty, to be indexable with the teams names (dictionary or pandas series)

    results = results_in.copy()
    if "game_result" not in results:
        results = filter_results(results, PAR)  # adds the column "game_result"

    home = PAR["home"]  # HFA
    beta = PAR["beta"]
    epsilon = PAR["epsilon"]
    v0 = PAR["v0"]
    scale = PAR["scale"]

    KFtype = PAR["rating_algorithm"]
    # it_nr = PAR["it"]
    if "PAR_gen" not in PAR.keys():
        PAR_gen = {"scenario": "real-world"}  # no information about data generation (for real-world data)
    else:
        PAR_gen = PAR["PAR_gen"]

    is_data = False
    if "data" in PAR.keys():
        is_data = True
        theta_org = PAR["data"]

    N = len(results)  # number of games

    unique_players = set(results["team_home"])
    unique_players.update(set(results["team_away"]))
    unique_players = list(unique_players)
    M = len(unique_players)  # number of players

    player2index = {p: i for i, p in enumerate(unique_players)}


    if PAR_gen["scenario"] == "switch":
        t_switch = PAR_gen["d_switch"] * int(M/2)

    F = 1  # number of players in each group (home or away)
    skills = np.zeros((N, M))  # all skills through time

    # initialization of the skills
    initialization_skills_exists = (len(rating_0)>0)
    theta = np.zeros((1, M))  # holder of skills
    if initialization_skills_exists:   # use the initilization provided in the rating_0
        init_players = set(rating_0.index)
        for player in player2index:
            if player in init_players:
                player_ind = player2index[player]
                theta[0, player_ind] = rating_0[player]

    #   initialization of the covarianve matrix
    initialization_variance_exists = (len(variance_0) > 0)
    if KFtype == "KF":
        if initialization_variance_exists:
            raise Exception("code is here but not yet tested for KF")
            V_t = variance_0
            for player in player2index:
                if player not in init_players:
                    player_ind = player2index[player]
                    V_t[player_ind, :] = 0
                    V_t[:, player_ind] = 0
                    V_t[player_ind, player_ind] = v0
        else:
            V_t = np.identity(M) * v0  # covariance matrix
    elif KFtype == "vSKF":
        if initialization_variance_exists:
            V_t = variance_0
            for player in player2index:
                if player not in init_players:
                    player_ind = player2index[player]
                    V_t[player_ind] = v0
        else:
            V_t = np.ones(M) * v0  # variance vector
    elif KFtype == "sSKF":
        if initialization_variance_exists:
            raise Exception("initialization not implemented for sSKF")
        V_t = v0  # scalar variance
    elif KFtype == "fSKF" or KFtype == "SG":
        None
    else:
        raise Exception('Kalman type not defined')

    V = list([])  # covariance matrices/vectors/scalars
    LS = np.zeros(N)
    MSE = np.zeros(N)

    debug_on = False
    ####   main loop
    for n in range(N):
        i_t = player2index[results["team_home"].iloc[n]]
        j_t = player2index[results["team_away"].iloc[n]]
        y_t = results["game_result"].iloc[n]

        if PAR_gen["scenario"] != "real-world":
            prob1_real = results["real_proba"].iloc[n]

        if n > 0:
            delta_t = results["time_stamp"].iloc[n] - results["time_stamp"].iloc[n - 1]
        else:
            delta_t = 0
        beta_t = beta ** delta_t  # time-dependent version of beta
        epsilon_t = epsilon * delta_t

        # this is only used in the sythetic data
        if PAR_gen["scenario"] == "switch" and n == t_switch:  ## special treatement for the switch-time
            n_switch = PAR_gen["n_switch"]      #   number of players switched
            theta[:n_switch] = 0
            if KFtype == "KF":
                V_tmp = V_t[n_switch:, n_switch:]
                V_t   = np.identity(M) * v0  - epsilon_t * np.identity(M)  # matrix (subtract epsilon_t: it will be added later)
                V_t[n_switch:, n_switch:] = V_tmp
            elif KFtype == "vSKF":
                V_t[:n_switch] = v0 - epsilon_t   # (subtract epsilon_t: it will be added later)
            elif KFtype == "sSKF":
                V_t = (v0-V_t)* n_switch/M + V_t - epsilon_t

        #   find the posterior mode
        z_t = theta[0,i_t] - theta[0,j_t]
        z0 = beta_t*z_t

        ########################################################################
        ## this is where we depend on the model
        ##
        (g_t, h_t, L_t) = ghL_func(z0 / scale + home, y_t, PAR)
        ##
        ########################################################################
        ## this is where we depend on the algorithm
        if KFtype == "SG":
            h_t = 0
        ##
        ########################################################################

        # prepare update
        if KFtype == "KF":
            V_bar = (beta_t ** 2) * V_t + epsilon_t * np.identity(M)  # matrix
            omega_t = V_bar[i_t, i_t] + V_bar[j_t, j_t] - 2 * V_bar[i_t, j_t]
            vv = V_bar[:, i_t] - V_bar[:, j_t]
            V_t = V_bar - np.outer(vv, vv * (h_t / (scale ** 2 + h_t * omega_t)))
        elif KFtype == "vSKF":
            V_bar = (beta_t ** 2) * V_t + epsilon_t  # vector
            omega_t = V_bar[i_t] + V_bar[j_t]
            vv = np.zeros(M)
            vv[i_t] = V_bar[i_t]
            vv[j_t] = -V_bar[j_t]
            V_t = V_bar * (1 - np.abs(vv) * (h_t / (scale ** 2 + h_t * omega_t)))
        elif KFtype == "sSKF":
            V_bar = (beta_t ** 2) * V_t + epsilon_t  # scalar
            omega_t = 2 * F * V_bar
            vv = np.zeros(M)
            vv[i_t] = V_bar
            vv[j_t] = -V_bar
            V_t = V_bar * (1 - omega_t / M * (h_t / (scale ** 2 + h_t * omega_t)))
        elif KFtype == "fSKF":
            omega_t = 2 * F * v0
            vv = np.zeros(M)
            vv[i_t] = v0
            vv[j_t] = -v0
        elif KFtype == "SG":
            omega_t = 0
            vv = np.zeros(M)
            vv[i_t] = v0
            vv[j_t] = -v0

        # update skills
        theta = beta_t * theta + vv * (scale * g_t) / (scale ** 2 + h_t * omega_t)

        ## the prediction and the performance metric
        ##   prediction and performance metrics should be calculated AFTER omega_t is obtained (important for SKF methods)
        if debug_on:
            print("n=" + str(n))

        p_hat_t = p_hat_GH(z0 / scale + home, omega_t / (scale ** 2), PAR)
        LS[n] = forecast_metric(p_hat_t, y_t, PAR)

        skills[n, :] = theta
        if KFtype in {"KF", "vSKF", "sSKF"}:
            V.append(V_t.copy())


    skills_frame = pd.DataFrame(skills, columns=unique_players)
    return (skills_frame, LS, V, MSE)


####################################################################################################
####################################################################################################
def TrueSkill(results_in, PAR):
    # results: Pandas frame with columns {"team_home","team_away","game_result","time_stamp"}
    # PAR: dictionary

    #def r_func(v):
    #    return 1/np.sqrt(1+v*aa/(scale ** 2) )
    results = results_in.copy()
    home = PAR["home"]  # HFA
    beta = PAR["beta"]
    if beta != 1:
        raise Exception("shoudn't beta equal one?")
    epsilon = PAR["epsilon"]
    v0 = PAR["v0"]
    scale = PAR["scale"]
    KFtype = PAR["rating_algorithm"]
    if KFtype != "TrueSkill":
        raise Exception("rating algorithm should say: TrueSkill")
    if PAR["rating_model"] != "Thurston":
        raise Exception("TrueSkill only works for Thurston model (just check: not that it is really used somewhere)")
    PAR_gen = PAR["PAR_gen"]
    is_data = False
    if "data" in PAR.keys():
        is_data = True
        theta_org = PAR["data"]

    aa = 3 * (np.log(10)/np.pi) ** 2 # to be used in r_func()


    N = len(results)  # number of games

    unique_players = set(results["team_home"].unique())
    unique_players.update(results["team_away"].unique())
    unique_players = list(unique_players)

    M = len(unique_players)  # number of players
    player2index = {unique_players[i]: i for i in range(M)}  # dictionary assigning player identifies to indices

    if PAR_gen["scenario"] == "switch":
        t_switch = PAR_gen["d_switch"] * int(M/2)

    skills = np.zeros((N, M))  # all skills through time
    theta = np.zeros(M)  # holder of skills
    V = list([])  # covariance matrices/vectors/scalars
    LS = np.zeros(N)
    MSE = np.zeros(N)

    # elif KFtype == "vSKF":
    V_t = np.ones(M) * v0  # variance vector

    ####   main loop
    for n in range(N):
        i_t = player2index[results["team_home"][n]]
        j_t = player2index[results["team_away"][n]]
        #   z_t = theta[i_t] - theta[j_t]
        #   z_t += home
        y_t = results["game_result"][n]
        prob1_real = results["real_proba"][n]

        delta_t = 0
        if n > 0:
            delta_t = results["time_stamp"][n] - results["time_stamp"][n - 1]

        beta_t = beta ** delta_t  # time-dependent version of beta
        epsilon_t = epsilon * delta_t

        if PAR_gen["scenario"] == "switch" and n == t_switch:  ## special treatement for the switch-time
            n_switch = PAR_gen["n_switch"]      #   number of players switched
            theta[:n_switch] = 0
            V_t[:n_switch] = v0

        #   find the posterior mode
        z_old_t = theta[i_t] - theta[j_t]

        # prepare update
        V_bar = (beta_t ** 2) * V_t + epsilon_t  # vector
        omega_t = V_bar[i_t] + V_bar[j_t]
        vv = np.zeros(M)
        vv[i_t] = V_bar[i_t]
        vv[j_t] = -V_bar[j_t]

        scale_tilde = scale * np.sqrt(1+omega_t/(scale ** 2))

        (g_t, h_t, L_t) = ghL_func(beta_t * z_old_t / scale_tilde, y_t, PAR)

        # update skills
        theta = beta_t * theta + vv * g_t / scale_tilde
        #   update variance
        V_t = V_bar * (1 - np.abs(vv) * (h_t / (scale ** 2 + omega_t)))

        #   prediction
        LS_t = forecast_metric(beta* z_old_t / scale, y_t, omega_t / (scale ** 2), PAR, prob1_real)
        LS[n] = LS_t

        skills[n, :] = theta

        V.append(V_t.copy())

    skills_frame = pd.DataFrame(skills, columns=unique_players)

    return (skills_frame, LS, V, MSE)
####################################################################################################
####################################################################################################
def Glicko(results_in, PAR):
    # results: Pandas frame with columns {"team_home","team_away","game_result","time_stamp"}
    # PAR: dictionary
    results = results_in.copy()
    def r_func(v):
        return np.sqrt(1+v*aa/(scale ** 2) )

    home = PAR["home"]  # HFA
    beta = PAR["beta"]
    if beta != 1:
        raise Exception("shoudn't beta equal one?")
    epsilon = PAR["epsilon"]
    v0 = PAR["v0"]
    scale = PAR["scale"]
    KFtype = PAR["rating_algorithm"]
    if KFtype != "Glicko":
        raise Exception("rating algorithm Should say: Glicko")
    if PAR["rating_model"] != "Bradley-Terry":
        raise Exception("Glicko only works for Bradley-Terry model (not that it realy matters here)")
    PAR_gen = PAR["PAR_gen"]
    is_data = False
    if "data" in PAR.keys():
        is_data = True
        theta_org = PAR["data"]

    aa = 3 * (np.log(10)/np.pi) ** 2 # to be used in r_func()


    N = len(results)  # number of games

    unique_players = set(results["team_home"].unique())
    unique_players.update(results["team_away"].unique())
    unique_players = list(unique_players)

    M = len(unique_players)  # number of players
    player2index = {unique_players[i]: i for i in range(M)}  # dictionary assigning player identifies to indices

    if PAR_gen["scenario"] == "switch":
        t_switch = PAR_gen["d_switch"] * int(M/2)

    skills = np.zeros((N, M))  # all skills through time
    theta = np.zeros(M)  # holder of skills
    V = list([])  # covariance matrices/vectors/scalars
    LS = np.zeros(N)
    MSE = np.zeros(N)

    # elif KFtype == "vSKF":
    V_t = np.ones(M) * v0  # variance vector

    ####   main loop
    for n in range(N):
        i_t = player2index[results["team_home"][n]]
        j_t = player2index[results["team_away"][n]]
        #   z_t = theta[i_t] - theta[j_t]
        #   z_t += home
        y_t = results["game_result"][n]
        prob1_real = results["real_proba"][n]

        if n > 0:
            delta_t = results["time_stamp"][n] - results["time_stamp"][n - 1]
        else:
            delta_t = 0
        beta_t = beta ** delta_t  # time-dependent version of beta
        epsilon_t = epsilon * delta_t

        if PAR_gen["scenario"] == "switch" and n == t_switch:  ## special treatement for the switch-time
            n_switch = PAR_gen["n_switch"]      #   number of players switched
            theta[:n_switch] = 0
            V_t[:n_switch] = v0

        #   find the posterior mode
        z_old_t = theta[i_t] - theta[j_t]

        # prepare update
        V_bar = (beta_t ** 2) * V_t + epsilon_t  # vector

        omega_t = V_bar[i_t] + V_bar[j_t]

        scale_i = scale * r_func( V_bar[j_t] )
        (gg_i, hh_i, L_t) = ghL_func(beta_t * z_old_t / scale_i, y_t, PAR)

        scale_j = scale * r_func(V_bar[i_t])
        (gg_j, hh_j, L_t) = ghL_func(beta_t * z_old_t / scale_j, y_t, PAR)

        # update skills
        theta = beta_t * theta
        theta[i_t] += V_bar[i_t] * (scale_i * gg_i) / (scale_i ** 2 + hh_i * V_bar[i_t])
        theta[j_t] -= V_bar[j_t] * (scale_j * gg_j) / (scale_j ** 2 + hh_j * V_bar[j_t])

        #   update variance
        V_t = V_bar
        V_t[i_t] *= (scale_i ** 2 / (scale_i ** 2 + hh_i * V_bar[i_t]))
        V_t[j_t] *= (scale_j ** 2 / (scale_j ** 2 + hh_j * V_bar[j_t]))

        #   prediction
        LS_t = forecast_metric(beta* z_old_t / scale, y_t, omega_t / (scale ** 2), PAR, prob1_real)
        LS[n] = LS_t

        skills[n, :] = theta

        V.append(V_t.copy())

    skills_frame = pd.DataFrame(skills, columns=unique_players)

    return (skills_frame, LS, V, MSE)
####################################################################################################
####################################################################################################
def Elo(results_in, sigma, K, PAR):
    # results: Pandas frame with columns {"team_home","team_away","game_result","time_stamp"}
    # sigma : scaling
    # K: adaptation step
    # PAR: dictionary {"kappa", "home"}

    def e10(x):
        return 10 ** (x / sigma)

    def F_kappa(x):
        return (e10(x / 2) + 0.5 * kappa) / (e10(-x / 2) + e10(x / 2) + kappa)

    def Proba(x):
        pout = np.zeros()
        return (e10(x / 2) + 0.5 * kappa) / (e10(-x / 2) + e10(x / 2) + kappa)

    results = results_in.copy()
    kappa = PAR["kappa"]  # draw parameter
    home = PAR["home"]  # HFA

    N = len(results)  # number of games

    unique_players = set(results["team_home"].unique())  # set
    unique_players.update(results["team_away"].unique())
    unique_players = list(unique_players)  # convert to list

    M = len(unique_players)  # number of players
    Player2index = {unique_players[i]: i for i in range(M)}  # dictionary assigning player identifies to indices

    skills = np.zeros((N, M))  # all skills through time
    proba = np.zeros((N, 3))  # all skills through time
    theta = np.zeros(M)  # holder of skills

    ####   main loop
    for n in range(N):
        i_t = Player2index[results["team_home"][n]]
        j_t = Player2index[results["team_away"][n]]
        z_t = theta[i_t] - theta[j_t]
        z_t += home * sigma
        ex_score = F_kappa(z_t)
        y_t = results["game_result"][n]

        # update skills
        theta[i_t] += (K * sigma) * (y_t - ex_score)
        theta[j_t] -= (K * sigma) * (y_t - ex_score)

        skills[n, :] = theta

    skills_frame = pd.DataFrame(skills, columns=unique_players)

    return skills_frame
########################################################################
########################################################################
def batch_rating_optimize(results_in, PAR):

    def J_objective(theta, XX, yy, xi, is_home, PAR):

        home = PAR["home"] * is_home
        scale = PAR["scale"]
        gamma = 1/PAR["v0"]

        zz = np.dot(theta, XX)
        M = len(theta)
        N = len(zz)

        if PAR["rating_model"] == "Skellam":
            (g, h, L) = ghL_func((zz / scale + home), yy, PAR)
            J = np.log(L).sum()
        else:
            (g, h, L) = ghL_func((zz/scale + home)/2, yy, PAR)
            J = (np.log(L[yy, np.arange(N)]) * xi).sum()
            g /= 2

        grad = np.dot(XX, g*xi) / scale


        # change sign (this is minimization) and add the regularization term
        grad = -grad + gamma * theta
        J = -J + 0.5 * gamma * (np.linalg.norm(theta)**2)

        return J, grad

    ######## - start the function
    results = results_in.copy()
    if "game_result" not in results.columns:
        results = filter_results(results, PAR)  # adds the column of the "game_results"

    N = len(results)  # number of games

    unique_players = set(results["team_home"].unique())
    unique_players.update(results["team_away"].unique())
    unique_players = list(unique_players)
    M = len(unique_players)  # number of players
    #player2index = {unique_players[i]: i for i in range(M)}  # dictionary assigning player identifies to indices
    p2i = pd.Series(np.arange(M), index=unique_players)   # assigning player identifies to indices
    F = 1  # number of players in each group (home or away)

    ii = p2i[results["team_home"].values].values
    jj = p2i[results["team_away"].values].values
    X = np.zeros((M, N))
    X[ii, np.arange(N)] = 1
    X[jj, np.arange(N)] = -1
    theta = np.zeros(M)  # holder of skills
    
    xi_zeta = np.ones(N)  # no weighting
    yy = results["game_result"].values
    if PAR["rating_model"] == "Davidson":
        xi = np.ones(N)
        zeta = np.ones(N)
        if "weights_xi" in PAR:
            weights_xi = PAR["weights_xi"].copy()
            cat = results["category"].values
            xi = weights_xi[cat]
        if "weights_zeta" in PAR:
            weights_zeta = PAR["weights_zeta"].copy()
            zeta_bins = PAR["zeta_bins"]
            mov = np.digitize(np.abs(results["goals_home"] - results["goals_away"]).values, bins=zeta_bins)
            zeta = weights_zeta[mov]
        xi_zeta = xi * zeta
        

    is_home = (results["venue"] == "home").values * 1.0

    optimization_out = minimize(J_objective, theta, args=(X, yy, xi_zeta, is_home, PAR), method="CG",
                                jac=True, options={"gtol": 1.e-8})
    theta = optimization_out.x
    skills_out = pd.Series(theta, index=unique_players)

    ## approximate LOO cross-validation

    home = PAR["home"] * is_home
    scale = PAR["scale"]
    alpha = 1 / PAR["v0"]
    zz = np.dot(theta, X)
    if PAR["rating_model"] == "Skellam":
        (g, h, L) = ghL_func(zz / scale + home, yy, PAR)
    else:
        (g, h, L) = ghL_func((zz/scale + home)/2, yy, PAR)
        g /= 2      # takes care of the division zz/ 2
        h /= 4      # takes care of the division zz/ 2
    H0_inv = np.linalg.inv(np.dot(X*(xi_zeta*h/(scale**2)), X.T) + alpha*np.identity(M))
    a = (np.dot(H0_inv, X)*X).sum(axis=0)
    zz = np.dot(theta, X) + (xi_zeta*g*a*scale)/(xi_zeta*a*h-scale**2)

    return skills_out, optimization_out, zz

########################################################################
########################################################################
def batch_rating(results_in, PAR, Newton=True):

    results = results_in.copy()
    if "game_result" not in results.columns:
        results = filter_results(results, PAR)  # adds the column of the "game_results"

    if Newton: v0 = PAR["v0"]
    home = PAR["home"]
    scale = PAR["scale"]
    step = PAR["batch_step"]

    if not Newton and step > 0.01:
        raise Warning("step seems to be large in the batch rating")

    N = len(results)  # number of games

    unique_players = set(results["team_home"].unique())
    unique_players.update(results["team_away"].unique())
    unique_players = list(unique_players)
    M = len(unique_players)  # number of players
    p2i = pd.Series(np.arange(M), index=unique_players)  # assigning player identifies to indices
    ii = p2i[results["team_home"].values].values
    jj = p2i[results["team_away"].values].values

    XX = np.zeros((M, N))
    XX[ii, np.arange(N)] = 1
    XX[jj, np.arange(N)] = -1
    yy = results["game_result"].values

    theta = np.zeros(M)  # holder of skills
    gg = np.zeros(M)
    grad_th = 1.e-6

    while True:
        # zz = theta[ii] - theta[jj]
        zz = np.dot(theta, XX)
        (g, h, L) = ghL_func(zz/scale + home, yy, PAR)
        J = np.log(L[yy, np.arange(N)]).sum() - (np.linalg.norm(theta)**2)/(2*v0)
        grad = np.dot(XX, g)/scale - theta/v0

        hess = np.zeros((M, M))
        if Newton:
            hess2 = np.dot(XX*h, XX.T)
            for n in range(N):
                hess[ii[n], ii[n]] += h[n]
                hess[jj[n], jj[n]] += h[n]
                hess[ii[n], jj[n]] -= h[n]
                hess[jj[n], ii[n]] -= h[n]

            hess /= (scale**2)
            hess += np.identity(M)/v0
            dd = np.linalg.solve(hess, grad)   # solving (Hess+I/v0) *dd =grad
        else:
            dd = grad

        if np.linalg.norm(grad) < grad_th*M:
            break

        theta += step * dd

    skills_out = pd.Series(theta, index=unique_players)

    J = np.log(L[yy, np.arange(N)]).sum() - (np.linalg.norm(theta)**2)/(2*v0)
    extra = {"gradient": grad, "hessian": hess, "skills_difference": zz, "J_final": J}
    return (skills_out, extra)

####################################################################################################
########################################################################
def batch_rating_logistic_regression(results_in, PAR):
    from sklearn.linear_model import LogisticRegression

    if PAR["rating_model"] != "Bradley-Terry":
        raise Exception("Sorry dude! this works only for Bradley-Terry model")


    results = results_in.copy()
    if "game_result" not in results.columns:
        results = filter_results(results, PAR)  # adds the column of the "game_results"

    home = PAR["home"]
    scale = PAR["scale"]
    step = PAR["batch_step"]
    N = len(results)  # number of games

    unique_players = list(results["team_home"].unique())
    M = len(unique_players)  # number of players
    player2index = {unique_players[i]: i for i in range(M)}  # dictionary assigning player identifies to indices

    F = 1  # number of players in each group (home or away)
    theta = np.zeros(M)  # holder of skills

    ii = np.array([player2index[results["team_home"].values[n]] for n in range(N)])
    jj = np.array([player2index[results["team_away"].values[n]] for n in range(N)])
    yy = results["game_result"].values

    matX = np.zeros((N,M))
    for n in range(N):
        matX[n, ii[n]] = 1
        matX[n, jj[n]] = -1

    model = LogisticRegression(solver='newton-cg', random_state=0, fit_intercept=False, penalty="none")
    model.fit(matX, yy)

    skills_dict = {unique_players[i]: model.coef_[0,i] for i in range(M)}

    #extra = {"gradient": grad, "hessian": hess, "skills_difference": zz}
    return skills_dict

####################################################################################################
def MOV_parameters_from_file_or_optimization(sport_type, years_range_training, data_in, PAR_rating):

    dir = "results/"
    file_name = dir + "MOV_params_{}_{}_Delta={}".format(sport_type, years_range_training, PAR_rating["MOV.thresholds"])

    if False: #os.path.isfile(file_name):
        print(f"importing coefficients from file: {file_name}")
        out = load_save.my_load(file_name)
        PAR = out[0]
        PP = {"home": PAR["home"]}
        PP["alpha"] = PAR["alpha"]
        PP["delta"] = PAR["delta"]
    else:
        PAR = MOV_parameters_from_optimization(data_in, PAR_rating)
        PP = {"home": PAR["home"]}
        PP["alpha"] = PAR["alpha"]
        PP["delta"] = PAR["delta"]
        load_save.my_save(file_name, PP)
    return PP
####################################################################################################
def MOV_parameters_from_optimization(data_in, PAR_rating):

    data = [filter_results(dd, PAR_rating) for dd in data_in]

    scale = PAR_rating["scale"]
    J = 2 * PAR_rating["MOV.thresholds"].size
    alpha = np.zeros((J + 1, 1))
    delta = np.zeros((J + 1, 1))
    delta[:,0] = np.arange(0,J+1,1)/J
    eta = 0
    if False:
      if J == 4:
        alpha[:, 0] = np.array([0,0.18,0.35,0.18,0])
        delta[:, 0] = np.array([0, 0.31, 0.5, 1-0.31, 1])
        eta = 0.21


    step = 1.e-3
    grad_th = 1.e-4

    jj_low = np.arange(1, J/2, 1).astype(int)
    jj_mid = int(J/2)
    jj_high = np.arange(J-1, J/2, -1).astype(int)

    PAR = PAR_rating.copy()
    PAR["home"] = eta
    PAR["alpha"] = alpha
    PAR["delta"] = delta
    while True:   # iterations
        grad_alpha = np.zeros((J + 1, 1))
        grad_delta = np.zeros((J + 1, 1))
        grad_eta = 0
        for results in data:
            (skills, extra) = batch_rating(results, PAR, Newton=False)
            yy = results["game_result"].values
            zz = extra["skills_difference"]
            num = 10**(alpha + 2*delta * (zz/scale+eta))
            den = num.sum(axis=0)
            for j in jj_low:
                grad_alpha[j,0] += ((yy == j) - num[j, :]/den).mean()
                grad_delta[j,0] += (((yy == j) - num[j, :] / den) * (zz / scale + eta)).mean()
                j2 = J - j
                grad_alpha[j,0] += ((yy == j2) - num[j2, :] / den).mean()
                grad_delta[j,0] -= (((yy == j2) - num[j2, :] / den) * (zz / scale + eta)).mean()
            if np.mod(J,2)==0:
                j = jj_mid
                grad_alpha[j,0] += ((yy == j) - num[j, :] / den).mean()

            grad_eta += (delta[yy,0] - (delta.transpose() @ num )/den).mean(axis=1)

        if (np.linalg.norm(grad_alpha) < grad_th) and (np.linalg.norm(grad_delta) < grad_th) and (np.abs(grad_eta) < grad_th):
            break
        # zdiv = np.linalg.norm(np.concatenate((grad_alpha, grad_delta), axis=0))
        zdiv = 1
        alpha += step*grad_alpha/zdiv
        delta += step*grad_delta/zdiv
        eta += step*grad_eta/zdiv

        alpha[jj_high, 0] = alpha[jj_low, 0]
        delta[jj_high, 0] = 1-delta[jj_low, 0]

        PAR["alpha"] = alpha
        PAR["delta"] = delta
        PAR["home"] = eta

    return PAR

####################################################################################################
def parameters_from_frequency(data_in, PAR_rating):
    # data is a list of imported and filtered data (must contain the integer "game_result" field

    data = data_in.copy()
    hist = frequency_count(data, PAR_rating)

    model = PAR_rating["rating_model"]
    if model == "Bradley-Terry":
        if not set(hist.index).issubset({0,1}):
            raise Exception("the game results must be in the set {0,1}")
        PAR = {"home": np.log10(hist[1]/hist[0])}

    elif model == "Thurston":
        if not set(hist.index).issubset({0, 1}):
            raise Exception("the game results must be in the set {0,1}")
        raise Exception("not implemented for Thurston model")

    elif model == "Davidson":
        if not set(hist.index).issubset({0, 1, 2}):
            raise Exception("the game results must be in the set {0,1,2}")
        PAR = {"home": 0.5 * np.log10(hist[2] / hist[0])}
        PAR["draw"] = hist[1]/np.sqrt(hist[2] * hist[0])
    elif model == "MOV":
        d = PAR_rating["MOV.thresholds"] + 0.5
        J = 2*d.shape[0]
        alpha = np.zeros((J+1,1))
        delta = np.zeros((J+1,1))
        delta[J] = 1
        delta[0] = - 1
        ff = hist.values
        xi2 = ff[0]*ff[J]
        eta = 0.5 * np.log10(ff[J]/ff[0])
        alpha[1:J,0] = 0.5*np.log10(ff[1:J]*ff[J-1:0:-1]/xi2)
        delta[1:J,0] = np.log10(ff[1:J]/ff[J-1:0:-1])/(2*eta)
        delta = 0.5*(delta-delta[0])
        PAR = {"home": eta}
        PAR["alpha"] = alpha
        PAR["delta"] = delta

    if type(data) == list:
        MM = np.array([len(list(dd["team_home"].unique())) for dd in data])
        if not all(MM == MM[0]):
            raise Exception("number of teams in some seasons changed")
        M = MM[0]
        NN = np.array([len(dd) for dd in data])
        if not all(NN == NN[0]):
            raise Exception("number of games in some seasons changed")
        N = NN[0]
    else:
        M = len(list(data["team_home"].unique()))
        N = len(data)

    PAR["games.nr"] = N
    PAR["players.nr"] = M
    PAR["freq"] = hist      # suppose it is normalize to represent the probability
    PAR["entropy.freq"] = - sum(hist * np.log(hist))

    return PAR
####################################################################################################
def frequency_count(data_in, PAR_model):

    data = data_in.copy()
    if type(data) == list:
        data = pd.concat(data, axis=0, ignore_index=True)

    if "game_result" not in data.columns:
        data = filter_results(data, PAR_model)

    hist = data["game_result"].value_counts(normalize=True)
    hist.sort_index(inplace=True)

    return hist

####################################################################################################
def filter_results(results_in, PAR):
    # transform raw results into integer results by discretizing the goals difference

    results = results_in.copy()
    if type(results) == list:
        result_filtered = [filter_results(res, PAR) for res in results]
        return result_filtered

    model = PAR["rating_model"]
    game_results = -np.ones(results.shape[0]).astype(int)
    diff_goals = results["goals_home"] - results["goals_away"]

    if model in {"Thurston", "Bradley-Terry"}:
        game_results[diff_goals > 0] = 1
        game_results[diff_goals < 0] = 0
    else:
        if "SOOT" in results.columns:  # shootouts and overtime --> DRAW (for NHL)
            diff_goals[results["SOOT"] == "SO"] = 0
            diff_goals[results["SOOT"] == "OT"] = 0

        if model == "Davidson":
            game_results[diff_goals > 0] = 2  ## must be here (before draws are evaluated)
            game_results[diff_goals == 0] = 1
            game_results[diff_goals < 0] = 0
        elif model == "MOV":
            d = PAR["MOV.thresholds"]+0.5
            dd = np.concatenate((-d[::-1], d))   # full thresholding
            game_results = np.digitize( diff_goals, bins=dd)
        elif model == "Skellam":
            game_results = diff_goals
        else:
            raise Exception("this model is undefined")

    if "game_result" in results:
        results["game_result"] = game_results
    else:
        results.insert(5, "game_result",  game_results)
    return results