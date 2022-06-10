import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import generate_data as gen_data
import rating_algorithms
from rating_algorithms import Kalman

dir = "results/"
# fig_dir= "figures/"

## define parameters
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('font', size=15)

# line_styles = [":", "-", "-.", "--", (0, (4, 8))]
# line_colors = {"KF":"m", "vSKF": "r", "sSKF" :"b", "fSKF": "k", "SG":"g", "Glicko":"c", "TrueSkill":"tab:orange"}
# line_colors_n = ["m", "r", "b", "k", "g"]
# marker_styles = {"KF":"o", "vSKF": "s", "sSKF" :"D", "Glicko": "p", "TrueSkill": "*"}
# marker_styles_n = ["o", "s", "X"]
# marker_styles_v = ["^", "v", "X", "P", "<", ">"]

##############################################################################################################
##############################################################################################################
#####    FIGURE 5 : show the trajectories
# save_figure = True
PAR_rating = {"beta": 1.0}
# PAR_rating["v0"] = 1  # 1.0  #
# PAR_rating["epsilon"] = 0.001  # 0.004
PAR_rating["rating_model"] = "Davidson"  # rating_model = "Thurston", "Bradley-Terry", "Gauss"
PAR_rating["scale"] = 1  #
PAR_rating["rating_algorithm"] = "vSKF"  # "KF", "vSKF", "sSKF", "fSKF", "Glicko"
# PAR_rating["home"] = 0
PAR_rating["metric_type"] = "LS_0"  # "exact", "empirical", "empirical_0", "empirical_avg", "DKL"

## define the data
show_vacations = False  # show vacations time (no games) or stitch the seasons without any break
all_sport_types = ["EPL", "NHL"]     # ["EPL", "NHL"]
for sport_type in all_sport_types:
    if sport_type == "NHL":
        years_range_training = list(range(2005, 2009, 1))
        years_range_eval = list(range(2009, 2012, 1)) + list(range(2013, 2015, 1))  ## 2012-13 removed (short season)
        years_range = years_range_training + years_range_eval
    elif sport_type == "EPL":
        years_range = list(range(2009, 2019, 1))
    elif sport_type == "NFL":
        years_range = list(range(2009, 2019, 1))

    imported_data = gen_data.import_data(sport_type, years_range)
    # estimate the draw- and the HFA parameters from the overall statistics
    PP = rating_algorithms.parameters_from_frequency(imported_data, PAR_rating)
    PAR_rating.update(PP)

    ###### extracting information about the seasons and appearing/disappearing players (only for plotting the data)
    results1 = imported_data[0]
    unique_players1 = results1["team_home"].unique()
    results2 = imported_data[1]
    unique_players2 = results2["team_home"].unique()
    eliminated_players = set(unique_players1).difference(unique_players2)
    print(f"eliminated players: {eliminated_players}")
    new_players = set(unique_players2).difference(unique_players1)
    print(f"new players: {new_players}")
    # merge the sets : we have to fix the time stamps
    last_stamp_season1 = results1["time_stamp"].max()
    z1 = pd.to_datetime(results1["Date"], dayfirst=True)
    z2 = pd.to_datetime(results2["Date"], dayfirst=True)
    if show_vacations:
        first_stamp_season2 = (z2[0]-z1[0]).days
    else:
        first_stamp_season2 = last_stamp_season1 + 1
    results2["time_stamp"] += first_stamp_season2

    ## concantenate two seasons
    results = pd.concat([results1, results2])

    ## define the optimized parameters for the algorithms vSKF
    PAR_rating["rating_algorithm"] = "vSKF"
    if sport_type == "EPL":
        PAR_rating["v0"] = 0.04
        PAR_rating["epsilon"] = 0   #  1e-7
    elif sport_type == "NHL":
        PAR_rating["v0"] = 0.003
        PAR_rating["epsilon"] = 3e-5
    else:
        raise Exception("sport undefined")
    ## run the algorithm
    (Kalman_rating, LS_tmp, V_list, MSE_tmp) = Kalman(results, PAR_rating)
    Kalman_sigma = np.sqrt(pd.DataFrame(V_list, columns=Kalman_rating.columns))

    ## define the optimized parameters for the algorithms SG
    PAR_rating["rating_algorithm"] = "SG"
    if sport_type == "EPL":
        PAR_rating["v0"] = 0.015
    elif sport_type == "NHL":
        PAR_rating["v0"] = 0.003
    else:
        raise Exception("stop!")
    # run the algorithm
    (SG_rating, LS_tmp_SG, V_list_SG, MSE_tmp) = Kalman(results, PAR_rating)

    ## the rest is about plotting

    (fig, ax) = plt.subplots()
    time = results["time_stamp"]
    unique_players = results["team_home"].unique()
    ### define which players are potentially interesting and define the position at which their names are displayed
    if sport_type == "EPL":
        players_of_interest = ['Aston Villa', 'Chelsea', 'Portsmouth', 'Wolves', 'Sunderland', 'Blackpool']
        text_pos =              [(100,0.0), (400, 0.3),   (200,-0.3), (50,-0.08), (200,0.1), (300,-0.1)]
        ## you want to see these players
        players_to_show = ['Chelsea', 'Portsmouth', 'Aston Villa', 'Blackpool']
    elif sport_type == "NHL":
        players_of_interest = ['Chicago Blackhawks','Detroit Red Wings','Toronto Maple Leafs','St. Louis Blues','Ottawa Senators','San Jose Sharks']
        text_pos =          [(50, -0.15),           (250, 0.25) ,             (220, 0.05),            (50, -0.08),    (200, 0.1),     (400, -0.1)]
        ## you want to see these players
        players_to_show = ['Chicago Blackhawks', 'Detroit Red Wings', 'Toronto Maple Leafs' ]
    label2pos = {players_of_interest[i]: text_pos[i] for i in range(len(text_pos))}

    for player in players_to_show:
        if player in eliminated_players:    # indices to the first season
            fi = range((time <= last_stamp_season1).sum())
        elif player in new_players:  #   indices to the second season
            fi =  range((time >= first_stamp_season2).sum(), len(time))
        else:  # all elements
            fi  = range(len(time))

        a = ax.plot(time.iloc[fi], Kalman_rating[player].iloc[fi], linestyle='-', label=player)
        ax.fill_between(time.iloc[fi], Kalman_rating[player].iloc[fi] + Kalman_sigma[player].iloc[fi],
                                    Kalman_rating[player].iloc[fi] - Kalman_sigma[player].iloc[fi], alpha=0.2)
        ax.plot(time.iloc[fi], SG_rating[player].iloc[fi], linestyle='--', color=a[0].get_color())

    ax.grid()
    ax.set_xlim(1, time.iloc[-1])

    # show limits of the seasons
    if sport_type == "NHL":
        updown_limits = 0.3
    elif sport_type == "EPL":
        updown_limits = 0.45
    ax.plot(last_stamp_season1*np.array([1,1]), updown_limits*np.array([-1,1]), linestyle="--", color="k")
    if show_vacations:
        ax.plot(first_stamp_season2 * np.array([1, 1]), updown_limits*np.array([-1,1]), linestyle="--", color="k")
    year_season1 = z1[0].year-2000
    position_season1 = 0.45*last_stamp_season1
    position_season2 = first_stamp_season2 + position_season1
    ax.text(position_season1, -updown_limits, "20{:02d}/{:02d}".format(year_season1,year_season1+1))
    ax.text(position_season2, -updown_limits, "20{:02d}/{:02d}".format(year_season1+1,year_season1+2))

    for player in players_to_show:
        ax.text(label2pos[player][0], label2pos[player][1], player)

    ax.set_ylabel("$\\mu_{\\tau,m}$, $\\mu_{\\tau,m}\pm \sqrt{v_{\\tau,m}}$")
    ax.set_xlabel("$\\tau$ [day]")
    plt.subplots_adjust(left=0.15, bottom=0.15)

    #if save_figure:
    #    plt.savefig(fig_dir + PAR_rating["rating_model"] + sport_type + str(years_range[0]) + "_trajectories.png", transparent=True, format="png")
plt.show()
