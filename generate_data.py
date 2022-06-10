import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats._continuous_distns import _norm_cdf

########################################################################
def generate_skills_walk(PAR):
    players_nr = PAR["M"]  # number of players
    days_nr = PAR["D"]  # number of days
    beta = PAR["beta"]  # dumping factor

    epsilon = 1 - beta ** 2  # epsilon: variation on per-day basis
    th = np.random.normal(loc=0, scale=1, size=(players_nr, days_nr))
    if PAR["scenario"] == "classic":
        for i in range(1, days_nr):
            th[:, i] *= np.sqrt(epsilon)
            th[:, i] += beta * th[:, i - 1]
    elif PAR["scenario"] == "switch":
        d_switch = PAR["d_switch"]
        n_switch = PAR["n_switch"]
        for i in range(1, days_nr):
            if i != d_switch:
                th[:, i] *= np.sqrt(epsilon)
                th[:, i] += beta * th[:, i - 1]
            else:
                #   ii_switch = np.arange(n_switch)        #    switch these players (
                ii_stay = np.arange(n_switch, players_nr)   # keep theses players
                th[ii_stay, i] *= np.sqrt(epsilon)
                th[ii_stay, i] += beta * th[ii_stay, i - 1]
    else:
        raise Exception("generate skills: scenario undefined")


    return th

########################################################################
def generate_skills_deterministic(PAR):
    players_nr = PAR["M"]  # number of players
    days_nr = PAR["D"]  # number of days
    # mM
    mM = np.arange(1, players_nr + 1) / players_nr  # (1/M : 1)

    amplitude = np.zeros((players_nr, 1))
    amplitude[:, 0] = 0.25 * (mM + 2)

    frequency = np.zeros((players_nr, 1))
    frequency[:, 0] = 0.25 * (mM)

    offset = np.zeros((players_nr, 1))
    offset[:, 0] = np.pi * (2 * mM)

    time = np.ones((1, days_nr))
    time[0, :] = np.arange(days_nr)
    th = amplitude * np.cos(frequency * np.pi / days_nr * time + offset)
    return th

########################################################################
def generate_outcomes(skills, PAR):
    # all players play each day

    (players_nr, days_nr) = skills.shape

    players_list = np.arange(0, players_nr, 1)
    model = PAR["model"]
    sigma = PAR["sigma"]

    xx = np.zeros((1, 2))
    xx[0] = [1, -1]
    results_all = list([])
    games_per_day = int(players_nr / 2)  # number of games per day

    for day in range(days_nr):
        players_permuted = np.random.permutation(players_list)
        schedule_day = players_permuted.reshape(2, int(players_nr / 2))
        z = np.matmul(xx, skills[schedule_day, day])  # difference between the skills of the schedules players

        if model == "Gauss":
            vv = np.random.normal(loc=0, scale=1, size=z.shape)  ## Gaussian random numbers (zero-mean, unit variance)
            game_outcome = z + vv * sigma
            p = np.zeros_like(z)    # dummy variable
        elif model == "Thurston":
            # p = stats.norm.cdf(z / sigma)  # probability of home win, Thurston model
            p = _norm_cdf(z / sigma)  # probability of home win, Thurston model
            vv = np.random.random(z.shape)  ## random uniformly distributed numbers
            game_outcome = np.zeros(z.shape).astype(int)
            game_outcome[vv <= p] = 1
        elif model == "Bradley-Terry":
            # p = stats.logistic.cdf(z * np.log(10) / sigma)  # probability of home win, B-T model
            p = 1 / (1 + 10 ** (z/sigma))
            vv = np.random.random(z.shape)  ## random uniformly distributed numbers
            game_outcome = np.zeros(z.shape).astype(int)
            game_outcome[vv <= p] = 1
        else:
            raise Exception("Generate outcome: wrong model")

        for n in range(games_per_day):
            time_stamp = day
            result = {"team_home": schedule_day[0, n], "team_away": schedule_day[1, n],
                      "game_result": game_outcome[0, n], "time_stamp": time_stamp, "real_proba" : p[0, n]}
            results_all.append(result.copy())

    results_all_pd = pd.DataFrame(results_all)

    return results_all_pd

########################################################################
def import_data(sport_type, years_range):

    # sport_type : "NHL', "EPL"


    if sport_type == "NHL":
        dir = "DATA/NHL/"
        save_file_name_template = "NHL-{:4d}-{:4d}.csv"

        imported_data = []   # holder for the final results
        for year in years_range:
            df = pd.read_csv(dir + save_file_name_template.format(year, year + 1))

            df.drop(columns=["Att.", "LOG", "Notes"], inplace=True)
            name_changer = {"Visitor": "team_away", "Home": "team_home"}
            name_changer["G"] = "goals_away"
            name_changer["G.1"] = "goals_home"
            name_changer["Unnamed: 5"] = "SOOT"
            df.rename(columns=name_changer, inplace=True)
            df.fillna("", inplace=True)
            z = pd.to_datetime(df["Date"])
            day_stamp = [ (zz-z[0]).days for zz in z]
            # df["time_stamp"] = day_stamp # tranform to time difference in days
            df.insert(5, "time_stamp", day_stamp)

            goals = df["goals_home"]
            ind_drop = list(goals[goals == ""].index)   # find unspecified goal (game not played)
            df.drop(index=ind_drop, inplace=True)

            unique_players_home = df["team_home"].unique()
            unique_players_away = df["team_away"].unique()
            if len(unique_players_away) != len(unique_players_home):
                raise Exception("numbers of unique home- and away- teams are not the same")


            imported_data.append(df)
    elif sport_type == "EPL":
        dir = "DATA/EPL/"
        save_file_name_template = "season-{:02d}{:02d}.csv"

        imported_data = []  # holder for the final results
        for year in years_range:
            df = pd.read_csv(dir + save_file_name_template.format(year-2000, year - 2000 + 1))

            df.drop(columns=["Div", "Referee", "FTR", "HTHG", "HTAG", "HTR"], inplace=True)
            name_changer = {"AwayTeam": "team_away", "HomeTeam": "team_home"}
            name_changer["FTAG"] = "goals_away"
            name_changer["FTHG"] = "goals_home"
            df.rename(columns=name_changer, inplace=True)
            df.fillna("", inplace=True)
            z = pd.to_datetime(df["Date"], dayfirst=True)
            day_stamp = [(zz - z[0]).days for zz in z]
            if any(np.array(day_stamp) < 0):
                raise Exception("something is wrong: negative time difference")
            # df["time_stamp"] = day_stamp  # tranform to time difference in days
            df.insert(5, "time_stamp", day_stamp)

            goals = df["goals_home"]
            ind_drop = list(df.loc[goals == ""].index)  # find unspecified goal (game not played)
            df.drop(index=ind_drop, inplace=True)

            unique_players_home = df["team_home"].unique()
            unique_players_away = df["team_away"].unique()
            if len(unique_players_away) != len(unique_players_home):
                raise Exception("numbers of unique home- and away- teams are not the same")

            imported_data.append(df)
    elif sport_type == "NFL":
        dir = "DATA/NFL/"
        save_file_name_template = "NFL-{:4d}-{:4d}.csv"

        imported_data = []   # holder for the final results
        for year in years_range:
            df = pd.read_csv(dir + save_file_name_template.format(year, year + 1))

            df.drop(columns=["Unnamed: 7", "YdsW", "TOW", "YdsL", "TOL", "Time"], inplace=True)
            name_changer = {"Winner/tie": "team_home", "Loser/tie": "team_away"}
            name_changer["Pts"] = "goals_home"
            name_changer["Pts.1"] = "goals_away"
            name_changer["Unnamed: 5"] = "HA-switch"    # the sign "@" indicates that the winner is away team
            df.rename(columns=name_changer, inplace=True)
            df.fillna("", inplace=True)

            ind_drop = list(df.loc[df["Week"] == "Week"].index)   # find empty lines (weeks separators)
            df.drop(index=ind_drop, inplace=True)
            ind_drop2 = list(range(df.loc[df["Date"] == "Playoffs"].index[0], df.index[-1]+1))  # remove from playoffs to the end
            df.drop(index=ind_drop2, inplace=True)

            switch_ind = df.loc[df["HA-switch"] == "@"].index
            player_tmp = df["team_home"][switch_ind]
            df["team_home"][switch_ind] = df["team_away"][switch_ind]
            df["team_away"][switch_ind] = player_tmp

            goals_tmp = df["goals_home"][switch_ind]
            df["goals_home"][switch_ind] = df["goals_away"][switch_ind]
            df["goals_away"][switch_ind] = goals_tmp

            df.drop(columns=["HA-switch"], inplace=True)


            z = pd.to_datetime(df["Date"])
            day_stamp = [(zz - z[0]).days for zz in z]
            if any(np.array(day_stamp) < 0):
                raise Exception("something is wrong: negative time difference")
            # df["time_stamp"] = day_stamp  # transform to time difference in days
            df.insert(5, "time_stamp", day_stamp)


            unique_players_home = df["team_home"].unique()
            unique_players_away = df["team_away"].unique()
            if len(unique_players_away) != len(unique_players_home):
                raise Exception("numbers of unique home- and away- teams are not the same")


            imported_data.append(df)
    else:
        raise Exception("sport:" + sport_type + " is undefined")

    return imported_data

