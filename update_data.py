import pandas as pd
import numpy as np
from io import StringIO
from collections import Counter, OrderedDict

wddict={True: 'W', False: 'L'}
def wd(x):
    return list(map(lambda y: wddict[y], x))

def order_seeds(x):
    return list(map(lambda y: 10 if y == 0 else y, x))

def dfx(i):
    dff = pd.DataFrame(data=np.c_[df.Team, df['W12 Opponent'].values, np.round(pts_w12[i], 1), wd(wins12[i]),  df['W13 Opponent'].values, np.round(pts_w13[i], 1), wd(wins13[i]), np.round(total_pts[i], 1), total_wins[i],
                                 seeds[i]],
                        columns=['Team', 'W12 Opponent', 'W12Pts', 'W12W', 'W13 Opponent', 'W13Pts', 'W13W', 'Pts', 'Wins', 'Seed'])
    dff['Sim'] = i
    dffOut = dff.set_index(['Team'])
    return dffOut

def indices(opponents):
    return [list(teams).index(x) for x in opponents]

def match_inds(src, dest):
    src_list = list(src)
    return [src_list.index(x) for x in dest]

def gamma(mean, shape):
    return np.random.default_rng().gamma(mean/5, 5, shape)

def playoff_seeds(wins, pts):
    df1 = pd.DataFrame(data=np.c_[wins,pts], index=teams, columns=["Wins", "Pts"])
    df1_sorted = df1.sort_values(["Wins", "Pts"], ascending=False)
    record_qualifiers = df1_sorted[:4].index.values
    pts_qualifiers = teams[np.argsort(pts)[::-1]]
    teams_sorted = unique_unsorted(np.r_[record_qualifiers, pts_qualifiers])[:6]
    return teams_sorted

def unique_unsorted(x):
    return x[sorted(np.unique(x, return_index=True)[1])]

def seed(wins, pts):
    playoff_teams = playoff_seeds(wins, pts)
    out = []
    for team in teams:
        if team in playoff_teams:
            out.append(list(playoff_teams).index(team)+1)
        else:
            out.append(0)
    return out


x = """Team	Wins	Pts	W12 Opponent	W13 Opponent	Pts/week	Regressed
Daniel	8	1360.58	Mike	AJ	123.7	123.0
Torry	7	1385.48	James	Mike	126.0	124.3
Mitch	6	1435.58	Bryan	David	130.5	127.0
Ryan	5	1427.78	David	Bryan	129.8	126.6
Parm	4	1445.7	AJ	James	131.4	127.6
Mike	6	1336.36	Daniel	Torry	121.5	121.6
James	6	1328.28	Torry	Parm	120.8	121.2
AJ	6	1249.32	Parm	Daniel	113.6	116.9
David	5	1297.4	Ryan	Mitch	117.9	119.5
Bryan	2	1136.7	Mitch	Ryan	103.3	110.7"""
df = pd.read_table(StringIO(x))

teams = df["Team"].values
w12Opp = df['W12 Opponent'].values

w13Opp = df['W13 Opponent'].values

wins = df.Wins.values
pts = df.Pts.values
ptsMean = pts.mean()/11

df12 = pd.read_excel('live-pts.xlsx', 'W12')

scale_factor = 0.953

df12['left'] = (df12['Proj'] - df12['Pts'])*scale_factor # Scale factor
df12Inds = match_inds(df12.Team.values, teams)

df13 = pd.read_excel('live-pts.xlsx', 'W13')
df13['TempProj'] = (df13['Proj']*0.6 + ptsMean*0.4/scale_factor) # Fix by Thursday
df13['left'] = (df13['TempProj'] - df13['Pts'])*scale_factor # Scale factor
df13Inds = match_inds(df13.Team.values, teams)


# Pts
N = 5000
curr_pts12 = df12.Pts.values[df12Inds]
pts_left12 = df12.left.values[df12Inds]


pts_w12 = np.r_[[gamma(pts, N) for pts in pts_left12]].T + curr_pts12
wins12 = pts_w12 > pts_w12[:, indices(w12Opp)]

curr_pts13 = df13.Pts.values[df13Inds]
pts_left13 = df13.left.values[df13Inds]

pts_w13 = np.r_[[gamma(pts, N) for pts in pts_left13]].T + curr_pts13
wins13 = pts_w13 > pts_w13[:, indices(w13Opp)]


total_pts = pts_w12+pts_w13+pts
total_wins = wins + wins12.astype(int) + wins13.astype(int)

seeds = np.array([seed(total_wins[i], total_pts[i]) for i in range(N)])

np.save('w12.npy', pts_w12)
np.save('w13.npy', pts_w13)
np.save('seeds.npy', seeds)