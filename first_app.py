import streamlit as st

import pandas as pd
import numpy as np
import matplotlib
from io import StringIO
import copy
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

def makeSeeds(seeds):
    seedCounts = {team: Counter(s) for team, s in zip(teams, seeds.T)}
    dfSeeds = pd.DataFrame.from_dict(seedCounts, orient='index')
    dfSeedsPercent = dfSeeds.fillna(0) / len(seeds) * 100
    dfSeedsOrdered = dfSeedsPercent[[1,2,3,4,5,6,0]]
    dfS = dfSeedsOrdered.rename(columns={0:"Miss Playoffs"})
    dfS['avgSeed'] = np.round(dfS[[1,2,3,4,5,6,'Miss Playoffs']].values @ np.r_[np.arange(1,7), [8.5]] / 100, 1)
    dfS['Playoff %'] = 100 - dfS['Miss Playoffs']
    dfSS=dfS.sort_values('avgSeed', ascending=True)
    del dfSS['avgSeed']
    del dfSS['Miss Playoffs']
    return dfSS

def playoffs(wins, pts, seedsRow):
    
    seedsRow = list(seedsRow)
    bonus1 = pts[seedsRow.index(1)]*0.01
    bonus2 = pts[seedsRow.index(2)]*0.005
    inds = [seedsRow.index(i) for i in [4,5,6]]
    seed1opp = inds[np.argmin(pts[inds])]
    inds2Seed = list(inds)
    inds2Seed.remove(seed1opp)
    seed2opp = inds2Seed[np.argmin(pts[inds2Seed])]
    inds2Seed.remove(seed2opp)
    matchups = [[teams[seedsRow.index(1)], teams[seed1opp]],
                [teams[seedsRow.index(2)], teams[seed2opp]],
                [teams[seedsRow.index(3)], teams[inds2Seed[0]]]]
    dfP = pd.DataFrame(
        np.c_[pts/13, seedsRow], index=teams, columns=['avgPts', 'seed']
    )
    dfP = dfP[dfP['seed'] > 0]
    dfP['regressed'] = dfP.avgPts.mean() * 0.7 + dfP.avgPts*0.3
    playoffPts = np.random.randn(6, 3)*25 + dfP.regressed.values.reshape(6,1)
    dfP['P1'] = playoffPts[:,0]
    dfP['P2'] = playoffPts[:,1]
    dfP['P3'] = playoffPts[:,2]
    dfP['P1B'] = 0
    dfP.loc[teams[seedsRow.index(1)], 'P1B'] = bonus1
    dfP.loc[teams[seedsRow.index(2)], 'P1B'] = bonus2
    reversed_matchups = [[x[1], x[0]] for x in matchups]
    all_matchups = copy.copy(matchups)
    all_matchups.extend(reversed_matchups)
    all_matchups = np.array(all_matchups)
    dfP.loc[all_matchups[:,0], "P1Opp"]=all_matchups[:, 1]
    dfP["P1Win"] = (dfP["P1"].values + dfP['P1B'].values) > dfP.loc[dfP['P1Opp'].values].P1.values
    dfPW2 = dfP[dfP["P1Win"]]
    dfP["W2"]=dfPW2.P2 + dfPW2.P3
    dfP.sort_values('W2', ascending=False, inplace=True)
    dfP.loc[dfP.index.values, "Result"] = [1,2,3,4,4,4]
    yyy = dfP.loc[teams, 'Result']
    yyy = yyy.fillna(7)
    return yyy

def get_matchups(opponents):
    return list(set(
        tuple(sorted(x)) for x in list(zip(np.arange(10), indices(w13Opp)))
        ))

def format_table(table_str):
    return table_str.replace('border="1" class="dataframe"',
                'class="table table-striped table-sm"').replace(
                '<tr style="text-align: right;">',
                '<tr>').replace('<th></th>', '<th>Team</th>')

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
orderw12 = np.c_[np.arange(0,5), indices(w12Opp)[:5]].flatten()


w13Opp = df['W13 Opponent'].values
w13matchups = get_matchups(w13Opp)
orderw13 = np.array(w13matchups).flatten()
df['R2']=(df['Pts'].values.mean() * 0.7 + df['Pts']*0.3)/11
ptsAvg = df["R2"].values

w12Current = """
Team	Pts	Proj
Ryan	101.18	116.80
David	148.5	148.5
Mitch	91.6	91.6
Bryan	105.3	105.3
AJ	137.82	137.82
Parm	154.24	154.24
Torry	107	107
James	91.3	116.04
Daniel	97.18	97.18
Mike	135.34	145.8
"""

df12 = pd.read_excel('live-pts.xlsx', 'W12')

df12['left'] = (df12['Proj'] - df12['Pts'])*0.953 # Scale factor
df12Inds = match_inds(df12.Team.values, teams)

df13 = pd.read_excel('live-pts.xlsx', 'W13')
df13['TempProj'] = (df13['Proj']*0.6 + df.Pts.mean()*0.4*1.045) # Fix by Thursday
df13['left'] = (df13['TempProj'] - df13['Pts'])*0.953 # Scale factor


st.title('Playoff Chances')

pts_w12 = np.load('w12.npy')
pts_w13 = np.load('w13.npy')
seeds = np.load('seeds.npy')

N = len(pts_w12)

wins12 = pts_w12 > pts_w12[:, indices(w12Opp)]
wins13 = pts_w13 > pts_w13[:, indices(w13Opp)]

total_pts = pts_w12+pts_w13+df["Pts"].values
total_wins = df['Wins'].values + wins12.astype(int) + wins13.astype(int)



slot1 = st.empty()

st.write("Week 12")
TJwinner = st.radio(label='Winner', options=['Any', 'Torry', 'James'])

st.write("Week 13")
n_cols = 5
cols = st.beta_columns(n_cols)

buttons = []
for i, x in enumerate(w13matchups):
    buttons.append(cols[i % n_cols].radio(label='Winner', options=['Any', teams[x[0]], teams[x[1]]])
    )

inds = np.arange(N, dtype=int)

if TJwinner !='Any':
    inds_match = np.nonzero(wins12[:, list(teams).index(TJwinner)])
    inds = np.intersect1d(inds, inds_match, assume_unique=True)

for button in buttons:
    if button != 'Any':
        inds_match = np.nonzero(wins13[:, list(teams).index(button)])
        inds = np.intersect1d(inds, inds_match, assume_unique=True)

# for button in buttons:
#     if button is not None:
#         filter()

avgWins = np.round(np.mean(total_wins[inds], axis=0),2)
avgPts = np.round(np.mean(total_pts[inds], axis=0), 1)
makePlayoffs = np.sum(seeds[inds] > 0, axis=0)

dfSS = makeSeeds(seeds[inds])

dfAvg = pd.DataFrame(data=np.c_[avgWins, avgPts, makePlayoffs/len(inds)*100],
                    index=teams,
                    columns=['avgWins', 'avgPts', 'playoffPercent'])\
                        .round(1)\
                        .sort_values('playoffPercent', ascending=False)

dfWinProb = pd.DataFrame(
    np.c_[wins12[inds].mean(axis=0)*100, df12.Pts.values[df12Inds], pts_w12[inds].mean(axis=0)], index=teams,
    columns=['Win Prob', 'Pts', 'Proj']).loc[teams[orderw12], :]

dfWinProb13 = pd.DataFrame(
    np.c_[wins13[inds].mean(axis=0)*100, pts_w13[inds].mean(axis=0)], index=teams,
    columns=['Win Prob', 'Proj']).loc[teams[orderw13], :]


slot1cols = slot1.beta_columns(2)
slot1cols[0].write("Week 12")
slot1cols[0].dataframe(
    dfWinProb.style.format("{:.1f}")\
        .background_gradient(cmap='RdBu_r', low=1.25, high=1.25, axis=0, subset=['Win Prob'])
        )

slot1cols[1].write("Week 13")
slot1cols[1].dataframe(
    dfWinProb13.style.format("{:.1f}")\
        .background_gradient(cmap='RdBu_r', low=1.25, high=1.25, axis=0, subset=['Win Prob'])
        )



st.dataframe(dfSS.style.format("{:.1f}")\
    .background_gradient(cmap='Greens', low=0.0, high=0.7))
st.dataframe(dfAvg.style.format("{:.1f}")\
    .background_gradient(cmap='RdBu_r', low=1, high=1, axis=0))
