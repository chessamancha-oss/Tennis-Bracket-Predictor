# ultimate_tennis_tournament_final.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# -----------------------------
# Step 1: Gather Player Info
# -----------------------------
players = []

num_players = int(input("Enter number of players (power of 2: 8, 16, 32): "))
assert num_players & (num_players - 1) == 0, "Number of players must be a power of 2!"

for i in range(num_players):
    print(f"\nEnter info for Player {i+1}:")
    name = input("Name: ")
    age = int(input("Age: "))
    serve = float(input("Serve accuracy (0-1): "))
    ret = float(input("Return accuracy (0-1): "))
    aces = float(input("Average aces per match: "))
    df = float(input("Average double faults per match: "))
    recent_win = float(input("Recent win ratio (0-1): "))
    straight_sets = float(input("Straight sets win ratio (0-1): "))
    win_right = float(input("Win ratio vs right-handers (0-1): "))
    win_left = float(input("Win ratio vs left-handers (0-1): "))
    injury = float(input("Recent injuries impact (0-1): "))
    
    players.append({
        "Name": name,
        "Age": age,
        "Serve": serve,
        "Return": ret,
        "Aces": aces,
        "Double_Faults": df,
        "Recent_Win": recent_win,
        "Straight_Sets": straight_sets,
        "Win_Right": win_right,
        "Win_Left": win_left,
        "Injury": injury
    })

df = pd.DataFrame(players)
main_player_name = input("\nEnter the name of the player to simulate as 'you': ")

# -----------------------------
# Step 2: Court & Tournament Type
# -----------------------------
court_type = input("Enter court type (Grass / Hard / Clay): ").strip().lower()
tournament_type = input("Enter tournament type (Open / Grand Slam / Local): ").strip().lower()

# -----------------------------
# Step 3: Win Probability Function
# -----------------------------
def realistic_win_prob(p, o, court=court_type, tour=tournament_type):
    serve_factor = p['Serve']*0.35 + p['Return']*0.25
    ace_factor = (p['Aces']/(p['Aces']+o['Aces']+1e-6))*0.1
    df_factor = (1 - p['Double_Faults']/(p['Double_Faults']+o['Double_Faults']+1e-6))*0.05
    form_factor = p['Recent_Win']*0.15
    straight_factor = p['Straight_Sets']*0.05
    hand_factor = ((p['Win_Right'] if o['Age']%2==0 else p['Win_Left'])*0.05)
    injury_factor = -p['Injury']*0.1
    
    # Court advantage
    court_factor = 0
    if court == 'grass':
        court_factor += p['Serve']*0.05 + (p['Aces']/10)*0.05
    elif court == 'clay':
        court_factor += p['Return']*0.05 + p['Recent_Win']*0.02
    elif court == 'hard':
        court_factor += 0.03*(p['Serve'] + p['Return'])
    
    # Tournament pressure
    tour_factor = 0
    if tour == 'grand slam':
        tour_factor -= 0.02*p['Injury']
    elif tour == 'open':
        tour_factor += 0.01*p['Recent_Win']
    
    score = serve_factor + ace_factor + df_factor + form_factor + straight_factor + hand_factor + injury_factor + court_factor + tour_factor
    opp_score = realistic_raw_score(o, p)
    prob = score / (score + opp_score)
    return min(max(prob,0),1)

def realistic_raw_score(p, o):
    serve_factor = p['Serve']*0.35 + p['Return']*0.25
    ace_factor = (p['Aces']/(p['Aces']+o['Aces']+1e-6))*0.1
    df_factor = (1 - p['Double_Faults']/(p['Double_Faults']+o['Double_Faults']+1e-6))*0.05
    form_factor = p['Recent_Win']*0.15
    straight_factor = p['Straight_Sets']*0.05
    hand_factor = ((p['Win_Right'] if o['Age']%2==0 else p['Win_Left'])*0.05)
    injury_factor = -p['Injury']*0.1
    return serve_factor + ace_factor + df_factor + form_factor + straight_factor + hand_factor + injury_factor

# -----------------------------
# Step 4: Match & Bracket Simulation
# -----------------------------
def simulate_match(p1, p2, court=court_type, tour=tournament_type):
    return p1 if np.random.rand() < realistic_win_prob(p1, p2, court, tour) else p2

def simulate_bracket(players, court=court_type, tour=tournament_type):
    rounds = int(math.log2(len(players)))
    bracket = [players.copy()]
    current_round = players.copy()
    
    for r in range(rounds):
        next_round = []
        for i in range(0, len(current_round), 2):
            winner = simulate_match(current_round[i], current_round[i+1], court, tour)
            next_round.append(winner)
        bracket.append(next_round)
        current_round = next_round
    return bracket

# -----------------------------
# Step 5: Multi-Simulation Probabilities
# -----------------------------
def tournament_probabilities(df, n_sim=5000):
    players = df.to_dict('records')
    prob_dict = {p['Name']:0 for p in players}
    for _ in range(n_sim):
        bracket = simulate_bracket(players)
        winner = bracket[-1][0]['Name']
        prob_dict[winner] += 1
    for k in prob_dict:
        prob_dict[k] /= n_sim
    return prob_dict

tourn_probs = tournament_probabilities(df, n_sim=5000)
tourn_df = pd.DataFrame(list(tourn_probs.items()), columns=['Player','Win_Prob']).sort_values('Win_Prob', ascending=False)
print("\nTournament Win Probabilities:")
print(tourn_df)

# -----------------------------
# Step 6: Bracket Tree Visualization
# -----------------------------
def draw_bracket_tree(df, tourn_probs):
    rounds = int(math.log2(len(df)))
    fig, ax = plt.subplots(figsize=(14, rounds*1.5))
    
    player_names = list(df['Name'])
    n_players = len(player_names)
    y_pos = np.linspace(0, n_players-1, n_players)
    positions = {player_names[i]: y_pos[i] for i in range(n_players)}
    
    for r in range(rounds):
        next_positions = {}
        for i in range(0, len(player_names), 2):
            p1 = player_names[i]
            p2 = player_names[i+1]
            winner = p1 if tourn_probs[p1] > tourn_probs[p2] else p2
            y_winner = (positions[p1] + positions[p2]) / 2
            next_positions[winner] = y_winner
            ax.plot([r, r+1], [positions[p1], y_winner], color='black')
            ax.plot([r, r+1], [positions[p2], y_winner], color='black')
        player_names = list(next_positions.keys())
        positions = next_positions
    
    for name, y in positions.items():
        ax.text(rounds + 0.1, y, f"{name} ({tourn_probs[name]:.1%})", va='center', fontsize=12)
    
    ax.set_xlim(-0.5, rounds+1)
    ax.set_ylim(-1, n_players)
    ax.axis('off')
    ax.set_title('Tournament Bracket Simulation', fontsize=16)
    plt.tight_layout()
    plt.show()

draw_bracket_tree(df, tourn_probs)

# -----------------------------
# Step 7: Probability Bar Chart
# -----------------------------
plt.figure(figsize=(12,6))
bars = plt.barh(tourn_df['Player'], tourn_df['Win_Prob'], color='#4C72B0', edgecolor='black')
plt.xlabel('Tournament Win Probability', fontsize=14)
plt.title('Overall Tournament Win Probabilities', fontsize=16)
plt.xlim(0,1)

for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{tourn_df["Win_Prob"].iloc[i]:.1%}', va='center', fontsize=12)

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
