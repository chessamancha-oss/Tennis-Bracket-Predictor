from tennis_predictor.models import Handedness, Player


def make_player(name="Player", **overrides):
    values = {
        "name": name,
        "handedness": Handedness.RIGHT,
        "serve_accuracy": 0.70,
        "return_accuracy": 0.70,
        "aces_per_match": 6.0,
        "double_faults_per_match": 2.0,
        "recent_win_ratio": 0.70,
        "straight_sets_win_ratio": 0.60,
        "win_vs_right": 0.70,
        "win_vs_left": 0.65,
        "injury_impact": 0.05,
    }
    values.update(overrides)
    return Player(**values)
