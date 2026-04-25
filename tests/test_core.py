from __future__ import annotations

import pytest

from cricket_draft_assistant.core import DraftState, MCEngine, Player, load_players_from_text


def _full_sized_state() -> DraftState:
    my_captain = Player("Me", 3, 2)
    opp_captain = Player("Opponent", 3, 2)

    my_picks = [
        Player("My Batter 1", 5, 1),
        Player("My Batter 2", 4, 1),
        Player("My Batter 3", 4, 0),
        Player("My AR 1", 3, 3),
        Player("My AR 2", 3, 3),
        Player("My Bowler 1", 2, 4),
        Player("My Bowler 2", 1, 4),
        Player("My Bowler 3", 1, 3),
        Player("My Bowler 4", 1, 3),
    ]
    opp_picks = [
        Player("Opp Batter 1", 5, 1),
        Player("Opp Batter 2", 4, 1),
        Player("Opp Batter 3", 4, 0),
        Player("Opp AR 1", 3, 3),
        Player("Opp AR 2", 3, 3),
        Player("Opp Bowler 1", 2, 4),
        Player("Opp Bowler 2", 1, 4),
        Player("Opp Bowler 3", 1, 3),
        Player("Opp Bowler 4", 1, 3),
    ]
    final_bowler = Player("Need Bowler", 1, 4)
    final_batter = Player("Tempting Batter", 5, 0)

    state = DraftState(
        my_captain=my_captain,
        opp_captain=opp_captain,
        my_picks=list(my_picks),
        opp_picks=list(opp_picks),
        all_players=[*my_picks, *opp_picks, final_bowler, final_batter],
        my_first=True,
        picks_per_captain=10,
        team_size=11,
    )
    return state


def _balanced_ranking_state() -> DraftState:
    my_captain = Player("Me", 3, 2)
    opp_captain = Player("Opponent", 3, 2)

    my_picks = [
        Player("My Batter 1", 5, 1),
        Player("My Batter 2", 4, 1),
        Player("My Batter 3", 4, 0),
    ]
    opp_picks = [
        Player("Opp Batter 1", 5, 1),
        Player("Opp Batter 2", 4, 1),
        Player("Opp Batter 3", 4, 0),
    ]
    pool = [
        Player("Dhruv", 3, 2),
        Player("Anil", 5, 4),
        Player("Dadun", 4, 5),
        Player("Vaibhav", 2, 2),
        Player("Other 1", 2, 4),
        Player("Other 2", 2, 4),
        Player("Other 3", 2, 3),
        Player("Other 4", 2, 3),
        Player("Other 5", 3, 1),
        Player("Other 6", 3, 1),
        Player("Other 7", 3, 2),
        Player("Other 8", 1, 4),
        Player("Other 9", 1, 3),
        Player("Other 10", 4, 2),
    ]
    return DraftState(
        my_captain=my_captain,
        opp_captain=opp_captain,
        my_picks=my_picks,
        opp_picks=opp_picks,
        all_players=pool,
        my_first=True,
        picks_per_captain=10,
        team_size=11,
    )


def test_load_players_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Duplicate player names"):
        load_players_from_text("A, 4, 2\nA, 2, 4\n")


def test_state_round_trip_preserves_history() -> None:
    state = _full_sized_state()
    state.pick_player(Player("Need Bowler", 1, 4), is_mine=True)
    payload = state.to_json()

    restored = DraftState.from_json(payload)

    assert restored.history == state.history
    assert restored.undo_last() == "Need Bowler"
    assert {player.name for player in restored.available} == {"Need Bowler", "Tempting Batter"}


def test_engine_is_deterministic_for_same_seed() -> None:
    state = _full_sized_state()
    engine_a = MCEngine(num_sims=120, noise_std=0.2, base_seed=99)
    engine_b = MCEngine(num_sims=120, noise_std=0.2, base_seed=99)

    recs_a = engine_a.evaluate_candidates(
        state.my_members,
        state.opp_members,
        state.available,
        state.my_picks_left,
        state.opp_picks_left,
    )
    recs_b = engine_b.evaluate_candidates(
        state.my_members,
        state.opp_members,
        state.available,
        state.my_picks_left,
        state.opp_picks_left,
    )

    assert [rec.player.name for rec in recs_a] == [rec.player.name for rec in recs_b]
    assert recs_a[0].win_probability == pytest.approx(recs_b[0].win_probability)
    assert recs_a[0].expected_margin == pytest.approx(recs_b[0].expected_margin)


def test_engine_prioritizes_bowling_coverage_when_one_pick_remains() -> None:
    state = _full_sized_state()
    engine = MCEngine(num_sims=150, noise_std=0.1, base_seed=7)

    recommendations = engine.evaluate_candidates(
        state.my_members,
        state.opp_members,
        state.available,
        state.my_picks_left,
        state.opp_picks_left,
    )

    assert recommendations[0].player.name == "Need Bowler"
    assert recommendations[0].win_probability >= recommendations[1].win_probability


def test_engine_balances_upside_against_threshold_bowling() -> None:
    state = _balanced_ranking_state()
    engine = MCEngine(num_sims=400, noise_std=0.55, base_seed=42)

    recommendations = engine.evaluate_candidates(
        state.my_members,
        state.opp_members,
        state.available,
        state.my_picks_left,
        state.opp_picks_left,
    )

    ranked_names = [recommendation.player.name for recommendation in recommendations]
    assert ranked_names[0] == "Dadun"
    assert ranked_names.index("Dadun") < ranked_names.index("Dhruv")
    assert ranked_names.index("Anil") < ranked_names.index("Dhruv")


def test_engine_marks_close_calls_and_refines_top_candidates() -> None:
    state = _balanced_ranking_state()
    engine = MCEngine(num_sims=400, noise_std=0.55, base_seed=42)

    recommendations = engine.evaluate_candidates(
        state.my_members,
        state.opp_members,
        state.available,
        state.my_picks_left,
        state.opp_picks_left,
    )

    assert recommendations[0].simulation_count == 400
    assert any(recommendation.simulation_count < 400 for recommendation in recommendations)
    assert sum(1 for recommendation in recommendations[:5] if recommendation.close_call) >= 2
    assert recommendations[0].win_probability_ci > 0
