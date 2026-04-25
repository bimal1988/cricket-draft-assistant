from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any

MATCH_OVERS = 16
MAX_OVERS_PER_BOWLER = 2
MIN_BOWLERS = 7
IDEAL_BOWLERS = 8
TEAM_SIZE = 11
PICKS_PER_CAPTAIN = TEAM_SIZE - 1

BATTING_WEIGHT = 1.4
BOWLING_WEIGHT = 1.0
ALLROUNDER_BONUS = 2.0
BOWLING_THRESHOLD = 2

DEFAULT_SIMS = 400
DEFAULT_NOISE = 0.55


@dataclass(frozen=True)
class DraftPersona:
    name: str
    batting_bias: float
    bowling_bias: float
    allrounder_bias: float
    scarcity_bias: float


PERSONAS: tuple[tuple[DraftPersona, float], ...] = (
    (
        DraftPersona(
            name="balanced",
            batting_bias=1.0,
            bowling_bias=1.0,
            allrounder_bias=1.0,
            scarcity_bias=1.0,
        ),
        0.50,
    ),
    (
        DraftPersona(
            name="run-feast aggressor",
            batting_bias=1.18,
            bowling_bias=0.92,
            allrounder_bias=1.05,
            scarcity_bias=0.85,
        ),
        0.25,
    ),
    (
        DraftPersona(
            name="bowling coverage prioritizer",
            batting_bias=0.94,
            bowling_bias=1.15,
            allrounder_bias=1.10,
            scarcity_bias=1.35,
        ),
        0.25,
    ),
)


@dataclass(frozen=True)
class Player:
    name: str
    batting: float
    bowling: float

    @property
    def role(self) -> str:
        if self.batting >= 3 and self.bowling >= 3:
            return "ALL-ROUNDER"
        if self.batting >= 3 and self.bowling >= 2:
            return "BAT-AR"
        if self.bowling >= 3 and self.batting >= 2:
            return "BOWL-AR"
        if self.batting >= 3:
            return "BATSMAN"
        if self.bowling >= 3:
            return "BOWLER"
        if self.batting >= 2 and self.bowling >= 2:
            return "UTILITY"
        if self.bowling >= 2:
            return "PART-BOWL"
        return "SPECIALIST"

    @property
    def is_bowler(self) -> bool:
        return self.bowling >= BOWLING_THRESHOLD

    @property
    def reliable_bowler(self) -> bool:
        return self.bowling >= 3

    @property
    def bowling_capacity(self) -> float:
        if self.bowling < 2:
            return 0.0
        if self.bowling < 3:
            return 1.0
        if self.bowling < 4:
            return 1.5
        return float(MAX_OVERS_PER_BOWLER)

    @property
    def raw_value(self) -> float:
        return BATTING_WEIGHT * self.batting + BOWLING_WEIGHT * self.bowling

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "batting": self.batting, "bowling": self.bowling}


@dataclass(frozen=True)
class Recommendation:
    player: Player
    win_probability: float
    expected_margin: float
    margin_std: float
    expected_my_score: float
    expected_opp_score: float
    steal_risk: float
    ranking_score: float
    win_probability_ci: float
    simulation_count: int
    close_call: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player.to_dict(),
            "role": self.player.role,
            "win_probability": self.win_probability,
            "expected_margin": self.expected_margin,
            "margin_std": self.margin_std,
            "expected_my_score": self.expected_my_score,
            "expected_opp_score": self.expected_opp_score,
            "steal_risk": self.steal_risk,
            "ranking_score": self.ranking_score,
            "win_probability_ci": self.win_probability_ci,
            "simulation_count": self.simulation_count,
            "close_call": self.close_call,
        }


def recommendation_ranking_score(
    *,
    win_probability: float,
    expected_margin: float,
    margin_std: float,
    steal_risk: float,
) -> float:
    urgency_bonus = min(0.08, max(0.0, expected_margin) * steal_risk * 0.004)
    return (
        win_probability
        + 0.010 * expected_margin
        - 0.0025 * margin_std
        + urgency_bonus
    )


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"|")
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


@dataclass
class DraftState:
    my_captain: Player
    opp_captain: Player
    my_picks: list[Player] = field(default_factory=list)
    opp_picks: list[Player] = field(default_factory=list)
    all_players: list[Player] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    my_first: bool = True
    picks_per_captain: int = PICKS_PER_CAPTAIN
    team_size: int = TEAM_SIZE

    @property
    def my_members(self) -> list[Player]:
        return [self.my_captain, *self.my_picks]

    @property
    def opp_members(self) -> list[Player]:
        return [self.opp_captain, *self.opp_picks]

    @property
    def available(self) -> list[Player]:
        taken = {player.name for player in self.my_picks + self.opp_picks}
        return [player for player in self.all_players if player.name not in taken]

    @property
    def my_picks_left(self) -> int:
        return self.picks_per_captain - len(self.my_picks)

    @property
    def opp_picks_left(self) -> int:
        return self.picks_per_captain - len(self.opp_picks)

    @property
    def total_picks_done(self) -> int:
        return len(self.my_picks) + len(self.opp_picks)

    @property
    def is_my_turn(self) -> bool:
        if self.my_first:
            return self.total_picks_done % 2 == 0
        return self.total_picks_done % 2 == 1

    @property
    def draft_complete(self) -> bool:
        return not self.available or (self.my_picks_left <= 0 and self.opp_picks_left <= 0)

    def player_lookup(self) -> dict[str, Player]:
        return {player.name: player for player in self.all_players}

    def validate_pick(self, player: Player, is_mine: bool) -> None:
        if player.name not in {candidate.name for candidate in self.available}:
            raise ValueError(f"{player.name} is no longer available")
        if is_mine and self.my_picks_left <= 0:
            raise ValueError("Your team is already full")
        if not is_mine and self.opp_picks_left <= 0:
            raise ValueError("Opponent team is already full")

    def pick_player(self, player: Player, is_mine: bool) -> None:
        self.validate_pick(player, is_mine)
        self.history.append(
            {
                "player": player.name,
                "is_mine": is_mine,
                "my_picks": [pick.name for pick in self.my_picks],
                "opp_picks": [pick.name for pick in self.opp_picks],
            }
        )
        if is_mine:
            self.my_picks.append(player)
        else:
            self.opp_picks.append(player)

    def undo_last(self) -> str | None:
        if not self.history:
            return None
        snapshot = self.history.pop()
        lookup = self.player_lookup()
        self.my_picks = [lookup[name] for name in snapshot["my_picks"]]
        self.opp_picks = [lookup[name] for name in snapshot["opp_picks"]]
        return str(snapshot["player"])

    def to_payload(self) -> dict[str, Any]:
        return {
            "my_captain": self.my_captain.to_dict(),
            "opp_captain": self.opp_captain.to_dict(),
            "my_picks": [player.name for player in self.my_picks],
            "opp_picks": [player.name for player in self.opp_picks],
            "all_players": [player.to_dict() for player in self.all_players],
            "history": self.history,
            "my_first": self.my_first,
            "picks_per_captain": self.picks_per_captain,
            "team_size": self.team_size,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), indent=2, sort_keys=True)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:12]

    @classmethod
    def from_json(cls, payload: str) -> DraftState:
        data = json.loads(payload)
        all_players = [Player(**player_data) for player_data in data["all_players"]]
        lookup = {player.name: player for player in all_players}
        state = cls(
            my_captain=Player(**data["my_captain"]),
            opp_captain=Player(**data["opp_captain"]),
            all_players=all_players,
            history=data.get("history", []),
            my_first=bool(data["my_first"]),
            picks_per_captain=int(data.get("picks_per_captain", PICKS_PER_CAPTAIN)),
            team_size=int(data.get("team_size", TEAM_SIZE)),
        )
        state.my_picks = [lookup[name] for name in data["my_picks"]]
        state.opp_picks = [lookup[name] for name in data["opp_picks"]]
        return state


def evaluate_team(members: list[Player]) -> float:
    bat_sorted = sorted((player.batting for player in members), reverse=True)
    batting_value = 0.0
    for index, batting in enumerate(bat_sorted):
        batting_value += BATTING_WEIGHT * batting * max(0.5, 1.0 - index * 0.045)

    bowlers = sorted(
        (player for player in members if player.is_bowler),
        key=lambda player: player.bowling,
        reverse=True,
    )
    bowling_value = 0.0
    overs_left = MATCH_OVERS
    for bowler in bowlers:
        if overs_left <= 0:
            break
        overs = min(bowler.bowling_capacity, overs_left)
        bowling_value += BOWLING_WEIGHT * bowler.bowling * overs
        overs_left -= overs

    bowler_count = len(bowlers)
    total_capacity = sum(bowler.bowling_capacity for bowler in bowlers)
    reliable_bowler_count = sum(1 for bowler in bowlers if bowler.reliable_bowler)
    coverage = min(total_capacity, MATCH_OVERS) * 0.75
    coverage -= max(0.0, MATCH_OVERS - total_capacity) * 3.5
    coverage += min(reliable_bowler_count, IDEAL_BOWLERS) * 0.5
    coverage -= max(0, MIN_BOWLERS - bowler_count) * 1.5

    allrounders = sum(1 for player in members if player.batting >= 3 and player.bowling >= 3)
    batting_depth = min(sum(1 for player in members if player.batting >= 3), 7) * 0.5
    bowling_variety = min(sum(1 for player in members if player.bowling >= 3), 5) * 0.4

    return (
        batting_value
        + bowling_value
        + coverage
        + allrounders * ALLROUNDER_BONUS
        + batting_depth
        + bowling_variety
    )


def load_players_from_text(content: str) -> list[Player]:
    players: list[Player] = []
    for row in csv.reader(io.StringIO(content)):
        if len(row) < 3:
            continue
        name = row[0].strip()
        try:
            batting = float(row[1].strip())
            bowling = float(row[2].strip())
        except ValueError:
            continue
        if name:
            players.append(Player(name=name, batting=batting, bowling=bowling))

    duplicates = sorted(
        {
            player.name
            for player in players
            if sum(other.name == player.name for other in players) > 1
        }
    )
    if duplicates:
        raise ValueError(f"Duplicate player names found: {', '.join(duplicates)}")

    if not players:
        raise ValueError("No valid player rows found")

    return players


def load_players(path: str | Path) -> list[Player]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return load_players_from_text(handle.read())


class MCEngine:
    def __init__(
        self,
        num_sims: int = DEFAULT_SIMS,
        noise_std: float = DEFAULT_NOISE,
        base_seed: int | None = None,
    ) -> None:
        self.num_sims = num_sims
        self.noise_std = noise_std
        self.base_seed = base_seed if base_seed is not None else stable_seed(num_sims, noise_std)

    def _rng_for(self, *parts: Any) -> random.Random:
        return random.Random(stable_seed(self.base_seed, *parts))

    def _noisy(self, rating: float, rng: random.Random) -> float:
        return max(0.0, min(5.0, rating + rng.gauss(0, self.noise_std)))

    def _noisy_player(self, player: Player, rng: random.Random) -> Player:
        return Player(
            name=player.name,
            batting=self._noisy(player.batting, rng),
            bowling=self._noisy(player.bowling, rng),
        )

    def _sample_persona(self, rng: random.Random) -> DraftPersona:
        roll = rng.random()
        cumulative = 0.0
        for persona, weight in PERSONAS:
            cumulative += weight
            if roll <= cumulative:
                return persona
        return PERSONAS[-1][0]

    def _replacement_gap(self, player: Player, pool: list[Player]) -> float:
        batting_alts = sorted(
            (
                candidate.batting
                for candidate in pool
                if candidate != player and candidate.batting >= 3
            ),
            reverse=True,
        )
        bowling_alts = sorted(
            (
                candidate.bowling
                for candidate in pool
                if candidate != player and candidate.is_bowler
            ),
            reverse=True,
        )

        bat_alt = batting_alts[min(3, len(batting_alts) - 1)] if batting_alts else 0.0
        bowl_alt = bowling_alts[min(2, len(bowling_alts) - 1)] if bowling_alts else 0.0

        return (
            max(0.0, player.batting - bat_alt) * 0.45
            + max(0.0, player.bowling - bowl_alt) * (0.55 + 0.20 * player.bowling_capacity)
            + min(player.batting, player.bowling) * 0.12
        )

    def _candidate_summary(
        self,
        *,
        candidate: Player,
        margins: list[float],
        my_scores: list[float],
        opp_scores: list[float],
        wins: float,
        sim_count: int,
        steal_risk: float,
    ) -> Recommendation:
        win_probability = wins / sim_count
        expected_margin = mean(margins)
        margin_std = stdev(margins) if len(margins) > 1 else 0.0
        return Recommendation(
            player=candidate,
            win_probability=win_probability,
            expected_margin=expected_margin,
            margin_std=margin_std,
            expected_my_score=mean(my_scores),
            expected_opp_score=mean(opp_scores),
            steal_risk=steal_risk,
            ranking_score=recommendation_ranking_score(
                win_probability=win_probability,
                expected_margin=expected_margin,
                margin_std=margin_std,
                steal_risk=steal_risk,
            ),
            win_probability_ci=1.96 * sqrt(win_probability * (1 - win_probability) / sim_count),
            simulation_count=sim_count,
        )

    def _pick_value(
        self,
        player: Player,
        team_members: list[Player],
        available: list[Player],
        picks_left: int,
        persona: DraftPersona,
        rng: random.Random,
    ) -> float:
        team_bowlers = sum(1 for member in team_members if member.is_bowler)
        team_bowling_capacity = sum(member.bowling_capacity for member in team_members)
        reliable_bowlers = sum(1 for member in team_members if member.reliable_bowler)
        team_top_batters = sum(1 for member in team_members if member.batting >= 3)
        pool_bowling_capacity = sum(candidate.bowling_capacity for candidate in available)
        pool_top_batters = sum(1 for candidate in available if candidate.batting >= 3)

        bowlers_needed = max(0, IDEAL_BOWLERS - team_bowlers)
        overs_needed = max(0.0, MATCH_OVERS - team_bowling_capacity)
        reliable_gap = max(0, MIN_BOWLERS - reliable_bowlers)
        batters_needed = max(0, 6 - team_top_batters)
        bowl_pressure = overs_needed / max(1.0, picks_left * MAX_OVERS_PER_BOWLER)
        batting_pressure = batters_needed / max(1, picks_left)

        value = persona.batting_bias * BATTING_WEIGHT * player.batting
        value += persona.bowling_bias * BOWLING_WEIGHT * player.bowling

        if player.batting >= 3 and player.bowling >= 3:
            value += persona.allrounder_bias * ALLROUNDER_BONUS

        if player.bowling_capacity > 0:
            value += persona.scarcity_bias * bowl_pressure * (
                0.8 + player.bowling * player.bowling_capacity
            )
            value += min(player.bowling_capacity, overs_needed) * 1.2
            if player.reliable_bowler:
                value += reliable_gap * 0.8
            if pool_bowling_capacity <= overs_needed + 2.0:
                value += 2.0 * persona.scarcity_bias * player.bowling_capacity / 2.0
        else:
            value -= persona.scarcity_bias * max(0.0, bowl_pressure - 1.0) * 4.5

        if player.batting >= 3:
            value += persona.batting_bias * batting_pressure * (0.4 + 0.4 * player.batting)
            if pool_top_batters <= batters_needed + 2:
                value += 1.2 * persona.batting_bias
        else:
            value -= persona.batting_bias * max(0.0, batting_pressure - 1.0) * 1.5

        if bowlers_needed <= 2 and player.batting >= 4 and player.bowling >= 3:
            value += 1.2

        value += self._replacement_gap(player, available)
        value += rng.gauss(0, 0.12)
        return value

    def _sim_pick_index(
        self,
        pool: list[Player],
        team_members: list[Player],
        picks_left: int,
        persona: DraftPersona,
        rng: random.Random,
    ) -> int:
        scored = [
            self._pick_value(player, team_members, pool, picks_left, persona, rng)
            for player in pool
        ]
        return max(range(len(pool)), key=scored.__getitem__)

    def _simulate_remaining(
        self,
        my_members: list[Player],
        opp_members: list[Player],
        pool: list[Player],
        my_picks_left: int,
        opp_picks_left: int,
        my_turn: bool,
        rng: random.Random,
    ) -> tuple[list[Player], list[Player]]:
        my_team = list(my_members)
        opp_team = list(opp_members)
        available = sorted(pool, key=lambda player: (player.name, -player.raw_value))
        noisy_pool = {player.name: self._noisy_player(player, rng) for player in available}
        my_persona = PERSONAS[0][0]
        opp_persona = self._sample_persona(rng)

        current_my_turn = my_turn
        while available and (my_picks_left > 0 or opp_picks_left > 0):
            if current_my_turn and my_picks_left > 0:
                team_members = my_team
                picks_left = my_picks_left
                persona = my_persona
            elif not current_my_turn and opp_picks_left > 0:
                team_members = opp_team
                picks_left = opp_picks_left
                persona = opp_persona
            else:
                current_my_turn = not current_my_turn
                continue

            noisy_available = [noisy_pool[player.name] for player in available]
            picked_index = self._sim_pick_index(
                noisy_available,
                team_members,
                picks_left,
                persona,
                rng,
            )
            picked = available.pop(picked_index)
            del noisy_pool[picked.name]

            if current_my_turn:
                my_team.append(picked)
                my_picks_left -= 1
            else:
                opp_team.append(picked)
                opp_picks_left -= 1
            current_my_turn = not current_my_turn

        return my_team, opp_team

    def _calc_steal_risk(
        self,
        candidate: Player,
        opp_members: list[Player],
        available: list[Player],
        opp_picks_left: int,
        context_key: Any,
    ) -> float:
        if opp_picks_left <= 0:
            return 0.0

        steals = 0
        samples = min(250, max(100, self.num_sims // 2))
        for sim_index in range(samples):
            rng = self._rng_for("steal", context_key, candidate.name, sim_index)
            noisy_pool = {player.name: self._noisy_player(player, rng) for player in available}
            noisy_available = [noisy_pool[player.name] for player in available]
            persona = self._sample_persona(rng)
            picked_index = self._sim_pick_index(
                noisy_available,
                opp_members,
                opp_picks_left,
                persona,
                rng,
            )
            if available[picked_index].name == candidate.name:
                steals += 1
        return steals / samples

    def _evaluate_candidate(
        self,
        *,
        candidate: Player,
        my_members: list[Player],
        opp_members: list[Player],
        ordered_available: list[Player],
        my_picks_left: int,
        opp_picks_left: int,
        sim_count: int,
        context_key: Any,
        stage_tag: str,
        steal_risk: float | None = None,
    ) -> Recommendation:
        margins: list[float] = []
        my_scores: list[float] = []
        opp_scores: list[float] = []
        wins = 0.0

        for sim_index in range(sim_count):
            rng = self._rng_for(stage_tag, context_key, candidate.name, sim_index)
            final_my, final_opp = self._simulate_remaining(
                my_members=[*my_members, candidate],
                opp_members=list(opp_members),
                pool=[player for player in ordered_available if player != candidate],
                my_picks_left=my_picks_left - 1,
                opp_picks_left=opp_picks_left,
                my_turn=False,
                rng=rng,
            )
            my_score = evaluate_team(final_my)
            opp_score = evaluate_team(final_opp)
            margin = my_score - opp_score

            my_scores.append(my_score)
            opp_scores.append(opp_score)
            margins.append(margin)
            if margin > 0:
                wins += 1.0
            elif margin == 0:
                wins += 0.5

        resolved_steal_risk = steal_risk
        if resolved_steal_risk is None:
            resolved_steal_risk = self._calc_steal_risk(
                candidate,
                opp_members,
                ordered_available,
                opp_picks_left,
                context_key,
            )

        return self._candidate_summary(
            candidate=candidate,
            margins=margins,
            my_scores=my_scores,
            opp_scores=opp_scores,
            wins=wins,
            sim_count=sim_count,
            steal_risk=resolved_steal_risk,
        )

    def evaluate_candidates(
        self,
        my_members: list[Player],
        opp_members: list[Player],
        available: list[Player],
        my_picks_left: int,
        opp_picks_left: int,
    ) -> list[Recommendation]:
        ordered_available = sorted(available, key=lambda player: (-player.raw_value, player.name))
        context_key = (
            tuple(sorted(player.name for player in my_members)),
            tuple(sorted(player.name for player in opp_members)),
            tuple(sorted(player.name for player in ordered_available)),
            my_picks_left,
            opp_picks_left,
        )

        screen_sims = min(self.num_sims, max(120, self.num_sims // 3))
        screen_recommendations = [
            self._evaluate_candidate(
                candidate=candidate,
                my_members=my_members,
                opp_members=opp_members,
                ordered_available=ordered_available,
                my_picks_left=my_picks_left,
                opp_picks_left=opp_picks_left,
                sim_count=screen_sims,
                context_key=context_key,
                stage_tag="screen",
            )
            for candidate in ordered_available
        ]

        screen_recommendations.sort(
            key=lambda recommendation: (
                recommendation.ranking_score,
                recommendation.expected_margin,
                recommendation.win_probability,
            ),
            reverse=True,
        )

        shortlist_size = min(max(5, len(ordered_available) // 3), len(ordered_available))
        shortlist_names = {
            recommendation.player.name for recommendation in screen_recommendations[:shortlist_size]
        }
        best_screen = screen_recommendations[0]
        for recommendation in screen_recommendations:
            close_gap = best_screen.win_probability - recommendation.win_probability
            uncertainty_band = (
                best_screen.win_probability_ci + recommendation.win_probability_ci
            )
            if close_gap <= uncertainty_band + 0.01:
                shortlist_names.add(recommendation.player.name)

        full_recommendations: list[Recommendation] = []
        for recommendation in screen_recommendations:
            if recommendation.player.name in shortlist_names and self.num_sims > screen_sims:
                full_recommendations.append(
                    self._evaluate_candidate(
                        candidate=recommendation.player,
                        my_members=my_members,
                        opp_members=opp_members,
                        ordered_available=ordered_available,
                        my_picks_left=my_picks_left,
                        opp_picks_left=opp_picks_left,
                        sim_count=self.num_sims,
                        context_key=context_key,
                        stage_tag="refine",
                        steal_risk=recommendation.steal_risk,
                    )
                )
            else:
                full_recommendations.append(recommendation)

        best_win = max(recommendation.win_probability for recommendation in full_recommendations)
        best_ci = max(
            recommendation.win_probability_ci
            for recommendation in full_recommendations
            if recommendation.win_probability == best_win
        )
        recommendations = []
        for recommendation in full_recommendations:
            close_call = (
                best_win - recommendation.win_probability
                <= best_ci + recommendation.win_probability_ci
            )
            recommendations.append(
                Recommendation(
                    player=recommendation.player,
                    win_probability=recommendation.win_probability,
                    expected_margin=recommendation.expected_margin,
                    margin_std=recommendation.margin_std,
                    expected_my_score=recommendation.expected_my_score,
                    expected_opp_score=recommendation.expected_opp_score,
                    steal_risk=recommendation.steal_risk,
                    ranking_score=recommendation.ranking_score,
                    win_probability_ci=recommendation.win_probability_ci,
                    simulation_count=recommendation.simulation_count,
                    close_call=close_call,
                )
            )

        recommendations.sort(
            key=lambda recommendation: (
                recommendation.close_call,
                (
                    recommendation.ranking_score
                    if recommendation.close_call
                    else recommendation.win_probability
                ),
                recommendation.expected_margin,
                recommendation.expected_my_score,
                recommendation.steal_risk,
            ),
            reverse=True,
        )
        return recommendations
