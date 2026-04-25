from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from cricket_draft_assistant.core import (
    DEFAULT_NOISE,
    DEFAULT_SIMS,
    IDEAL_BOWLERS,
    MATCH_OVERS,
    MAX_OVERS_PER_BOWLER,
    DraftState,
    MCEngine,
    Player,
    evaluate_team,
    load_players_from_text,
    stable_seed,
)

ROLE_COLORS = {
    "ALL-ROUNDER": "#9b59b6",
    "BAT-AR": "#8e44ad",
    "BOWL-AR": "#8e44ad",
    "BATSMAN": "#3498db",
    "BOWLER": "#27ae60",
    "UTILITY": "#1abc9c",
    "PART-BOWL": "#2ecc71",
    "SPECIALIST": "#95a5a6",
}


def _configure_page() -> None:
    st.set_page_config(
        page_title="Cricket Draft Assistant",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .role-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            color: white;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .pick-banner {
            text-align: center;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            color: white;
        }
        .stat-note {
            color: #64748b;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session() -> None:
    st.session_state.setdefault("phase", "setup")
    st.session_state.setdefault("num_sims", DEFAULT_SIMS)
    st.session_state.setdefault("noise", DEFAULT_NOISE)
    st.session_state.setdefault("refresh_nonce", 0)


def _clear_draft_session() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(("my_pick_sel_", "opp_pick_sel_")):
            del st.session_state[key]
    for key in ("draft_state", "recommendations", "last_fingerprint"):
        st.session_state.pop(key, None)
    st.session_state["phase"] = "setup"
    st.session_state["refresh_nonce"] = 0


def _role_badge(role: str) -> str:
    color = ROLE_COLORS.get(role, "#95a5a6")
    return f'<span class="role-badge" style="background:{color}">{role}</span>'


def _bar(value: float, maximum: int = 5) -> str:
    rounded = int(round(value))
    return "█" * rounded + "░" * (maximum - rounded)


def _player_label(player: Player) -> str:
    return f"{player.name} — Bat:{player.batting:.0f} Bowl:{player.bowling:.0f} ({player.role})"


def _recommendation_reasons(player: Player, team_members: list[Player]) -> list[str]:
    reasons: list[str] = []
    bowlers_needed = max(0, IDEAL_BOWLERS - sum(member.is_bowler for member in team_members))
    if player.batting >= 4 and player.bowling >= 3:
        reasons.append("Elite all-rounder")
    elif player.batting >= 4:
        reasons.append("Top-tier batting")
    elif player.bowling >= 4:
        reasons.append("Top-tier bowling")
    if bowlers_needed > 0 and player.is_bowler:
        reasons.append(f"Closes bowling gap ({bowlers_needed} needed)")
    if player.batting >= 3 and player.bowling >= 3:
        reasons.append("Flexible in both innings")
    if not reasons:
        reasons.append("Solid roster fit")
    return reasons


def _players_table(players: list[Player]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Name": player.name,
                "Batting": player.batting,
                "Bowling": player.bowling,
                "Role": player.role,
                "Value": round(player.raw_value, 2),
            }
            for player in sorted(players, key=lambda item: (-item.raw_value, item.name))
        ]
    )


def _sanitize_players(
    players: list[Player],
    my_name: str,
    opp_name: str,
) -> tuple[list[Player], list[str]]:
    captain_names = {my_name.strip().lower(), opp_name.strip().lower()}
    removed = [player.name for player in players if player.name.strip().lower() in captain_names]
    filtered = [player for player in players if player.name.strip().lower() not in captain_names]
    return filtered, removed


@st.cache_data(show_spinner=False)
def compute_recommendations_cached(
    state_json: str,
    num_sims: int,
    noise: float,
    refresh_nonce: int,
) -> list[dict[str, Any]]:
    state = DraftState.from_json(state_json)
    seed = stable_seed(state_json, num_sims, noise, refresh_nonce)
    engine = MCEngine(num_sims=num_sims, noise_std=noise, base_seed=seed)
    recommendations = engine.evaluate_candidates(
        my_members=state.my_members,
        opp_members=state.opp_members,
        available=state.available,
        my_picks_left=state.my_picks_left,
        opp_picks_left=state.opp_picks_left,
    )
    return [recommendation.to_dict() for recommendation in recommendations]


def _show_team(members: list[Player], label: str, captain: Player, detail: bool = True) -> None:
    bowlers = sum(member.is_bowler for member in members)
    overs = min(bowlers * MAX_OVERS_PER_BOWLER, MATCH_OVERS)
    st.markdown(f"#### {label}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Players", len(members))
    c2.metric("Bowlers", f"{bowlers}/{IDEAL_BOWLERS}")
    c3.metric("Overs", f"{overs}/{MATCH_OVERS}")
    if detail:
        st.caption(
            f"Batting power: {sum(player.batting for player in members):.0f} │ "
            f"Bowling power: {sum(player.bowling for player in members if player.is_bowler):.0f}"
        )
    for player in members:
        captain_marker = " 👑" if player == captain else ""
        st.markdown(
            f"**{player.name}**{captain_marker} &nbsp;{_role_badge(player.role)}"
            f" &ensp; Bat: `{_bar(player.batting)}` {player.batting:.0f}"
            f" &ensp; Bowl: `{_bar(player.bowling)}` {player.bowling:.0f}",
            unsafe_allow_html=True,
        )


def _show_bowling_plan(members: list[Player]) -> None:
    bowlers = sorted(
        (player for player in members if player.is_bowler),
        key=lambda player: player.bowling,
        reverse=True,
    )
    overs_left = MATCH_OVERS
    st.markdown("**Bowling plan**")
    for bowler in bowlers:
        if overs_left <= 0:
            break
        overs = min(MAX_OVERS_PER_BOWLER, overs_left)
        bar = f"{'█' * overs}{'░' * (MAX_OVERS_PER_BOWLER - overs)}"
        st.text(f"{bowler.name:15s} [{bar}] {overs} ov")
        overs_left -= overs
    if overs_left > 0:
        st.error(f"⚠ {overs_left} overs still uncovered")
    else:
        st.success(f"✓ All {MATCH_OVERS} overs covered")


def _recommendation_dataframe(recommendations: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, recommendation in enumerate(recommendations[:8], start=1):
        player = recommendation["player"]
        rows.append(
            {
                "#": index,
                "Player": player["name"],
                "Role": recommendation["role"],
                "Win %": f"{recommendation['win_probability']:.0%}",
                "± Win": f"{recommendation['win_probability_ci']:.0%}",
                "Expected Margin": f"{recommendation['expected_margin']:+.1f}",
                "Risk (σ)": f"{recommendation['margin_std']:.1f}",
                "Steal Risk": f"{recommendation['steal_risk']:.0%}",
                "Call": "Close" if recommendation["close_call"] else "Clear",
            }
        )
    return pd.DataFrame(rows)


def setup_page() -> None:
    st.title("🏏 Cricket Draft Assistant")
    st.caption("Opponent-aware Monte Carlo drafting for a 16-over run-feast")
    st.info(
        "Recommendations are ranked by win probability first, then expected margin. "
        "The simulator now models opponent strategy as a weighted mix of drafting personas."
    )

    uploaded = st.file_uploader("Upload player CSV", type=["csv"], key="player_upload")
    players: list[Player] | None = None
    if uploaded is not None:
        try:
            players = load_players_from_text(uploaded.read().decode("utf-8-sig"))
            st.success(f"Loaded {len(players)} players")
            st.dataframe(_players_table(players), use_container_width=True, hide_index=True)
        except ValueError as exc:
            st.error(str(exc))
            return

    restore_file = st.file_uploader("Restore saved draft JSON", type=["json"], key="restore_upload")
    if restore_file is not None:
        try:
            restored = restore_file.read().decode("utf-8")
            st.session_state["draft_state"] = DraftState.from_json(restored)
            st.session_state["phase"] = "draft"
            st.session_state["refresh_nonce"] = 0
            st.rerun()
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.error(f"Could not restore draft: {exc}")

    if not players:
        st.caption("Need a CSV to start a new draft.")
        return

    with st.form("setup_form"):
        left, right = st.columns(2)
        with left:
            my_name = st.text_input("Your name", value="Me")
            my_bat = st.slider("Your batting", min_value=0.0, max_value=5.0, value=3.0, step=0.5)
            my_bowl = st.slider("Your bowling", min_value=0.0, max_value=5.0, value=2.0, step=0.5)
        with right:
            opp_name = st.text_input("Opponent name", value="Opponent")
            opp_bat = st.slider(
                "Opponent batting",
                min_value=0.0,
                max_value=5.0,
                value=3.0,
                step=0.5,
            )
            opp_bowl = st.slider(
                "Opponent bowling",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            first_pick = st.radio(
                "Who picks first?",
                options=[my_name or "Me", opp_name or "Opponent"],
            )
        with c2:
            num_sims = st.select_slider(
                "Simulations per candidate",
                options=[150, 250, 400, 600, 800, 1000],
                value=DEFAULT_SIMS,
            )
        with c3:
            noise = st.slider(
                "Rating uncertainty (σ)",
                min_value=0.0,
                max_value=1.5,
                value=float(DEFAULT_NOISE),
                step=0.05,
            )

        submitted = st.form_submit_button("Start draft", type="primary", use_container_width=True)

    if not submitted:
        return

    filtered_players, removed = _sanitize_players(players, my_name, opp_name)
    if removed:
        st.warning("Removed captains from the draft pool: " + ", ".join(removed))
    if len(filtered_players) < 2:
        st.error("Need at least two non-captain players in the pool")
        return
    if len(filtered_players) % 2 != 0:
        st.error(
            "The player pool excluding captains must be even so both captains "
            "get the same number of picks"
        )
        return

    picks_per_captain = len(filtered_players) // 2
    team_size = picks_per_captain + 1
    st.session_state["draft_state"] = DraftState(
        my_captain=Player(name=my_name.strip() or "Me", batting=my_bat, bowling=my_bowl),
        opp_captain=Player(name=opp_name.strip() or "Opponent", batting=opp_bat, bowling=opp_bowl),
        all_players=filtered_players,
        my_first=first_pick == (my_name or "Me"),
        picks_per_captain=picks_per_captain,
        team_size=team_size,
    )
    st.session_state["num_sims"] = num_sims
    st.session_state["noise"] = noise
    st.session_state["refresh_nonce"] = 0
    st.session_state["phase"] = "draft"
    st.rerun()


def draft_page() -> None:
    state: DraftState = st.session_state["draft_state"]
    if state.draft_complete:
        st.session_state["phase"] = "complete"
        st.rerun()
        return

    st.title("🏏 Cricket Draft Assistant")
    pick_number = state.total_picks_done + 1
    total_picks = len(state.all_players)
    st.progress(
        state.total_picks_done / max(total_picks, 1),
        text=f"Pick {pick_number} of {total_picks}",
    )

    my_turn = state.is_my_turn
    banner_color = "#16a34a" if my_turn else "#dc2626"
    banner_text = "YOUR PICK" if my_turn else "OPPONENT'S PICK"
    picks_left = state.my_picks_left if my_turn else state.opp_picks_left
    st.markdown(
        f'<div class="pick-banner" style="background:{banner_color}">'
        f'<h2 style="margin:0;">{banner_text} · #{pick_number}</h2>'
        f"<div>{picks_left} picks remaining for this side</div></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Draft controls")
        st.caption("These settings invalidate cached recommendations immediately.")
        new_sims = st.select_slider(
            "Simulations",
            options=[150, 250, 400, 600, 800, 1000],
            value=int(st.session_state["num_sims"]),
            key="sidebar_num_sims",
        )
        new_noise = st.slider(
            "Rating uncertainty (σ)",
            min_value=0.0,
            max_value=1.5,
            value=float(st.session_state["noise"]),
            step=0.05,
            key="sidebar_noise",
        )
        if new_sims != st.session_state["num_sims"] or new_noise != st.session_state["noise"]:
            st.session_state["num_sims"] = new_sims
            st.session_state["noise"] = new_noise
            st.session_state["refresh_nonce"] += 1

        if st.button("↩ Undo last pick", use_container_width=True):
            undone = state.undo_last()
            if undone:
                st.session_state["refresh_nonce"] += 1
                st.toast(f"Undid pick: {undone}")
                st.rerun()
            st.warning("Nothing to undo")

        st.download_button(
            "💾 Download save file",
            data=state.to_json(),
            file_name="draft-save.json",
            mime="application/json",
            use_container_width=True,
        )
        if st.button("🔄 Start over", use_container_width=True):
            _clear_draft_session()
            st.rerun()

        st.markdown("---")
        _show_bowling_plan(state.my_members)

    team_left, team_right = st.columns(2)
    with team_left:
        with st.container(border=True):
            _show_team(state.my_members, f"🟢 {state.my_captain.name}'s team", state.my_captain)
    with team_right:
        with st.container(border=True):
            _show_team(
                state.opp_members,
                f"🔴 {state.opp_captain.name}'s team",
                state.opp_captain,
                detail=False,
            )

    recommendations: list[dict[str, Any]] = []
    if my_turn and state.my_picks_left > 0:
        with st.status("Running opponent-aware Monte Carlo simulation…", expanded=False):
            recommendations = compute_recommendations_cached(
                state.to_json(),
                int(st.session_state["num_sims"]),
                float(st.session_state["noise"]),
                int(st.session_state["refresh_nonce"]),
            )
        st.markdown("### 🤖 Recommended picks")
        close_group = [rec for rec in recommendations if rec["close_call"]]
        if len(close_group) > 1:
            names = ", ".join(rec["player"]["name"] for rec in close_group[:4])
            st.info(
                "The top options are statistically close after simulation uncertainty. "
                f"Treat {names} as a cluster and use steal risk plus team fit as the tiebreaker."
            )
        st.dataframe(
            _recommendation_dataframe(recommendations),
            use_container_width=True,
            hide_index=True,
        )
        for index, recommendation in enumerate(recommendations[:5], start=1):
            player = Player(**recommendation["player"])
            reasons = _recommendation_reasons(player, state.my_members)
            badge = "⭐ " if index == 1 else ""
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 2, 2])
                with c1:
                    st.markdown(
                        (
                            f"**{badge}#{index} {player.name}** &nbsp;"
                            f"{_role_badge(recommendation['role'])}"
                        ),
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Bat `{_bar(player.batting)}` {player.batting:.0f} │ "
                        f"Bowl `{_bar(player.bowling)}` {player.bowling:.0f}"
                    )
                with c2:
                    st.metric("Win probability", f"{recommendation['win_probability']:.0%}")
                    st.caption(
                        f"Expected margin {recommendation['expected_margin']:+.1f}"
                        f" · ± win {recommendation['win_probability_ci']:.0%}"
                    )
                with c3:
                    st.metric("Steal risk", f"{recommendation['steal_risk']:.0%}")
                    st.caption(
                        f"Margin σ {recommendation['margin_std']:.1f}"
                        f" · {recommendation['simulation_count']} sims"
                    )
                if recommendation["close_call"]:
                    st.caption(
                        "Close call: this option is within the model's simulated win-rate "
                        "uncertainty band of the current best pick."
                    )
                st.caption(" │ ".join(reasons))

    available = sorted(state.available, key=lambda player: (-player.raw_value, player.name))
    with st.expander(f"Available players ({len(available)})", expanded=not my_turn):
        st.dataframe(_players_table(available), use_container_width=True, hide_index=True)

    st.markdown("---")
    fingerprint = state.fingerprint()
    options = [_player_label(player) for player in available]
    lookup = {_player_label(player): player for player in available}

    if my_turn:
        st.markdown("### 🎯 Make your pick")
        default_index = 0
        if recommendations:
            best_label = _player_label(Player(**recommendations[0]["player"]))
            if best_label in options:
                default_index = options.index(best_label)
        label = st.selectbox(
            "Choose a player (type to search)",
            options=options,
            index=default_index,
            key=f"my_pick_sel_{fingerprint}",
        )
        selected = lookup[label]
        if recommendations and selected.name != recommendations[0]["player"]["name"]:
            alternative = next(
                (rec for rec in recommendations if rec["player"]["name"] == selected.name),
                None,
            )
            best = recommendations[0]
            gap = best["expected_margin"] - (alternative["expected_margin"] if alternative else 0.0)
            st.warning(
                f"AI prefers {best['player']['name']} by about {gap:+.1f} expected margin. "
                f"Override is still allowed."
            )
        if st.button(
            f"✅ Add {selected.name} to my team",
            type="primary",
            use_container_width=True,
        ):
            state.pick_player(selected, is_mine=True)
            st.session_state["refresh_nonce"] += 1
            st.toast(f"Added {selected.name} to your team")
            st.rerun()
    else:
        st.markdown("### 🔴 Record opponent pick")
        label = st.selectbox(
            "Which player did the opponent pick?",
            options=options,
            key=f"opp_pick_sel_{fingerprint}",
        )
        selected = lookup[label]
        if st.button(
            f"📝 Record {selected.name} for opponent",
            type="primary",
            use_container_width=True,
        ):
            state.pick_player(selected, is_mine=False)
            st.session_state["refresh_nonce"] += 1
            st.toast(f"Recorded {selected.name} for opponent")
            st.rerun()


def results_page() -> None:
    state: DraftState = st.session_state["draft_state"]
    my_score = evaluate_team(state.my_members)
    opp_score = evaluate_team(state.opp_members)
    diff = my_score - opp_score

    st.title("🏆 Draft complete")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{state.my_captain.name} score", f"{my_score:.1f}")
    c2.metric("Margin", f"{diff:+.1f}")
    c3.metric(f"{state.opp_captain.name} score", f"{opp_score:.1f}")

    if diff > 5:
        st.success(f"Your squad projects clearly stronger: {diff:+.1f}")
    elif diff > 0:
        st.success(f"You finish with a small edge: {diff:+.1f}")
    elif diff > -5:
        st.warning(f"This draft looks pretty even: {diff:+.1f}")
    else:
        st.error(f"Opponent finishes ahead on projected strength: {diff:+.1f}")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            _show_team(state.my_members, f"🟢 {state.my_captain.name}'s team", state.my_captain)
            st.markdown("---")
            _show_bowling_plan(state.my_members)
    with right:
        with st.container(border=True):
            _show_team(state.opp_members, f"🔴 {state.opp_captain.name}'s team", state.opp_captain)
            st.markdown("---")
            _show_bowling_plan(state.opp_members)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "💾 Download final draft",
            data=state.to_json(),
            file_name="draft-results.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        if st.button("🔄 New draft", use_container_width=True):
            _clear_draft_session()
            st.rerun()


def main() -> None:
    _configure_page()
    _init_session()
    phase = st.session_state["phase"]
    if phase == "setup":
        setup_page()
    elif phase == "draft":
        draft_page()
    else:
        results_page()


if __name__ == "__main__":
    main()
