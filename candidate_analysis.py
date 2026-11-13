"""Engine evidence for one learner decision, without coaching or mistake grades."""

import hashlib
import json

import chess

from analysis_records import (
    AnalysisRecord, Candidate, CentipawnScore, PositionContext, SearchProvenance,
    UnavailableScore, comparison_issues,
)
from engine_session import StockfishSession


BOUND_SCORE_WARNING = "engine score is a bound, not an exact evaluation; reanalysis is required."


class IncompletePlayedLineError(ValueError):
    pass


def retry_budget(seconds):
    return min(StockfishSession.MAX_SEARCH_SECONDS, max(.5, float(seconds) * 3))


def recovery_note(seconds):
    return (f"Played-move line lacked an opponent reply; recovered after one complete-position retry "
            f"at {retry_budget(seconds):g}s per search (initial budget {seconds:g}s).")


def matches_search_budget(analysis, seconds):
    budgets = {c.search.time_limit_seconds for c in (*analysis.candidates, analysis.played)}
    return (all(c.search.depth_limit is None and c.search.node_limit is None for c in (*analysis.candidates, analysis.played))
            and (budgets == {seconds} or (budgets == {retry_budget(seconds)} and recovery_note(seconds) in analysis.uncertainty)))


def _candidates(session, board, infos, *, color, time_limit, multipv, root_moves=()):
    if not isinstance(infos, list) or not 1 <= len(infos) <= min(multipv, board.legal_moves.count()):
        raise ValueError("Stockfish returned an empty or invalid number of candidate lines.")
    profile = {"options": {**session.engine_options, "MultiPV": multipv}, "limit": {"time": float(time_limit)}}
    config_hash = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    candidates = []
    notes = []
    for rank, info in enumerate(infos, 1):
        if not isinstance(info, dict):
            raise ValueError("Stockfish returned a malformed candidate result.")
        if info.get("multipv", rank) != rank:
            raise ValueError("Stockfish MultiPV rank does not match the supplied result order.")
        if not all(key in info for key in ("depth", "nodes", "time")):
            raise ValueError("Stockfish result lacks depth/nodes/time; rerun instead of inventing search metadata.")
        search = SearchProvenance(
            engine=session.engine_name, config_sha256=config_hash, root_fen=board.fen(),
            time_limit_seconds=float(time_limit), depth=info["depth"], nodes=info["nodes"],
            elapsed_seconds=info["time"], root_moves=tuple(move.uci() for move in root_moves),
        )
        candidate = Candidate.from_engine(board, info, learner_color=color, search=search)
        candidates.append(candidate)
        if info.get("lowerbound") or info.get("upperbound"):
            notes.append(f"{candidate.san}: {BOUND_SCORE_WARNING}")
        if isinstance(candidate.score, UnavailableScore):
            notes.append(f"{candidate.san}: no score was returned; do not infer a numerical loss.")
        if search.depth < 8:
            notes.append(f"{candidate.san}: search reached depth {search.depth}; shallow evidence needs confirmation.")
    return tuple(candidates), notes


async def _analyze_once(session: StockfishSession, position: PositionContext, *, time_limit) -> AnalysisRecord:
    """Up to three candidates plus the played move, all at the original root.

    Each search gets the same time budget. MultiPV=3 and a restricted MultiPV=1
    search have different effort allocation, recorded in their profile hashes.
    """
    position = PositionContext.model_validate(position)
    board = position.board()
    infos = await session.analyse(board, time_limit=time_limit, multipv=3)
    candidates, notes = _candidates(
        session, board, infos, color=position.learner_color, time_limit=time_limit, multipv=3,
    )
    played = next((candidate for candidate in candidates if candidate.move_uci == position.played_uci), None)
    played_notes = []
    if played is None:
        roots = (chess.Move.from_uci(position.played_uci),)
        played_infos = await session.analyse(board, time_limit=time_limit, multipv=1, root_moves=roots)
        played_candidates, played_notes = _candidates(
            session, board, played_infos, color=position.learner_color, time_limit=time_limit,
            multipv=1, root_moves=roots,
        )
        played = played_candidates[0]
    after_played = board.copy(stack=True)
    after_played.push_uci(position.played_uci)
    if len(played.pv_uci) < 2 and not after_played.is_game_over():
        raise IncompletePlayedLineError("Played-move line lacks an opponent reply.")
    notes.extend(played_notes)
    notes.extend(comparison_issues(candidates, played))
    if len({candidate.search.depth for candidate in candidates}) > 1:
        notes.append("Candidate lines reached different depths; the final ranking may be incomplete or unstable.")
    best = candidates[0]
    if isinstance(best.score, CentipawnScore):
        if any(isinstance(c.score, CentipawnScore) and abs(best.score.cp - c.score.cp) < 30 for c in candidates[1:]):
            notes.append("Leading candidates are within 0.3 pawns; do not present the first move as uniquely correct.")
        if (best.move_uci != played.move_uci and isinstance(played.score, CentipawnScore)
                and abs(best.score.cp - played.score.cp) < 30):
            notes.append("The played move is within 0.3 pawns of the first candidate; this gap does not establish a mistake.")
    return AnalysisRecord(position=position, candidates=candidates, played=played, uncertainty=tuple(dict.fromkeys(notes)))


async def analyze_position(session: StockfishSession, position: PositionContext, *, time_limit=1.5) -> AnalysisRecord:
    """Retry only incomplete played PVs, once; replace all evidence at the larger budget."""
    try:
        return await _analyze_once(session, position, time_limit=time_limit)
    except IncompletePlayedLineError:
        try:
            record = await _analyze_once(session, position, time_limit=retry_budget(time_limit))
        except IncompletePlayedLineError:
            raise IncompletePlayedLineError("Played-move line still lacks an opponent reply after one longer retry; evidence remains incomplete.") from None
        return AnalysisRecord(position=record.position, candidates=record.candidates, played=record.played,
                              uncertainty=(*record.uncertainty, recovery_note(time_limit)))
