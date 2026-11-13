"""Two-stage selection of review evidence; no coaching text or mistake grades."""

from typing import Literal

import chess
from pydantic import Field, model_validator

from analysis_records import (
    AnalysisRecord, CentipawnScore, Color, Digest, MateScore, PositionContext, Text,
    ValidatedModel, VersionedRecord, score_for_ordering, source_game_id,
)
from candidate_analysis import BOUND_SCORE_WARNING, analyze_position, matches_search_budget
from engine_session import StockfishSession
from pgn_input import MAX_PLIES, learner_positions, parse_pgn


MAX_MOMENTS = 15
MAX_DEEP_POSITIONS = 30
Reason = Literal["mate_swing", "evaluation_loss", "mating_line", "forcing_choice", "strong_decision", "opening_moment"]
PRIORITY = {kind: rank for rank, kind in enumerate(reversed((
    "mate_swing", "evaluation_loss", "mating_line", "forcing_choice", "strong_decision", "opening_moment",
)))}
RATIONALE = {
    "mate_swing": "Engine mate status differs between candidate and played evidence; this is not a centipawn loss.",
    "evaluation_loss": "Deep evidence retains an observed loss of at least 100 cp; search qualifiers still apply.",
    "mating_line": "The played variation reaches checkmate within four plies; this does not establish a unique-error verdict.",
    "forcing_choice": "The best engine move was played in a check/capture/evasion context, with a substantially weaker supplied alternative.",
    "strong_decision": "The best engine move was played with a substantially weaker supplied alternative; quiet-move selection is heuristic.",
    "opening_moment": "The learner castled within the first twenty full moves; this is an opening-teaching heuristic, not a book-move claim.",
}


def _usable_comparison(analysis):
    candidates, played = analysis.candidates, analysis.played
    # shortcut: T05 bound flags live in canonical notes; type them when the evidence schema advances.
    if any(BOUND_SCORE_WARNING in note for note in analysis.uncertainty):
        return False
    scores = [score_for_ordering(c.score) for c in candidates]
    actual = score_for_ordering(played.score)
    if actual is None or any(score is None for score in scores) or actual > scores[0]:
        return False
    if any(a < b for a, b in zip(scores, scores[1:])):
        return False
    if candidates[0].move_uci == played.move_uci and candidates[0].score != played.score:
        return False
    profiles = {(c.search.engine, c.search.time_limit_seconds, c.search.depth_limit, c.search.node_limit)
                for c in (*candidates, played)}
    return (len(profiles) == 1 and min(candidates[0].search.depth, played.search.depth) >= 8
            and len({c.search.config_sha256 for c in candidates}) == 1
            and (played.search.config_sha256 == candidates[0].search.config_sha256
                 or played.search.root_moves == (played.move_uci,)))


def _reasons(analysis, *, preliminary=False) -> tuple[Reason, ...]:
    board = analysis.position.board()
    reasons = []
    if 1 <= len(analysis.played.pv_uci) <= 4:
        line = board.copy(stack=True)
        for uci in analysis.played.pv_uci:
            line.push_uci(uci)
        if line.is_checkmate():
            reasons.append("mating_line")
    played_move = chess.Move.from_uci(analysis.position.played_uci)
    if board.fullmove_number <= 20 and board.is_castling(played_move):
        reasons.append("opening_moment")
    # Screening signals warrant investigation; only deep selection requires a usable comparison.
    has_scores = all(score_for_ordering(c.score) is not None for c in (analysis.candidates[0], analysis.played))
    if (preliminary and has_scores) or _usable_comparison(analysis):
        best, played = analysis.candidates[0], analysis.played
        best_loses = isinstance(best.score, MateScore) and not best.score.winning
        played_loses = isinstance(played.score, MateScore) and not played.score.winning
        best_mates = isinstance(best.score, MateScore) and best.score.winning
        played_mates = isinstance(played.score, MateScore) and played.score.winning
        if (played_loses and not best_loses) or (best_mates and not played_mates):
            reasons.append("mate_swing")
        if analysis.loss_cp is not None and analysis.loss_cp >= 100:
            reasons.append("evaluation_loss")
        if len(analysis.candidates) >= 2 and best.move_uci == played.move_uci and board.legal_moves.count() > 1:
            alternative = analysis.candidates[1]
            cp_gap = (isinstance(best.score, CentipawnScore) and isinstance(alternative.score, CentipawnScore)
                      and best.score.cp - alternative.score.cp >= 100)
            mate_gap = best_mates and not (isinstance(alternative.score, MateScore) and alternative.score.winning)
            if cp_gap or mate_gap:
                forcing = board.is_check() or board.is_capture(played_move) or board.gives_check(played_move)
                reasons.append("forcing_choice" if forcing else "strong_decision")
    return tuple(sorted(set(reasons), key=lambda kind: -PRIORITY[kind]))


def _rank(analysis, *, preliminary=False):
    reasons = _reasons(analysis, preliminary=preliminary)
    return (-PRIORITY[reasons[0]], -max(analysis.loss_cp or 0, 0), analysis.position.ply)


def _same_lesson(first, second):
    earlier, later = sorted((first, second), key=lambda a: a.position.ply)
    offset = later.position.ply - earlier.position.ply
    if not 0 < offset <= 4:
        return False
    actual_prefix = later.position.history_uci[earlier.position.ply - 1:later.position.ply - 1]
    if earlier.played.pv_uci[:offset] != actual_prefix:
        return False
    remaining = earlier.played.pv_uci[offset:]
    width = min(2, len(remaining), len(later.played.pv_uci))
    return bool(width) and remaining[:width] == later.played.pv_uci[:width]


class ScanResult(ValidatedModel):
    analysis: AnalysisRecord
    reasons: tuple[Reason, ...] = Field(strict=False)

    @model_validator(mode="after")
    def verify_reasons(self):
        if self.reasons != _reasons(self.analysis, preliminary=True):
            raise ValueError("Scan reasons disagree with the supplied evidence.")
        return self


class DeepCheck(ValidatedModel):
    preliminary_analysis_id: Digest
    analysis: AnalysisRecord


class ReviewMoment(ValidatedModel):
    analysis_id: Digest
    preliminary_analysis_id: Digest
    reasons: tuple[Reason, ...] = Field(min_length=1, strict=False)
    rationale: tuple[Text, ...] = Field(min_length=1, strict=False)


class ReviewFailure(ValidatedModel):
    position: PositionContext
    stage: Literal["scan", "deep"]
    error: Text


def _select(checks):
    selected = []
    eligible = sorted((c for c in checks if _reasons(c.analysis)), key=lambda c: _rank(c.analysis))
    for check in eligible:
        if any(_same_lesson(check.analysis, previous.analysis) for previous in selected):
            continue
        selected.append(check)
        if len(selected) == MAX_MOMENTS:
            break
    return tuple(ReviewMoment(analysis_id=c.analysis.id, preliminary_analysis_id=c.preliminary_analysis_id,
                             reasons=_reasons(c.analysis), rationale=tuple(RATIONALE[r] for r in _reasons(c.analysis))) for c in selected)


def _budget(analysis, seconds):
    return matches_search_budget(analysis, seconds)


class GameReview(VersionedRecord):
    selection_version: Literal["t09-v2"] = "t09-v2"
    game_id: Digest
    learner_color: Color
    initial_fen: Text
    total_plies: int = Field(ge=0, le=MAX_PLIES)
    scan_time_seconds: float = Field(gt=0, le=5)
    deep_time_seconds: float = Field(gt=0, le=5)
    scanned: tuple[ScanResult, ...] = Field(max_length=100, strict=False)
    deep_checks: tuple[DeepCheck, ...] = Field(max_length=MAX_DEEP_POSITIONS, strict=False)
    failures: tuple[ReviewFailure, ...] = Field(max_length=130, strict=False)
    moments: tuple[ReviewMoment, ...] = Field(max_length=MAX_MOMENTS, strict=False)
    partial: bool
    notes: tuple[Text, ...] = Field(strict=False)

    @model_validator(mode="after")
    def verify_review(self):
        if self.deep_time_seconds <= self.scan_time_seconds:
            raise ValueError("Deep search must have a larger budget than the scan.")
        root = chess.Board(self.initial_fen)
        if not root.is_valid() or root.fen() != self.initial_fen:
            raise ValueError("Invalid game-review initial FEN.")
        first = 1 if root.turn == (self.learner_color == "white") else 2
        expected = set(range(first, self.total_plies + 1, 2))
        successful = {s.analysis.position.ply: s for s in self.scanned}
        failed_scan = [f.position.ply for f in self.failures if f.stage == "scan"]
        if (len(successful) != len(self.scanned) or len(set(failed_scan)) != len(failed_scan)
                or set(successful) & set(failed_scan) or set(successful) | set(failed_scan) != expected):
            raise ValueError("Scan coverage must contain each learner decision exactly once.")
        if [s.analysis.position.ply for s in self.scanned] != sorted(successful):
            raise ValueError("Scanned positions must be chronological.")
        positions = [s.analysis.position for s in self.scanned] + [f.position for f in self.failures]
        if any(p.game_id != self.game_id or p.learner_color != self.learner_color or p.initial_fen != self.initial_fen for p in positions):
            raise ValueError("Game-review evidence belongs to another game, side or initial position.")
        if any(not _budget(s.analysis, self.scan_time_seconds) for s in self.scanned):
            raise ValueError("Preliminary search provenance disagrees with the scan budget.")
        shortlist = sorted((s for s in self.scanned if s.reasons), key=lambda s: _rank(s.analysis, preliminary=True))[:MAX_DEEP_POSITIONS]
        wanted = {s.analysis.id: s.analysis.position for s in shortlist}
        checked = set()
        for check in self.deep_checks:
            if (check.preliminary_analysis_id not in wanted or check.preliminary_analysis_id in checked
                    or check.analysis.position != wanted[check.preliminary_analysis_id]
                    or not _budget(check.analysis, self.deep_time_seconds)):
                raise ValueError("Deep evidence is not a distinct shortlisted position with the deep budget.")
            checked.add(check.preliminary_analysis_id)
        for failure in (f for f in self.failures if f.stage == "deep"):
            source = successful.get(failure.position.ply)
            if source is None or source.analysis.id not in wanted or source.analysis.id in checked or source.analysis.position != failure.position:
                raise ValueError("Deep failure does not identify a distinct shortlisted position.")
            checked.add(source.analysis.id)
        if checked != set(wanted) or self.moments != _select(self.deep_checks) or self.partial != bool(self.failures):
            raise ValueError("Review selection, deep coverage or partial status disagrees with its evidence.")
        return self

    @property
    def selected_analyses(self):
        by_id = {check.analysis.id: check.analysis for check in self.deep_checks}
        return tuple(by_id[moment.analysis_id] for moment in self.moments)

    @property
    def top_three(self):
        return self.selected_analyses[:3]


async def review_game(session: StockfishSession, pgn: str | bytes, learner_color: Color, *, scan_time=0.15, deep_time=1.5) -> GameReview:
    if (type(scan_time) not in (int, float) or type(deep_time) not in (int, float)
            or not 0 < scan_time < deep_time <= StockfishSession.MAX_SEARCH_SECONDS):
        raise ValueError("Use finite budgets with 0 < scan_time < deep_time <= 5 seconds.")
    game = parse_pgn(pgn)
    decisions = learner_positions(game, learner_color)
    color = learner_color.casefold()
    game_id = source_game_id(game)
    scans, checks, failures = [], [], []
    for decision in decisions:
        position = PositionContext.from_decision(decision, game_id=game_id)
        try:
            analysis = AnalysisRecord.model_validate(await analyze_position(session, position, time_limit=scan_time))
            if analysis.position != position or not _budget(analysis, scan_time):
                raise ValueError("Preliminary engine evidence does not match the requested position/budget.")
            scans.append(ScanResult(analysis=analysis, reasons=_reasons(analysis, preliminary=True)))
        except ValueError as error:
            failures.append(ReviewFailure(position=position, stage="scan", error=str(error)))
    shortlist = sorted((s for s in scans if s.reasons), key=lambda s: _rank(s.analysis, preliminary=True))[:MAX_DEEP_POSITIONS]
    for scan in shortlist:
        position = scan.analysis.position
        try:
            analysis = AnalysisRecord.model_validate(await analyze_position(session, position, time_limit=deep_time))
            if analysis.position != position or not _budget(analysis, deep_time):
                raise ValueError("Deep engine evidence does not match the requested position/budget.")
            checks.append(DeepCheck(preliminary_analysis_id=scan.analysis.id, analysis=analysis))
        except ValueError as error:
            failures.append(ReviewFailure(position=position, stage="deep", error=str(error)))
    moments = _select(checks)
    notes = ["Moment selection is heuristic review prioritization, not a mistake grade or generated coaching explanation.",
             "Only selected deep analysis should reach the card generator; the whole review contains later game context.",
             "Scan reasons are provisional screening signals, even with bound/contradictory/shallow scores; only usable deep comparisons supply final comparative reasons.",
             "Opening-only selection currently identifies castling; opening statistics/tablebase/model integration remains T12."]
    if not moments:
        notes.append("No moments passed the selection checks; this does not certify error-free play.")
    if failures:
        notes.append("Some position evidence failed validation; this is a partial review, not a complete clean-game assessment.")
    return GameReview(game_id=game_id, learner_color=color, initial_fen=game.board().fen(),
                      total_plies=game.end().ply() - game.ply(), scan_time_seconds=float(scan_time), deep_time_seconds=float(deep_time),
                      scanned=tuple(scans), deep_checks=tuple(checks), failures=tuple(failures), moments=moments,
                      partial=bool(failures), notes=tuple(notes))
