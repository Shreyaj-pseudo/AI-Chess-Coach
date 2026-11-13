"""Versioned JSON records for engine evidence and separately reviewed explanations."""

import hashlib
import json
from typing import Annotated, Literal

import chess
import chess.engine
import chess.pgn
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from pgn_input import Decision, MAX_PLIES


Color = Literal["white", "black"]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
UCI = Annotated[str, StringConstraints(pattern=r"^[a-h][1-8][a-h][1-8][qrbn]?$")]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class ValidatedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, validate_default=True,
        revalidate_instances="always", allow_inf_nan=False,
    )


class VersionedRecord(ValidatedModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    id: str = ""

    @model_validator(mode="after")
    def verify_content_id(self):
        expected = _digest({"type": type(self).__name__, "data": self.model_dump(mode="json", exclude={"id"})})
        if self.id and self.id != expected:
            raise ValueError("Record ID does not match its validated content")
        object.__setattr__(self, "id", expected)
        return self


class CentipawnScore(ValidatedModel):
    kind: Literal["cp"] = "cp"
    pov: Color
    cp: int


class MateScore(ValidatedModel):
    kind: Literal["mate"] = "mate"
    pov: Color
    moves: int = Field(ge=0)
    winning: bool


class UnavailableScore(ValidatedModel):
    kind: Literal["unavailable"] = "unavailable"
    pov: Color
    reason: Text


Score = Annotated[CentipawnScore | MateScore | UnavailableScore, Field(discriminator="kind")]


def normalize_score(score: chess.engine.PovScore | None, learner_color: Color) -> Score:
    if learner_color not in ("white", "black"):
        raise ValueError("Learner color must be white or black")
    if score is None:
        return UnavailableScore(pov=learner_color, reason="Engine returned no score")
    relative = score.pov(learner_color == "white")
    if relative.is_mate():
        return MateScore(pov=learner_color, moves=abs(relative.mate()), winning=relative > chess.engine.Cp(0))
    return CentipawnScore(pov=learner_color, cp=relative.score())


def centipawn_loss(best: Score, played: Score) -> int | None:
    if best.pov != played.pov:
        raise ValueError("Normalize both scores to the same learner before comparing")
    if isinstance(best, CentipawnScore) and isinstance(played, CentipawnScore):
        return best.cp - played.cp
    return None


def score_for_ordering(score):
    """Order normalized scores with native mate semantics, without inventing centipawns."""
    if isinstance(score, CentipawnScore):
        return chess.engine.Cp(score.cp)
    if isinstance(score, MateScore):
        if score.winning and score.moves == 0:
            return chess.engine.MateGiven
        return chess.engine.Mate(score.moves if score.winning else -score.moves)
    return None


def comparison_issues(candidates, played):
    """Shared consistency checks for constructing and importing analysis records."""
    scores = [score_for_ordering(candidate.score) for candidate in candidates]
    played_score = score_for_ordering(played.score)
    issues = []
    if any(a is not None and b is not None and a < b for a, b in zip(scores, scores[1:])):
        issues.append("Candidate scores disagree with their engine ranking; reanalyze before judging the move.")
    if scores[0] is not None and played_score is not None and played_score > scores[0]:
        issues.append("The played-move search scores above the first candidate; the comparison is unstable, not a positive loss.")
    if candidates[0].move_uci == played.move_uci and candidates[0].score != played.score:
        issues.append("Separate results disagree on the same move's score; reanalysis is required.")
    profiles = {(c.search.engine, c.search.config_sha256, c.search.time_limit_seconds,
                 c.search.depth_limit, c.search.node_limit) for c in (*candidates, played)}
    if len(profiles) > 1:
        issues.append("Search profiles differ (including MultiPV width); equal time does not imply equal effort per move.")
    return tuple(issues)


def _board(fen):
    board = chess.Board(fen)
    if len(fen.split()) != 6 or not board.is_valid() or board.has_chess960_castling_rights():
        raise ValueError("Expected a valid six-field standard-chess FEN")
    if board.fen() != fen:
        raise ValueError("FEN must use python-chess canonical serialization")
    return board


def _replay(board, moves):
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if board.is_game_over() or move not in board.legal_moves:
            raise ValueError("History or variation contains an illegal move or continues after game over")
        board.push(move)
    return board


def source_game_id(game: chess.pgn.Game) -> str:
    """Content identity ignores usernames/comments but includes the complete move sequence."""
    return _digest({"initial_fen": game.board().fen(), "moves": [move.uci() for move in game.mainline_moves()]})


class PositionContext(ValidatedModel):
    game_id: Digest
    ply: int = Field(ge=1, le=MAX_PLIES)
    initial_fen: Text
    history_uci: tuple[UCI, ...] = Field(max_length=MAX_PLIES - 1, strict=False)
    fen: Text
    learner_color: Color
    played_uci: UCI

    def board(self):
        return _replay(_board(self.initial_fen), self.history_uci)

    @model_validator(mode="after")
    def verify_position(self):
        board = self.board()
        if len(self.history_uci) != self.ply - 1 or board.fen() != self.fen:
            raise ValueError("Position FEN/ply does not match its initial FEN and prior history")
        if board.turn != (self.learner_color == "white"):
            raise ValueError("Learner must be the side making this decision")
        _replay(board, (self.played_uci,))
        return self

    @classmethod
    def from_decision(cls, decision: Decision, *, game_id: str):
        return cls(
            game_id=game_id, ply=decision.ply, initial_fen=decision.board.root().fen(),
            history_uci=tuple(move.uci() for move in decision.board.move_stack), fen=decision.fen,
            learner_color="white" if decision.board.turn else "black", played_uci=decision.uci,
        )


class SearchProvenance(ValidatedModel):
    engine: Text
    config_sha256: Digest
    root_fen: Text
    time_limit_seconds: float | None = Field(default=None, gt=0)
    depth_limit: int | None = Field(default=None, gt=0)
    node_limit: int | None = Field(default=None, gt=0)
    depth: int = Field(ge=0)
    nodes: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    root_moves: tuple[UCI, ...] = Field(default=(), strict=False)

    @model_validator(mode="after")
    def verify_search(self):
        board = _board(self.root_fen)
        if all(value is None for value in (self.time_limit_seconds, self.depth_limit, self.node_limit)):
            raise ValueError("At least one actual engine search limit is required")
        if len(set(self.root_moves)) != len(self.root_moves):
            raise ValueError("Restricted search root moves must be unique")
        if any(chess.Move.from_uci(move) not in board.legal_moves for move in self.root_moves):
            raise ValueError("Restricted search roots must be legal in the search position")
        return self


class Candidate(VersionedRecord):
    root_fen: Text
    move_uci: UCI
    san: Text
    pv_uci: tuple[UCI, ...] = Field(min_length=1, strict=False)
    score: Score
    search: SearchProvenance

    @model_validator(mode="after")
    def verify_line(self):
        board = _board(self.root_fen)
        if self.search.root_fen != self.root_fen:
            raise ValueError("Search provenance belongs to a different root position")
        if self.pv_uci[0] != self.move_uci:
            raise ValueError("Candidate move must be the first move of its PV")
        _replay(board.copy(), self.pv_uci)
        if board.san(chess.Move.from_uci(self.move_uci)) != self.san:
            raise ValueError("Candidate SAN does not match its legal root move")
        if self.search.root_moves and self.move_uci not in self.search.root_moves:
            raise ValueError("Candidate move was not allowed by the declared restricted search")
        return self

    @property
    def pv_san(self):
        board = _board(self.root_fen)
        result = []
        for uci in self.pv_uci:
            move = chess.Move.from_uci(uci)
            result.append(board.san(move))
            board.push(move)
        return tuple(result)

    @classmethod
    def from_engine(cls, board, info, *, learner_color: Color, search: SearchProvenance):
        if not info.get("pv"):
            raise ValueError("Engine result has no principal variation")
        if info["pv"][0] not in board.legal_moves:
            raise ValueError("Engine principal variation starts with an illegal move")
        return cls(
            root_fen=board.fen(), move_uci=info["pv"][0].uci(), san=board.san(info["pv"][0]),
            pv_uci=tuple(move.uci() for move in info["pv"]),
            score=normalize_score(info.get("score"), learner_color), search=search,
        )


class AnalysisRecord(VersionedRecord):
    position: PositionContext
    candidates: tuple[Candidate, ...] = Field(min_length=1, max_length=3, strict=False)
    played: Candidate
    uncertainty: tuple[Text, ...] = Field(default=(), strict=False)

    @model_validator(mode="after")
    def verify_analysis(self):
        if len({candidate.move_uci for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("Engine candidates must have distinct moves")
        if self.played.move_uci != self.position.played_uci:
            raise ValueError("Played-move evidence does not describe the actual played move")
        for candidate in (*self.candidates, self.played):
            if candidate.root_fen != self.position.fen:
                raise ValueError("Every candidate must start at the reviewed position")
            if candidate.score.pov != self.position.learner_color:
                raise ValueError("Every score must be normalized to the learner")
            _replay(self.position.board(), candidate.pv_uci)
        if comparison_issues(self.candidates, self.played) and not self.uncertainty:
            raise ValueError("Conflicting engine results require reanalysis or an explicit uncertainty note")
        return self

    @property
    def loss_cp(self):
        return centipawn_loss(self.candidates[0].score, self.played.score)


class LineReference(ValidatedModel):
    candidate_id: Digest
    ply: int = Field(ge=0)


class CoachingCard(VersionedRecord):
    analysis_id: Digest
    verdict: Text
    summary: Text
    played_move_explanation: Text
    recommended_uci: UCI
    recommendation_explanation: Text
    candidate_comparison: Text
    supporting_lines: tuple[LineReference, ...] = Field(min_length=1, strict=False)
    lesson: Text

    def validate_for(self, analysis: AnalysisRecord):
        if self.analysis_id != analysis.id:
            raise ValueError("Coaching card is linked to a different analysis record")
        lines = {candidate.id: candidate for candidate in (*analysis.candidates, analysis.played)}
        if self.recommended_uci not in {line.move_uci for line in lines.values()}:
            raise ValueError("Recommended move has no supplied candidate/played evidence")
        for reference in self.supporting_lines:
            if reference.candidate_id not in lines or reference.ply > len(lines[reference.candidate_id].pv_uci):
                raise ValueError("Supporting line/ply does not exist in the linked analysis")
        return self


class LabelProvenance(ValidatedModel):
    origin: Literal["human", "model"]
    author_or_model: Text
    source_license: Text
    permission_reference: Text


class TrainingExample(VersionedRecord):
    analysis: AnalysisRecord
    card: CoachingCard
    revision: int = Field(default=1, ge=1)
    status: Literal["draft", "approved", "rejected"] = "draft"
    reviewer: Text | None = None
    provenance: LabelProvenance

    @model_validator(mode="after")
    def verify_example(self):
        self.card.validate_for(self.analysis)
        if self.status == "approved" and self.reviewer is None:
            raise ValueError("Approved training examples require a named human reviewer")
        return self
