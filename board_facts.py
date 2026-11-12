"""Deterministic observations, not tactical verdicts, at exact analysis line plies."""

import re
from typing import Annotated, Literal

import chess
from pydantic import Field, model_validator

from analysis_records import AnalysisRecord, Digest, LineReference, Text, ValidatedModel, VersionedRecord


FactKind = Literal[
    "piece", "material_count", "material_points", "turn", "in_check", "termination",
    "legal_check", "legal_capture", "legal_check_count", "legal_capture_count",
    "attackers", "defenders", "absolute_pin", "open_file", "passed_pawn",
    "isolated_pawn", "doubled_pawn", "hanging_piece", "winning_tactic",
]
Square = Annotated[str, Field(pattern=r"^[a-h][1-8]$")]
Squares = Annotated[tuple[Square, ...], Field(strict=False)]
FactValue = bool | int | str | Squares | None
COLORS = {chess.WHITE: "white", chess.BLACK: "black"}
POINTS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class BoardFact(ValidatedModel):
    kind: FactKind
    subject: Text
    value: FactValue

    @model_validator(mode="after")
    def verify_shape(self):
        square_kinds = {"piece", "attackers", "defenders", "absolute_pin", "passed_pawn", "isolated_pawn", "doubled_pawn", "hanging_piece"}
        bool_kinds = {"in_check", "absolute_pin", "open_file", "passed_pawn", "isolated_pawn", "doubled_pawn"}
        count_kinds = {"material_count", "material_points", "legal_check_count", "legal_capture_count"}
        if self.kind in square_kinds:
            valid_subject = re.fullmatch(r"[a-h][1-8]", self.subject)
        elif self.kind == "material_count":
            valid_subject = re.fullmatch(r"(white|black)\.(pawn|knight|bishop|rook|queen|king)", self.subject)
        elif self.kind == "open_file":
            valid_subject = self.subject in chess.FILE_NAMES
        elif self.kind in {"legal_check", "legal_capture"}:
            valid_subject = re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", self.subject)
        elif self.kind in {"turn", "termination"}:
            valid_subject = self.subject == "board"
        else:
            valid_subject = self.subject in ("white", "black")
        if self.kind in bool_kinds:
            valid_value = type(self.value) is bool
        elif self.kind in count_kinds:
            valid_value = type(self.value) is int and self.value >= 0
        elif self.kind in {"attackers", "defenders"}:
            valid_value = isinstance(self.value, tuple) and len(self.value) == len(set(self.value))
        elif self.kind == "piece":
            valid_value = self.value is None or (type(self.value) is str and self.value in tuple("PNBRQKpnbrqk"))
        elif self.kind == "turn":
            valid_value = self.value in ("white", "black")
        elif self.kind == "termination":
            valid_value = self.value is None or self.value in (
                "checkmate", "stalemate", "insufficient_material", "seventyfive_moves", "fivefold_repetition",
            )
        elif self.kind in {"hanging_piece", "winning_tactic"}:
            valid_value = self.value is None
        else:
            valid_value = type(self.value) is str and bool(self.value.strip())
        if not valid_subject or not valid_value:
            raise ValueError(f"Invalid subject/value shape for {self.kind}; tactical conclusions must remain unknown.")
        return self


class FactReference(LineReference):
    kind: FactKind
    subject: Text


class FactSnapshot(ValidatedModel):
    reference: LineReference
    fen: Text
    facts: tuple[BoardFact, ...] = Field(min_length=1, strict=False)
    added: tuple[BoardFact, ...] = Field(default=(), strict=False)
    removed: tuple[BoardFact, ...] = Field(default=(), strict=False)


class BoardEvidence(VersionedRecord):
    analysis_id: Digest
    positions: tuple[FactSnapshot, ...] = Field(min_length=1, strict=False)

    def validate_for(self, analysis: AnalysisRecord):
        checked = type(self).model_validate(self)
        if checked != build_board_evidence(analysis):
            raise ValueError("Board evidence does not match the analysis, exact line/ply, facts or changes.")
        return self


def extract_board_facts(board: chess.Board) -> tuple[BoardFact, ...]:
    """Use full history for automatic draws; attack/defense lists include pinned pieces."""
    if not isinstance(board, chess.Board) or not board.is_valid() or board.chess960 or board.uci_variant != "chess":
        raise ValueError("Board facts require a valid standard chess board.")
    facts = []

    def add(kind, subject, value):
        facts.append(BoardFact(kind=kind, subject=subject, value=value))

    outcome = board.outcome(claim_draw=False)
    add("turn", "board", COLORS[board.turn])
    add("in_check", COLORS[board.turn], board.is_check())
    add("termination", "board", outcome.termination.name.lower() if outcome else None)
    for color, name in COLORS.items():
        for piece_type in chess.PIECE_TYPES:
            add("material_count", f"{name}.{chess.piece_name(piece_type)}", len(board.pieces(piece_type, color)))
        add("material_points", name, sum(POINTS[p] * len(board.pieces(p, color)) for p in chess.PIECE_TYPES))
        add("winning_tactic", name, None)
    all_pawns = board.pawns
    for file, name in enumerate(chess.FILE_NAMES):
        add("open_file", name, not bool(all_pawns & chess.BB_FILES[file]))
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        name = chess.square_name(square)
        add("piece", name, piece.symbol() if piece else None)
        if piece is None:
            continue
        add("attackers", name, tuple(chess.square_name(s) for s in board.attackers(not piece.color, square)))
        add("defenders", name, tuple(chess.square_name(s) for s in board.attackers(piece.color, square)))
        add("absolute_pin", name, board.is_pinned(piece.color, square))
        add("hanging_piece", name, None)
        if piece.piece_type == chess.PAWN:
            file, rank = chess.square_file(square), chess.square_rank(square)
            own = board.pieces(chess.PAWN, piece.color)
            enemy = board.pieces(chess.PAWN, not piece.color)
            add("doubled_pawn", name, sum(chess.square_file(s) == file for s in own) > 1)
            add("isolated_pawn", name, not any(abs(chess.square_file(s) - file) == 1 for s in own))
            add("passed_pawn", name, not any(
                abs(chess.square_file(s) - file) <= 1 and
                (chess.square_rank(s) > rank if piece.color else chess.square_rank(s) < rank) for s in enemy
            ))
    checks = captures = 0
    # A drawn position may still have geometric legal moves, but the game has ended.
    for move in list(board.legal_moves) if outcome is None else ():
        if board.gives_check(move):
            add("legal_check", move.uci(), board.san(move))
            checks += 1
        if board.is_capture(move):
            add("legal_capture", move.uci(), board.san(move))
            captures += 1
    add("legal_check_count", COLORS[board.turn], checks)
    add("legal_capture_count", COLORS[board.turn], captures)
    return tuple(sorted(facts, key=lambda f: (f.kind, f.subject)))


def _lines(analysis):
    return {line.id: line for line in (*analysis.candidates, analysis.played)}


def fact_at(analysis: AnalysisRecord, reference: FactReference) -> BoardFact:
    """Recompute a supported fact at exactly this line/ply; absent facts are errors."""
    analysis = AnalysisRecord.model_validate(analysis)
    reference = FactReference.model_validate(reference)
    line = _lines(analysis).get(reference.candidate_id)
    if line is None or reference.ply > len(line.pv_uci):
        raise ValueError("Fact reference has an unknown candidate or out-of-range line ply.")
    board = analysis.position.board()
    for uci in line.pv_uci[:reference.ply]:
        board.push_uci(uci)
    for fact in extract_board_facts(board):
        if fact.kind == reference.kind and fact.subject == reference.subject:
            return fact
    raise ValueError("No supported fact exists for this subject at the referenced position.")


def build_board_evidence(analysis: AnalysisRecord) -> BoardEvidence:
    """Root=ply 0; later snapshots are after that many line moves, with exact deltas."""
    analysis = AnalysisRecord.model_validate(analysis)
    positions = []
    for line in _lines(analysis).values():
        board = analysis.position.board()
        previous = ()
        for ply in range(len(line.pv_uci) + 1):
            facts = extract_board_facts(board)
            old_set, new_set = set(previous), set(facts)
            positions.append(FactSnapshot(
                reference=LineReference(candidate_id=line.id, ply=ply), fen=board.fen(), facts=facts,
                added=tuple(f for f in facts if f not in old_set) if ply else (),
                removed=tuple(f for f in previous if f not in new_set) if ply else (),
            ))
            previous = facts
            if ply < len(line.pv_uci):
                board.push_uci(line.pv_uci[ply])
    return BoardEvidence(analysis_id=analysis.id, positions=tuple(positions))
