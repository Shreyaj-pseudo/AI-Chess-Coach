"""Original fixed boards and synthetic search records; no engine or model calls."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

import chess
import chess.engine
import chess.variant
from pydantic import ValidationError

from analysis_records import AnalysisRecord, Candidate, LineReference, PositionContext, SearchProvenance, source_game_id
from board_facts import BoardEvidence, BoardFact, FactReference, build_board_evidence, extract_board_facts, fact_at
from pgn_input import learner_positions, parse_pgn
from scripts import analyze_position as cli


FIXTURES = Path(__file__).parent / "fixtures"
BOARDS = json.loads((FIXTURES / "board_facts.json").read_text())


def board(name):
    return chess.Board(BOARDS[name]["fen"])


def values(position):
    return {(fact.kind, fact.subject): fact.value for fact in extract_board_facts(position)}


def setup(fen, moves):
    return f'[SetUp "1"]\n[FEN "{fen}"]\n\n{moves}'


def analysis(pgn="1.f3 e5 2.g4 Qh4# 0-1", *, color="white", index=1,
             lines=("e2e4 b8c6", "b1c3 g8f6"), played="g2g4 d8h4"):
    game = parse_pgn(pgn)
    position = PositionContext.from_decision(learner_positions(game, color)[index], game_id=source_game_id(game))

    def candidate(line):
        root = position.board()
        search = SearchProvenance(engine="Synthetic facts fixture 1.0", config_sha256="a" * 64,
                                  root_fen=root.fen(), time_limit_seconds=0.1, depth=10, nodes=42, elapsed_seconds=0.1)
        return Candidate.from_engine(root, {"pv": [chess.Move.from_uci(m) for m in line.split()],
                                           "score": chess.engine.PovScore(chess.engine.Cp(0), root.turn)},
                                     learner_color=color, search=search)

    candidates = tuple(candidate(line) for line in lines)
    played_candidate = next((c for c in candidates if c.pv_uci == tuple(played.split())), None) or candidate(played)
    return AnalysisRecord(position=position, candidates=candidates, played=played_candidate,
                          uncertainty=("Synthetic search values for board-fact tests only.",))


def reference(record, ply, kind, subject, line=None):
    return FactReference(candidate_id=(line or record.played).id, ply=ply, kind=kind, subject=subject)


class ExtractionTests(unittest.TestCase):
    def test_starting_material_counts_nominal_points_and_no_legal_checks_or_captures(self):
        facts = values(chess.Board())
        for color in ("white", "black"):
            self.assertEqual(facts["material_count", f"{color}.pawn"], 8)
            self.assertEqual(facts["material_count", f"{color}.queen"], 1)
            self.assertEqual(facts["material_count", f"{color}.king"], 1)
            self.assertEqual(facts["material_points", color], 39)
        self.assertEqual(facts["legal_check_count", "white"], 0)
        self.assertEqual(facts["legal_capture_count", "white"], 0)
        self.assertFalse(any(facts["open_file", f] for f in chess.FILE_NAMES))

    def test_pinned_defender_counts_geometrically_but_cannot_make_illegal_recapture(self):
        position = board("pinned_defender")
        facts = values(position)
        self.assertEqual(facts["defenders", "f4"], ("e4",))
        self.assertEqual(facts["attackers", "f4"], ("h5",))
        self.assertTrue(facts["absolute_pin", "e4"])
        self.assertEqual(facts["legal_capture", "h5f4"], "Nxf4")
        position.push_uci("h5f4")
        after = values(position)
        self.assertIn("e4", after["attackers", "f4"])
        self.assertNotIn(("legal_capture", "e4f4"), after)
        self.assertIsNone(after["hanging_piece", "f4"])

    def test_attacked_defended_piece_is_not_asserted_hanging_or_safe(self):
        position = board("defended_attacked_piece")
        facts = values(position)
        self.assertEqual(facts["attackers", "f3"], ("g4",))
        self.assertEqual(facts["defenders", "f3"], ("g2",))
        self.assertIsNone(facts["hanging_piece", "f3"])
        self.assertIsNone(facts["winning_tactic", "black"])
        position.remove_piece_at(chess.G2)
        after = values(position)
        self.assertEqual(after["defenders", "f3"], ())
        self.assertIsNone(after["hanging_piece", "f3"])

    def test_open_file_requires_both_sides_to_have_no_pawns_on_it(self):
        position = board("opening_file")
        before = values(position)
        self.assertFalse(before["open_file", "b"])
        self.assertFalse(before["open_file", "c"])
        self.assertTrue(before["open_file", "a"])
        position.push_uci("b4c5")
        after = values(position)
        self.assertTrue(after["open_file", "b"])
        self.assertFalse(after["open_file", "c"])

    def test_passed_isolated_and_doubled_pawns_have_explicit_geometric_definitions(self):
        position = board("pawn_structure")
        facts = values(position)
        self.assertTrue(facts["passed_pawn", "e5"])
        self.assertFalse(facts["passed_pawn", "e3"])
        self.assertFalse(facts["passed_pawn", "g2"])
        for square in ("e3", "e5"):
            self.assertTrue(facts["doubled_pawn", square])
            self.assertTrue(facts["isolated_pawn", square])
        position.set_piece_at(chess.D2, chess.Piece(chess.PAWN, chess.WHITE))
        self.assertFalse(values(position)["isolated_pawn", "e5"])

    def test_black_passed_pawns_use_the_opposite_rank_direction(self):
        facts = values(board("pawn_structure").mirror())
        self.assertTrue(facts["passed_pawn", "e4"])
        self.assertFalse(facts["passed_pawn", "e6"])
        self.assertEqual(facts["piece", "e4"], "p")

    def test_en_passant_is_a_legal_capture_without_a_geometric_attack_on_the_victim_square(self):
        position = chess.Board("7k/8/8/3pP3/8/8/8/7K w - d6 0 2")
        facts = values(position)
        self.assertEqual(facts["legal_capture", "e5d6"], "exd6")
        self.assertEqual(facts["attackers", "d5"], ())
        position.push_uci("e5d6")
        after = values(position)
        self.assertIsNone(after["piece", "d5"])
        self.assertEqual(after["piece", "d6"], "P")
        self.assertEqual(after["material_count", "black.pawn"], 0)

    def test_castling_moves_both_pieces_in_facts(self):
        position = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
        position.push_uci("e1g1")
        facts = values(position)
        self.assertEqual(facts["piece", "g1"], "K")
        self.assertEqual(facts["piece", "f1"], "R")
        self.assertIsNone(facts["piece", "e1"])
        self.assertIsNone(facts["piece", "h1"])

    def test_underpromotion_updates_material_and_automatic_draw(self):
        position = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
        position.push_uci("a7a8n")
        facts = values(position)
        self.assertEqual(facts["material_count", "white.pawn"], 0)
        self.assertEqual(facts["material_count", "white.knight"], 1)
        self.assertEqual(facts["material_points", "white"], 3)
        self.assertEqual(facts["termination", "board"], "insufficient_material")
        self.assertEqual(facts["legal_check_count", "black"], 0)

    def test_stalemate_and_automatic_draw_suppress_postgame_move_facts(self):
        stalemate = values(chess.Board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"))
        self.assertEqual(stalemate["termination", "board"], "stalemate")
        self.assertFalse(stalemate["in_check", "black"])
        drawn = chess.Board("7k/8/8/8/8/8/R7/K7 w - - 150 76")
        self.assertGreater(drawn.legal_moves.count(), 0)
        facts = values(drawn)
        self.assertEqual(facts["termination", "board"], "seventyfive_moves")
        self.assertEqual(facts["legal_check_count", "white"], 0)
        self.assertEqual(facts["legal_capture_count", "white"], 0)
        self.assertFalse(any(kind in ("legal_check", "legal_capture") for kind, subject in facts))

    def test_claimable_threefold_is_not_reported_as_automatic_game_over(self):
        position = chess.Board()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            position.push_uci(move)
        self.assertTrue(position.can_claim_threefold_repetition())
        self.assertIsNone(values(position)["termination", "board"])

    def test_extraction_is_deterministic_preserves_input_and_rejects_invalid_variants(self):
        position = chess.Board()
        position.push_san("e4")
        fen, history = position.fen(), tuple(position.move_stack)
        first = extract_board_facts(position)
        self.assertEqual(first, extract_board_facts(position))
        self.assertEqual(position.fen(), fen)
        self.assertEqual(tuple(position.move_stack), history)
        self.assertEqual(len(first), len({(f.kind, f.subject) for f in first}))
        for invalid in (None, "FEN", chess.Board(None), chess.Board(chess960=True), chess.variant.CrazyhouseBoard()):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                extract_board_facts(invalid)


class EvidenceTests(unittest.TestCase):
    def test_facts_use_the_referenced_ply_not_the_unrelated_root(self):
        record = analysis()
        self.assertEqual(fact_at(record, reference(record, 1, "legal_check", "d8h4")).value, "Qh4#")
        with self.assertRaisesRegex(ValueError, "No supported fact"):
            fact_at(record, reference(record, 0, "legal_check", "d8h4"))
        self.assertEqual(fact_at(record, reference(record, 2, "termination", "board")).value, "checkmate")
        self.assertTrue(fact_at(record, reference(record, 2, "in_check", "white")).value)
        self.assertEqual(fact_at(record, reference(record, 2, "legal_capture_count", "white")).value, 0)
        self.assertEqual(fact_at(record, reference(record, 1, "piece", "g4")).value, "P")
        other = record.candidates[0]
        self.assertIsNone(fact_at(record, reference(record, 1, "piece", "g4", other)).value)

    def test_full_snapshots_and_changes_cover_each_unique_line_and_every_ply(self):
        record = analysis()
        evidence = build_board_evidence(record)
        lines = {line.id: line for line in (*record.candidates, record.played)}
        self.assertEqual(len(evidence.positions), sum(len(line.pv_uci) + 1 for line in lines.values()))
        for line in lines.values():
            snapshots = [s for s in evidence.positions if s.reference.candidate_id == line.id]
            self.assertEqual([s.reference.ply for s in snapshots], list(range(len(line.pv_uci) + 1)))
            self.assertEqual((snapshots[0].added, snapshots[0].removed), ((), ()))
            previous = set(snapshots[0].facts)
            for snapshot in snapshots[1:]:
                self.assertEqual((previous - set(snapshot.removed)) | set(snapshot.added), set(snapshot.facts))
                previous = set(snapshot.facts)

    def test_file_opening_and_capture_have_exact_added_and_removed_facts(self):
        record = analysis(setup(BOARDS["opening_file"]["fen"], "1. bxc5 *"), index=0,
                          lines=("b4c5 h8g7",), played="b4c5 h8g7")
        evidence = build_board_evidence(record)
        self.assertEqual(len(evidence.positions), 3)  # Played evidence reused from the candidate.
        snapshot = evidence.positions[1]
        added = {(f.kind, f.subject): f.value for f in snapshot.added}
        removed = {(f.kind, f.subject): f.value for f in snapshot.removed}
        self.assertTrue(added["open_file", "b"])
        self.assertFalse(removed["open_file", "b"])
        self.assertEqual((removed["piece", "b4"], added["piece", "b4"]), ("P", None))
        self.assertEqual((removed["piece", "c5"], added["piece", "c5"]), ("p", "P"))
        self.assertEqual((removed["material_count", "black.pawn"], added["material_count", "black.pawn"]), (1, 0))
        self.assertTrue(fact_at(record, reference(record, 1, "passed_pawn", "c5")).value)

    def test_history_based_automatic_draw_is_preserved_at_referenced_ply(self):
        pgn = "1.Nf3 Nf6 2.Ng1 Ng8 3.Nf3 Nf6 4.Ng1 Ng8 5.Nf3 Nf6 6.Ng1 Ng8 7.Nf3 Nf6 8.Ng1 Ng8 *"
        record = analysis(pgn, color="black", index=7, lines=("f6g8",), played="f6g8")
        evidence = build_board_evidence(record)
        last = evidence.positions[-1]
        self.assertEqual(fact_at(record, reference(record, 1, "termination", "board")).value, "fivefold_repetition")
        self.assertIsNone(values(chess.Board(last.fen))["termination", "board"])

    def test_unknown_candidates_out_of_range_plies_and_unsupported_subjects_fail(self):
        record = analysis()
        invalid = (FactReference(candidate_id="f" * 64, ply=0, kind="piece", subject="e1"),
                   reference(record, 3, "piece", "e1"), reference(record, 0, "passed_pawn", "e1"),
                   reference(record, 0, "legal_check", "a2a4"))
        for ref in invalid:
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                fact_at(record, ref)
        with self.assertRaises(ValidationError):
            reference(record, -1, "piece", "e1")

    def test_unknown_tactical_conclusions_cannot_be_promoted_into_claimed_facts(self):
        record = analysis()
        self.assertIsNone(fact_at(record, reference(record, 0, "hanging_piece", "g2")).value)
        self.assertIsNone(fact_at(record, reference(record, 0, "winning_tactic", "black")).value)
        for kind, subject in (("hanging_piece", "g2"), ("winning_tactic", "black")):
            for conclusion in (True, False, "yes", ("g2",)):
                with self.subTest(kind=kind, conclusion=conclusion), self.assertRaises(ValidationError):
                    BoardFact(kind=kind, subject=subject, value=conclusion)

    def test_fact_shape_prevents_bool_counts_unknown_kinds_and_malformed_squares(self):
        invalid = (("material_count", "white.pawn", True), ("material_points", "white", -1),
                   ("piece", "i9", "P"), ("piece", "e4", "queen"), ("absolute_pin", "e4", 1),
                   ("attackers", "e4", ("i9",)), ("defenders", "e4", ("e3", "e3")),
                   ("is_winning", "white", True))
        for kind, subject, value in invalid:
            with self.subTest(kind=kind, subject=subject), self.assertRaises(ValidationError):
                BoardFact(kind=kind, subject=subject, value=value)

    def test_bundle_roundtrip_content_ids_and_analysis_compatibility(self):
        record = analysis()
        original = record.model_dump_json()
        evidence = build_board_evidence(record)
        restored = BoardEvidence.model_validate_json(evidence.model_dump_json())
        self.assertEqual(restored, evidence)
        self.assertIs(restored.validate_for(record), restored)
        self.assertEqual(build_board_evidence(record).id, evidence.id)
        self.assertEqual(record.model_dump_json(), original)
        self.assertEqual(evidence.analysis_id, record.id)
        self.assertTrue(BoardEvidence.model_json_schema())

    def test_foreign_analysis_fabricated_facts_deltas_fens_and_missing_positions_are_rejected(self):
        record = analysis()
        evidence = build_board_evidence(record)
        for mutation in ("foreign", "fact", "delta", "fen", "reference", "missing", "duplicate"):
            data = evidence.model_dump(mode="json", exclude={"id"})
            if mutation == "foreign":
                data["analysis_id"] = "f" * 64
            elif mutation == "fact":
                fact = next(f for f in data["positions"][0]["facts"] if f["kind"] == "material_count" and f["subject"] == "white.queen")
                fact["value"] = 0
            elif mutation == "delta":
                data["positions"][0]["added"] = [{"kind": "open_file", "subject": "e", "value": True}]
            elif mutation == "fen":
                data["positions"][0]["fen"] = chess.STARTING_FEN
            elif mutation == "reference":
                data["positions"][0]["reference"]["ply"] = 1
            elif mutation == "missing":
                data["positions"].pop()
            else:
                data["positions"].append(data["positions"][0])
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "does not match"):
                BoardEvidence.model_validate(data).validate_for(record)


class CLITests(unittest.TestCase):
    def test_facts_flag_exports_linked_evidence_and_the_original_analysis(self):
        record = analysis()
        output = io.StringIO()
        with patch.object(cli, "_analyze", new=AsyncMock(return_value=record)), redirect_stdout(output):
            code = cli.main([str(FIXTURES / "pgn/played_blunder.pgn"), "--color", "white", "--ply", "3", "--facts"])
        self.assertEqual(code, 0)
        envelope = json.loads(output.getvalue())
        exported = AnalysisRecord.model_validate(envelope["analysis"])
        self.assertEqual(exported, record)
        BoardEvidence.model_validate(envelope["board_evidence"]).validate_for(exported)

    def test_fact_failure_suppresses_partial_json_with_actionable_error(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(cli, "_analyze", new=AsyncMock(return_value=analysis())), \
                patch.object(cli, "build_board_evidence", side_effect=ValueError("fact extraction failed")), \
                redirect_stdout(output), redirect_stderr(errors):
            code = cli.main([str(FIXTURES / "pgn/played_blunder.pgn"), "--color", "white", "--ply", "3", "--facts"])
        self.assertEqual(code, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("fact extraction failed", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
