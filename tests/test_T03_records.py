"""All engine values and human labels here are explicit synthetic test fixtures."""

import json
from pathlib import Path
import unittest

import chess
import chess.engine
from pydantic import TypeAdapter, ValidationError

from analysis_records import (
    AnalysisRecord, Candidate, CentipawnScore, CoachingCard, LabelProvenance,
    LineReference, MateScore, PositionContext, Score, SearchProvenance,
    TrainingExample, UnavailableScore, centipawn_loss, normalize_score, source_game_id,
)
from pgn_input import learner_positions, parse_pgn


FIXTURES = Path(__file__).parent / "fixtures" / "pgn"


def changed(record, **updates):
    payload = record.model_dump(exclude={"id"})
    payload.update(updates)
    return type(record).model_validate(payload)


def context(pgn="1.e4 e5 2.Nf3 *", color="white", index=0):
    game = parse_pgn(pgn)
    return PositionContext.from_decision(learner_positions(game, color)[index], game_id=source_game_id(game))


def search(board=None, **updates):
    payload = dict(engine="Synthetic fixture engine 1.0", config_sha256="a" * 64,
                   root_fen=(board or chess.Board()).fen(), time_limit_seconds=0.1,
                   depth=10, nodes=1234, elapsed_seconds=0.1)
    payload.update(updates)
    return SearchProvenance(**payload)


def candidate(board=None, line=("e2e4", "e7e5"), cp=30, pov="white", **updates):
    board = chess.Board() if board is None else board
    payload = dict(root_fen=board.fen(), move_uci=line[0], san=board.san(chess.Move.from_uci(line[0])),
                   pv_uci=line, score=CentipawnScore(pov=pov, cp=cp), search=search(board))
    payload.update(updates)
    return Candidate(**payload)


def analysis():
    return AnalysisRecord(
        position=context(), candidates=(candidate(), candidate(line=("d2d4", "d7d5"), cp=20)),
        played=candidate(),
    )


def card(record):
    return CoachingCard(
        analysis_id=record.id, verdict="good", summary="Synthetic schema test card, not an audited chess lesson.",
        played_move_explanation="The fixture move advances a central pawn.", recommended_uci="e2e4",
        recommendation_explanation="Fixture explanation backed only by the supplied test line.",
        candidate_comparison="Both supplied fixture candidates are legal.",
        supporting_lines=(LineReference(candidate_id=record.candidates[0].id, ply=1),),
        lesson="This text is for testing the schema, not for training.",
    )


def example():
    record = analysis()
    return TrainingExample(
        analysis=record, card=card(record), provenance=LabelProvenance(
            origin="human", author_or_model="synthetic-fixture-author", source_license="test fixture only",
            permission_reference="tests/test_T03_records.py; no actual human approval",
        ),
    )


class ScoreTests(unittest.TestCase):
    def test_before_and_after_scores_keep_white_perspective(self):
        before = normalize_score(chess.engine.PovScore(chess.engine.Cp(70), chess.WHITE), "white")
        after = normalize_score(chess.engine.PovScore(chess.engine.Cp(-20), chess.BLACK), "white")
        self.assertEqual((before.cp, after.cp), (70, 20))
        self.assertEqual(centipawn_loss(before, after), 50)

    def test_before_and_after_scores_keep_black_perspective(self):
        before = normalize_score(chess.engine.PovScore(chess.engine.Cp(70), chess.BLACK), "black")
        after = normalize_score(chess.engine.PovScore(chess.engine.Cp(-20), chess.WHITE), "black")
        self.assertEqual((before.cp, after.cp), (70, 20))
        self.assertEqual(centipawn_loss(before, after), 50)

    def test_mixed_perspective_subtraction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same learner"):
            centipawn_loss(CentipawnScore(pov="white", cp=70), CentipawnScore(pov="black", cp=20))

    def test_negative_gap_is_preserved_not_clamped(self):
        self.assertEqual(centipawn_loss(CentipawnScore(pov="white", cp=10), CentipawnScore(pov="white", cp=20)), -10)

    def test_mate_scores_have_distance_and_direction_not_centipawns(self):
        engine_score = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
        white = normalize_score(engine_score, "white")
        black = normalize_score(engine_score, "black")
        self.assertEqual((white.moves, white.winning), (3, True))
        self.assertEqual((black.moves, black.winning), (3, False))
        self.assertFalse(hasattr(white, "cp"))
        self.assertIsNone(centipawn_loss(white, CentipawnScore(pov="white", cp=0)))

    def test_mate_given_and_mated_zero_remain_distinct(self):
        for raw, winning in ((chess.engine.MateGiven, True), (chess.engine.Mate(0), False)):
            with self.subTest(raw=raw):
                score = normalize_score(chess.engine.PovScore(raw, chess.WHITE), "white")
                self.assertEqual((score.moves, score.winning), (0, winning))
                opposite = normalize_score(chess.engine.PovScore(raw, chess.WHITE), "black")
                self.assertEqual(opposite.winning, not winning)

    def test_unavailable_score_is_not_zero(self):
        score = normalize_score(None, "black")
        self.assertIsInstance(score, UnavailableScore)
        self.assertIsNone(centipawn_loss(score, CentipawnScore(pov="black", cp=0)))

    def test_score_tags_strict_types_and_unknown_fields(self):
        adapter = TypeAdapter(Score)
        invalid = (
            {"kind": "cp", "pov": "white", "cp": "30"},
            {"kind": "cp", "pov": "white", "cp": True},
            {"kind": "cp", "pov": "white", "cp": 30, "mate": 2},
            {"kind": "mate", "pov": "white", "moves": -1, "winning": True},
            {"kind": "mate", "pov": "white", "moves": 2, "winning": "yes"},
            {"kind": "unavailable", "pov": "white", "reason": " "},
            {"kind": "mystery", "pov": "white", "value": 3},
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                adapter.validate_json(json.dumps(payload))
        with self.assertRaises(ValueError):
            normalize_score(None, "both")


class PositionTests(unittest.TestCase):
    def test_context_factory_preserves_repetition_stack(self):
        game = parse_pgn((FIXTURES / "repetition.pgn").read_bytes())
        position = PositionContext.from_decision(learner_positions(game, "white")[-1], game_id=source_game_id(game))
        self.assertEqual(position.ply, 9)
        self.assertTrue(position.board().is_repetition(3))
        self.assertFalse(chess.Board(position.fen).is_repetition(3))

    def test_custom_initial_position_and_black_turn_roundtrip(self):
        game = parse_pgn((FIXTURES / "black_setup.pgn").read_bytes())
        position = PositionContext.from_decision(learner_positions(game, "black")[0], game_id=source_game_id(game))
        self.assertEqual(position.ply, 1)
        self.assertEqual(position.board().fullmove_number, 42)
        self.assertEqual(PositionContext.model_validate_json(position.model_dump_json()), position)

    def test_wrong_history_fen_ply_side_and_played_move_are_rejected(self):
        original = context()
        updates = ({"ply": 2}, {"fen": chess.Board().mirror().fen()}, {"history_uci": ("e2e4",)},
                   {"learner_color": "black"}, {"played_uci": "e2e5"}, {"initial_fen": "invalid"},
                   {"game_id": "not-a-hash"})
        for update in updates:
            with self.subTest(update=update), self.assertRaises(ValidationError):
                changed(original, **update)

    def test_history_cannot_continue_after_an_automatic_draw(self):
        history = ("g1f3", "g8f6", "f3g1", "f6g8") * 4 + ("g1f3",)
        with self.assertRaisesRegex(ValidationError, "game over"):
            PositionContext(game_id="b" * 64, ply=18, initial_fen=chess.STARTING_FEN,
                            history_uci=history, fen=chess.STARTING_FEN, learner_color="black", played_uci="g8f6")

    def test_game_identity_ignores_comments_and_player_headers(self):
        first = parse_pgn('[White "Alice"]\n\n1.e4 {comment} e5 *')
        second = parse_pgn('[White "Bob"]\n\n1.e4 e5 *')
        self.assertEqual(source_game_id(first), source_game_id(second))
        self.assertNotEqual(source_game_id(first), source_game_id(parse_pgn("1.e4 c5 *")))


class EvidenceTests(unittest.TestCase):
    def test_engine_adapter_normalizes_and_derives_san_from_same_pv(self):
        board = chess.Board()
        info = {"pv": [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")],
                "score": chess.engine.PovScore(chess.engine.Cp(-30), chess.BLACK)}
        record = Candidate.from_engine(board, info, learner_color="white", search=search(board))
        self.assertEqual(record.score.cp, 30)
        self.assertEqual(record.pv_san, ("e4", "e5"))
        self.assertEqual(board.fen(), chess.STARTING_FEN)
        self.assertIsInstance(Candidate.from_engine(board, {"pv": info["pv"]}, learner_color="white", search=search()).score, UnavailableScore)
        with self.assertRaisesRegex(ValueError, "no principal variation"):
            Candidate.from_engine(board, {}, learner_color="white", search=search())
        with self.assertRaisesRegex(ValueError, "illegal move"):
            Candidate.from_engine(board, {"pv": [chess.Move.from_uci("e2e5")]}, learner_color="white", search=search())

    def test_search_provenance_is_required_and_bounded(self):
        data = candidate().model_dump(exclude={"id"})
        del data["search"]
        with self.assertRaises(ValidationError):
            Candidate.model_validate(data)
        for update in ({"engine": " "}, {"config_sha256": "unknown"}, {"time_limit_seconds": None},
                       {"time_limit_seconds": 0}, {"depth": -1}, {"nodes": True}, {"elapsed_seconds": float("nan")},
                       {"root_moves": ("e2e5",)}, {"root_moves": ("e2e4", "e2e4")}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                search(**update)

    def test_invalid_roots_pv_san_and_search_root_are_rejected(self):
        original = candidate()
        bad_search = search(root_moves=("d2d4",))
        other_board = chess.Board()
        other_board.push_san("d4")
        updates = ({"root_fen": "invalid"}, {"move_uci": "d2d4"}, {"pv_uci": ()},
                   {"pv_uci": ("e2e4", "e2e5")}, {"san": "Nf3"}, {"search": bad_search},
                   {"search": search(other_board)})
        for update in updates:
            with self.subTest(update=update), self.assertRaises(ValidationError):
                changed(original, **update)

    def test_played_move_outside_top_candidates_keeps_separate_evidence(self):
        record = AnalysisRecord(position=context("1.a3 *"), candidates=(candidate(),),
                                played=candidate(line=("a2a3", "e7e5"), cp=-40,
                                                 search=search(root_moves=("a2a3",))))
        self.assertEqual(record.played.san, "a3")
        self.assertEqual(record.played.search.root_moves, ("a2a3",))
        self.assertEqual(record.loss_cp, 70)
        self.assertNotEqual(record.candidates[0].id, record.played.id)

    def test_mixed_viewpoints_wrong_played_move_duplicate_and_extra_candidates_fail(self):
        record = analysis()
        four = tuple(candidate(line=(move,), cp=30-index) for index, move in enumerate(("e2e4", "d2d4", "c2c4", "g1f3")))
        updates = ({"played": candidate(pov="black")}, {"played": candidate(line=("a2a3",))},
                   {"candidates": (candidate(), candidate())}, {"candidates": four})
        for update in updates:
            with self.subTest(update=update), self.assertRaises(ValidationError):
                changed(record, **update)

    def test_other_legal_root_cannot_be_used_for_this_analysis(self):
        board = chess.Board()
        board.push_san("d4")
        other = candidate(board, line=("d7d5",), pov="white")
        with self.assertRaisesRegex(ValidationError, "reviewed position"):
            changed(analysis(), candidates=(other,))

    def test_pv_replays_against_real_history_not_only_root_fen(self):
        game = parse_pgn("1.Nf3 Nf6 2.Ng1 Ng8 3.Nf3 Nf6 4.Ng1 Ng8 5.Nf3 Nf6 6.Ng1 Ng8 7.Nf3 Nf6 8.Ng1 Ng8 *")
        position = PositionContext.from_decision(learner_positions(game, "black")[-1], game_id=source_game_id(game))
        line = candidate(position.board(), line=("f6g8", "g1f3"), pov="black")
        # It is legal from FEN alone, but the first move makes the fifth repetition.
        with self.assertRaisesRegex(ValidationError, "game over"):
            AnalysisRecord(position=position, candidates=(line,), played=line)

    def test_conflicting_searches_require_explicit_uncertainty(self):
        record = analysis()
        for played in (candidate(cp=40), candidate(cp=20), candidate(search=search(engine="Another fixture engine"))):
            with self.subTest(played=played.score), self.assertRaisesRegex(ValidationError, "uncertainty"):
                changed(record, played=played)
        changed_record = changed(record, played=candidate(cp=40), uncertainty=("Search results conflict; reanalyze.",))
        self.assertEqual(changed_record.loss_cp, -10)

    def test_misordered_candidates_require_uncertainty(self):
        with self.assertRaisesRegex(ValidationError, "uncertainty"):
            changed(analysis(), candidates=(candidate(cp=10), candidate(line=("d2d4",), cp=20)))

    def test_mate_and_unknown_analysis_loss_is_not_a_centipawn_number(self):
        for score in (MateScore(pov="white", moves=3, winning=True), UnavailableScore(pov="white", reason="fixture unavailable")):
            line = candidate(score=score)
            record = AnalysisRecord(position=context(), candidates=(line,), played=line)
            self.assertIsNone(record.loss_cp)

    def test_special_rule_moves_convert_to_records_and_roundtrip(self):
        for name, color, ply in (("castling", "white", 9), ("en_passant", "white", 5),
                                 ("promotion", "white", 1), ("black_setup", "black", 1)):
            with self.subTest(fixture=name):
                game = parse_pgn((FIXTURES / f"{name}.pgn").read_bytes())
                decision = next(item for item in learner_positions(game, color) if item.ply == ply)
                position = PositionContext.from_decision(decision, game_id=source_game_id(game))
                line = Candidate.from_engine(position.board(), {"pv": [decision.played_move]},
                                             learner_color=color, search=search(position.board()))
                record = AnalysisRecord(position=position, candidates=(line,), played=line)
                self.assertEqual(line.san, decision.san)
                self.assertEqual(AnalysisRecord.model_validate_json(record.model_dump_json()), record)


class RecordAndLabelTests(unittest.TestCase):
    def test_all_four_record_types_roundtrip_json_and_export_schemas(self):
        record = analysis()
        for value in (candidate(), record, card(record), example()):
            with self.subTest(model=type(value).__name__):
                encoded = value.model_dump_json()
                self.assertEqual(type(value).model_validate_json(encoded), value)
                self.assertEqual(type(value).model_validate(json.loads(encoded)), value)
                self.assertEqual(len(value.id), 64)
                self.assertTrue(json.dumps(type(value).model_json_schema()))

    def test_ids_are_stable_and_change_when_content_changes(self):
        first = analysis()
        self.assertEqual(first.id, analysis().id)
        self.assertNotEqual(first.id, changed(first, uncertainty=("Fixture review note",)).id)
        old_card = card(first)
        self.assertNotEqual(old_card.id, changed(old_card, lesson="A revised test-only lesson.").id)
        self.assertEqual(first.id, analysis().id)

    def test_tampered_id_and_unsupported_schema_version_fail(self):
        for value in (candidate(), analysis(), card(analysis()), example()):
            for update in ({"id": "0" * 64}, {"schema_version": 2}, {"schema_version": True}, {"schema_version": "1"}):
                with self.subTest(model=type(value).__name__, update=update), self.assertRaises(ValidationError):
                    type(value).model_validate({**value.model_dump(), **update})

    def test_frozen_models_and_tuples_prevent_accidental_nested_edits(self):
        value = example()
        with self.assertRaises(ValidationError):
            value.analysis.played.score.cp = 500
        with self.assertRaises(TypeError):
            value.analysis.candidates[0] = candidate()
        data = value.model_dump()
        data["card"]["lesson"] = "Changing a dump must not change the record"
        self.assertNotEqual(value.card.lesson, data["card"]["lesson"])

    def test_unsafe_model_copy_is_revalidated_at_record_boundary(self):
        unsafe = candidate().model_copy(update={"move_uci": "e2e5"})
        with self.assertRaises(ValidationError):
            AnalysisRecord(position=context(), candidates=(unsafe,), played=candidate())

    def test_unknown_fields_and_empty_card_text_are_rejected(self):
        with self.assertRaises(ValidationError):
            changed(candidate(), invented_engine_claim=True)
        with self.assertRaises(ValidationError):
            changed(card(analysis()), lesson=" ")

    def test_card_links_and_reference_ranges_are_checked(self):
        record = analysis()
        good = card(record)
        self.assertEqual(good.validate_for(record), good)
        invalid = (
            {"analysis_id": "0" * 64}, {"recommended_uci": "a2a3"},
            {"supporting_lines": (LineReference(candidate_id="0" * 64, ply=1),)},
            {"supporting_lines": (LineReference(candidate_id=record.candidates[0].id, ply=99),)},
        )
        for update in invalid:
            with self.subTest(update=update), self.assertRaises(ValueError):
                changed(good, **update).validate_for(record)

    def test_training_examples_start_unapproved_and_require_real_reviewer_metadata(self):
        draft = example()
        self.assertEqual(draft.status, "draft")
        self.assertIsNone(draft.reviewer)
        with self.assertRaisesRegex(ValidationError, "named human reviewer"):
            changed(draft, status="approved")
        approved = changed(draft, status="approved", reviewer="synthetic-fixture-reviewer")
        self.assertNotEqual(draft.id, approved.id)
        self.assertNotEqual(draft.id, changed(draft, revision=2).id)

    def test_training_example_rejects_mismatched_analysis_and_missing_provenance(self):
        draft = example()
        with self.assertRaisesRegex(ValidationError, "different analysis"):
            changed(draft, card=changed(draft.card, analysis_id="0" * 64))
        payload = draft.model_dump(exclude={"id", "provenance"})
        with self.assertRaises(ValidationError):
            TrainingExample.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
