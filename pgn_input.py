"""Strict single-game ingestion; chess rules and game representation use python-chess."""

from dataclasses import dataclass
import re

import chess
import chess.pgn


MAX_PGN_BYTES = 256 * 1024
MAX_PLIES = 200
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
# read_game() skips unknown text; consume every input character at this boundary.
TOKEN = re.compile(
    r'\s+|\{[^}]*\}|;[^\r\n]*|^%[^\r\n]*|\[[^\r\n]*\]|'
    r'\d+\.(?:\.\.)?|[()*]|\$\d+|[!?]{1,2}|[^\s{}();\[\]$!?*]+',
    re.MULTILINE,
)
HEADER = re.compile(r'\[([A-Za-z0-9][A-Za-z0-9_+#=:-]*)[ \t]+"((?:\\.|[^"\\\r\n])*)"\]')
MOVE_NUMBER = re.compile(r'(\d+)\.(\.\.)?')


class PGNError(ValueError):
    """An actionable input error, never a silently truncated game."""


def _mainline(text):
    headers, tokens = {}, []
    depth = 0
    separated = ended = False
    offset = 0
    while offset < len(text):
        match = TOKEN.match(text, offset)
        if match is None:
            raise PGNError(f"Malformed PGN delimiter on line {text.count(chr(10), 0, offset) + 1}.")
        token = match.group()
        offset = match.end()
        if token.isspace():
            if tokens and depth == 0 and re.search(r'\n[ \t\r]*\n', token):
                separated = True
            continue
        if token.startswith(("{", ";", "%")):
            continue
        if token == ")":
            if depth == 0:
                raise PGNError("Unmatched closing variation parenthesis.")
            depth -= 1
            continue
        if token == "(":
            if not tokens or ended:
                raise PGNError("A variation must follow a mainline move, before the result.")
            depth += 1
            continue
        if depth:
            continue
        if token.startswith(("$", "!", "?")):
            continue
        if token.startswith("["):
            if tokens:
                raise PGNError("Upload exactly one game; headers cannot follow movetext.")
            header = HEADER.fullmatch(token)
            if header is None:
                raise PGNError("Malformed PGN header; use [Name \"value\"] on its own line.")
            name, value = header.groups()
            if name in headers:
                raise PGNError(f"Duplicate PGN header: {name}.")
            headers[name] = re.sub(r'\\(["\\])', r'\1', value)
            continue
        if ended or separated:
            raise PGNError("Upload exactly one game; extra movetext follows a result or blank game separator.")
        tokens.append(token)
        ended = token in RESULTS
    if depth:
        raise PGNError("Unclosed variation parenthesis.")
    if not tokens:
        raise PGNError("PGN is empty or contains no movetext/result.")
    return headers, tokens


def _initial_board(headers):
    if headers.get("Variant", "Standard").casefold() not in ("standard", "chess", "normal"):
        raise PGNError("Only standard chess is supported; chess variants are not accepted.")
    setup = headers.get("SetUp", "0")
    if setup not in ("0", "1") or ("FEN" in headers) != (setup == "1"):
        raise PGNError('A setup position requires both [SetUp "1"] and a valid [FEN "..."] header.')
    fen = headers.get("FEN", chess.STARTING_FEN)
    try:
        if len(fen.split()) != 6:
            raise ValueError("FEN requires six fields")
        board = chess.Board(fen)
        if not board.is_valid() or board.has_chess960_castling_rights():
            raise ValueError("invalid standard position")
    except ValueError:
        raise PGNError("Invalid initial FEN; provide a valid six-field standard-chess position.") from None
    if headers.get("Result", "*") not in RESULTS:
        raise PGNError("Invalid Result header.")
    return board


def parse_pgn(source: str | bytes) -> chess.pgn.Game:
    """Validate one UTF-8 game and return only its legally replayed mainline."""
    if not isinstance(source, (str, bytes)):
        raise PGNError("PGN must be UTF-8 text or bytes.")
    if len(source) > MAX_PGN_BYTES:
        raise PGNError("PGN exceeds the 256 KiB upload limit.")
    try:
        text = source.decode("utf-8-sig") if isinstance(source, bytes) else source
        if len(source if isinstance(source, bytes) else source.encode("utf-8")) > MAX_PGN_BYTES:
            raise PGNError("PGN exceeds the 256 KiB upload limit.")
    except UnicodeError:
        raise PGNError("PGN must be valid UTF-8; save or export the file as UTF-8.") from None
    headers, tokens = _mainline(text.removeprefix("\ufeff"))
    board = _initial_board(headers)
    game = chess.pgn.Game(headers)
    game.setup(board)
    node = game
    plies = 0
    awaiting_move = False
    for token in tokens:
        if token in RESULTS:
            if awaiting_move:
                raise PGNError("A move number must be followed by a move.")
            if "Result" in headers and headers["Result"] != token:
                raise PGNError("The Result header and movetext result disagree.")
            game.headers["Result"] = token
            continue
        number = MOVE_NUMBER.fullmatch(token)
        if number:
            if awaiting_move or number[1] != str(board.fullmove_number) or bool(number[2]) != (board.turn == chess.BLACK):
                raise PGNError(f"Move number/side does not match the position before ply {plies + 1}.")
            awaiting_move = True
            continue
        if plies >= MAX_PLIES:
            raise PGNError("PGN exceeds 200 plies (100 full moves from its starting position).")
        if board.is_game_over():
            raise PGNError(f"Move after a terminal position at ply {plies + 1}.")
        try:
            move = board.parse_san(token)
            if not move or move not in board.legal_moves:
                raise ValueError("not a legal standard-chess move")
            if ("x" in token) != board.is_capture(move):
                raise ValueError("capture marker does not match the board")
        except ValueError:
            raise PGNError(f"Invalid or illegal mainline move at ply {plies + 1}.") from None
        board.push(move)
        if (token.endswith("#") and not board.is_checkmate()) or (token.endswith("+") and not board.is_check()):
            raise PGNError(f"Incorrect check/mate annotation at ply {plies + 1}.")
        node = node.add_main_variation(move)
        plies += 1
        awaiting_move = False
    if awaiting_move:
        raise PGNError("A move number must be followed by a move.")
    outcome = board.outcome()
    if outcome and game.headers.get("Result", "*") not in ("*", outcome.result()):
        raise PGNError("The declared result contradicts the terminal board position.")
    return game


@dataclass(frozen=True)
class Decision:
    """A learner's pre-move snapshot; board copies retain their entire prior stack."""

    ply: int
    board: chess.Board
    played_move: chess.Move

    @property
    def fen(self):
        return self.board.fen()

    @property
    def san(self):
        return self.board.san(self.played_move)

    @property
    def uci(self):
        return self.played_move.uci()

    @property
    def pgn_prefix(self):
        prefix = chess.pgn.Game.from_board(self.board)
        prefix.headers = chess.pgn.Headers({key: prefix.headers[key] for key in ("SetUp", "FEN") if key in prefix.headers})
        return prefix.accept(chess.pgn.StringExporter(headers=True, comments=False, variations=False))


def learner_positions(game: chess.pgn.Game, learner_color: str) -> list[Decision]:
    """Select White/Black decisions from a game returned by parse_pgn()."""
    if not isinstance(learner_color, str) or learner_color.casefold() not in ("white", "black"):
        raise PGNError("Choose exactly one learner color: white or black.")
    color = learner_color.casefold() == "white"
    board = game.board()
    positions = []
    for ply, move in enumerate(game.mainline_moves(), start=1):
        if board.turn == color:
            positions.append(Decision(ply, board.copy(stack=True), move))
        board.push(move)
    return positions
