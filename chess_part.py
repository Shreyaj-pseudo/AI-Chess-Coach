import chess
import chess.engine
import chess.pgn
import io
import requests


class ChessAnalyzer:
    def __init__(self, stockfish_path = "C:/Users/shrey/OneDrive/Documents/coding_shit/Vs_code/Chess_project/stockfish/stockfish-windows-x86-64-avx2.exe"):
        self.stockfish_path = stockfish_path
        self.best = None
        self.score = None
        self.phase = None
        self.played = None
        self.fen = None
        self.pv = []
        self.score_played = None
        self.all_open_moves = []
        self.opening_data = {}
        self.turn = None
        self.white_pieces = []
        self.black_pieces = []


    def whose_move(self, fen):
        board = chess.Board(fen)
        return "White" if board.turn == chess.WHITE else "Black"
    
    def get_best_move_and_score(self, fen):
        board = chess.Board(fen)
        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            result = engine.analyse(board, chess.engine.Limit(time=0.1))
            move = result['pv'][0]
            best_move = board.san(move)
            self.best = best_move
            self.score = result['score']
            return best_move, result['score']

    def opening_moves(self, pgn_string):
        game = chess.pgn.read_game(io.StringIO(pgn_string))
        if not game:
            return []
        board = game.board()
        moves = []
        for move in game.mainline_moves():
            san = board.san(move)
            moves.append(san)
            board.push(move)
            if len(moves) == 10:
                break
        self.all_open_moves = moves
        return moves

    def fen_to_unicode_board_with_labels(self, fen):
        board = chess.Board(fen)
        board_lines = str(board).split('\n')
        labeled_lines = [f"{8 - i} {line}" for i, line in enumerate(board_lines)]
        footer = "  a b c d e f g h"
        return '\n'.join(labeled_lines + [footer])

    def infer_phase_from_fen(self, fen: str):
        board = chess.Board(fen)
        total_material = sum(piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
                             for piece in board.piece_map().values())
        if total_material >= 14:
            self.phase = "opening"
            return "opening"
        elif total_material >= 6:
            self.phase = "middlegame"
            return "middlegame"
        else:
            self.phase = "endgame"
            return "endgame"

    def score_from_move_played(self, fen, played_move):
        board = chess.Board(fen)
        board.push_san(played_move)
        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            result = engine.analyse(board, chess.engine.Limit(time=0.1))
            self.score_played = result["score"]
            return result["score"]

    def get_stockfish_pv(self, fen):
        board = chess.Board(fen)
        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=20), multipv=1)[0]
            score = info["score"]
            
            if score.is_mate():
                eval_str = f"Mate in {score.relative.mate()}"
            else:
                eval_str = f"{score.relative.cp / 100:.2f}"  # centipawns to pawns

            pv = info["pv"]
            pv_san = []
            board_copy = board.copy()
            for move in pv:
                pv_san.append(board_copy.san(move))
                board_copy.push(move)
            self.pv = pv_san
            return {
                "score": eval_str,
                "pv": pv_san
            }
    def get_opening_from_moves(self,PGN):
        """
        From a PGN string, extracts the opening name and the exact moves that matched a known opening.
        """
    
        full_moves = self.all_open_moves

        for cut in reversed(range(1, len(full_moves) + 1)):
            test_moves = full_moves[:cut]
            board = chess.Board()
            try:
                for move in test_moves:
                    board.push_san(move)
            except Exception as e:
                print(f"Invalid move in test sequence: {test_moves} -> {e}")
                continue

            fen = board.fen()
            url = "https://explorer.lichess.ovh/masters"
            params = {"fen": fen}

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                opening = data.get("opening")
                if opening:
                    return {
                        "opening_name": opening.get("name", "Unknown"),
                        "opening_moves": test_moves  # These are the ones matching theory
                    }

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                break

        return {
            "opening_name": "Unknown",
            "opening_moves": []
        }
    
    def get_piece_position(self, fen):
        board = chess.Board(fen)

        for square, piece in board.piece_map().items():
            if piece.color:  # White
                self.white_pieces.append(f"{piece.symbol().upper()}@{chess.square_name(square)}")
            else:  # Black
                self.black_pieces.append(f"{piece.symbol()}@{chess.square_name(square)}")



    def load_position(self, fen: str, played_move: str, pgn_string: str = None):
        self.fen = fen
        self.played = played_move

        # get whose move it is
        self.turn = self.whose_move(fen)

        # Get best move and score
        self.get_best_move_and_score(fen)

        # Get score from the played move
        self.score_from_move_played(fen, played_move)

        # Infer game phase
        self.infer_phase_from_fen(fen)

        # Get principal variation
        self.get_stockfish_pv(fen)

        #get piece position
        self.get_piece_position(fen)

        # Get opening moves from PGN (optional)
        if pgn_string:
            self.opening_moves(pgn_string)
        else:
            self.all_open_moves = []

        #get opening_data
        self.opening_data = self.get_opening_from_moves(pgn_string)

