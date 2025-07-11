import chess
import requests
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from chess_part import ChessAnalyzer



def get_opening_from_moves(PGN, chess_analyzer: ChessAnalyzer):
    """
    From a PGN string, extracts the opening name and the exact moves that matched a known opening.
    """
    chess_analyzer = ChessAnalyzer()
    full_moves = chess_analyzer.opening_moves(PGN)

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
        "eco": "N/A",
        "opening_moves": []
    }


