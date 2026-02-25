import chess.pgn
import random
import io

def get_random_fen_and_move(pgn_file):
    # Read all games into a list
    games_raw = []
    with open(pgn_file, encoding="utf-8") as f:
        while True:
            start_offset = f.tell()  # remember where this game starts
            game = chess.pgn.read_game(f)
            if game is None:
                break
            # store the raw PGN text for this game
            f.seek(start_offset)
            raw_pgn = ""
            while True:
                line = f.readline()
                if not line.strip():  # empty line means moves start
                    raw_pgn += "\n"
                    while True:
                        move_line = f.readline()
                        if not move_line.strip():  # empty line after moves means end of game
                            break
                        raw_pgn += move_line
                    break
                raw_pgn += line
            games_raw.append((game, raw_pgn.strip()))

    # Pick random game
    chosen_game, chosen_pgn_text = random.choice(games_raw)

    # Extract positions
    board = chosen_game.board()
    positions = []
    for move in chosen_game.mainline_moves():
        fen_before = board.fen()
        positions.append((fen_before, move))
        board.push(move)

    # Pick random FEN + move
    fen, move = random.choice(positions)

    print("Full PGN:\n")
    print(chosen_pgn_text)
    print("\nRandom FEN:\n", fen)
    print("Played move:\n", move.uci())

if __name__ == "__main__":
    get_random_fen_and_move(r"C:\Users\shrey\OneDrive\Documents\coding_shit\Vs_code\Chess_project\data\lichess_db_standard_rated_2013-01.pgn\lichess_db_standard_rated_2013-01.pgn")
