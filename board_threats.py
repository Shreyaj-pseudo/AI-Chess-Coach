import chess
import chess.engine

def get_attacked_pieces(fen: str) -> dict:
    """
    Returns a dictionary of pieces under attack for the current player.
    Keys are square names (e.g., 'a1'), values are the piece on that square.
    """
    board = chess.Board(fen)
    attacked_pieces = {}
    
    # Determine the color of the current player
    current_player_color = board.turn

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == current_player_color:
            # Check if this piece is attacked by the opponent
            if board.is_attacked_by(not current_player_color, square):
                attacked_pieces[chess.square_name(square)] = str(piece)
    return attacked_pieces

def get_undefended_pieces(fen: str) -> dict:
    """
    Returns a dictionary of undefended pieces for the current player.
    Keys are square names (e.g., 'a1'), values are the piece on that square.
    """
    board = chess.Board(fen)
    undefended_pieces = {}
    current_player_color = board.turn

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == current_player_color:
            # Check if the piece is attacked by the opponent
            is_attacked = board.is_attacked_by(not current_player_color, square)
            
            # Count defenders
            defenders = 0
            for attacker_square in chess.SQUARES:
                attacker_piece = board.piece_at(attacker_square)
                if attacker_piece and attacker_piece.color == current_player_color:
                    if board.attacks(attacker_square, square):
                        defenders += 1
            
            if is_attacked and defenders == 0:
                undefended_pieces[chess.square_name(square)] = str(piece)
    return undefended_pieces

def get_underdefended_pieces(fen: str) -> dict:
    """
    Returns a dictionary of underdefended pieces for the current player.
    A piece is underdefended if it has fewer defenders than attackers.
    Keys are square names (e.g., 'a1'), values are the piece on that square.
    """
    board = chess.Board(fen)
    underdefended_pieces = {}
    current_player_color = board.turn

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == current_player_color:
            attackers_count = 0
            for attacker_square in chess.SQUARES:
                attacker_piece = board.piece_at(attacker_square)
                if attacker_piece and attacker_piece.color == (not current_player_color):
                    if board.attacks(attacker_square, square):
                        attackers_count += 1
            
            defenders_count = 0
            for defender_square in chess.SQUARES:
                defender_piece = board.piece_at(defender_square)
                if defender_piece and defender_piece.color == current_player_color:
                    if board.attacks(defender_square, square):
                        defenders_count += 1
            
            if attackers_count > 0 and defenders_count < attackers_count:
                underdefended_pieces[chess.square_name(square)] = str(piece)
    return underdefended_pieces

def get_fen_after_move(fen: str, move_uci: str) -> str:
    """
    Applies a move (in UCI format) to the given FEN and returns the new FEN.
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move in board.legal_moves:
        board.push(move)
        return board.fen()
    else:
        raise ValueError(f"Illegal move: {move_uci} for FEN: {fen}")

def get_fen_after_best_move(fen: str, stockfish_path: str, time_limit: float = 0.1) -> str:
    """
    Uses Stockfish to find the best move for the given FEN and returns the new FEN after applying it.
    """
    board = chess.Board(fen)
    try:
        engine = chess.engine.popen_uci(stockfish_path)
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        engine.quit()
        if result.move:
            board.push(result.move)
            return board.fen()
        else:
            raise RuntimeError("Stockfish did not return a move.")
    except Exception as e:
        raise RuntimeError(f"Error using Stockfish: {e}")

if __name__ == "__main__":
    # Example Usage:
    initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(f"Initial FEN: {initial_fen}")

    # Test get_attacked_pieces
    attacked = get_attacked_pieces(initial_fen)
    print(f"Attacked pieces (White): {attacked}") # Should be empty for initial FEN for white

    # Test with a FEN where white pieces are attacked (e.g., after black moves e5)
    fen_after_e5 = "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 2"
    attacked_after_e5 = get_attacked_pieces(fen_after_e5)
    print(f"Attacked pieces (White after e5): {attacked_after_e5}") # Should show white pieces attacked by black pawn on e5

    # Test get_undefended_pieces
    undefended = get_undefended_pieces(initial_fen)
    print(f"Undefended pieces (White): {undefended}") # Should be empty for initial FEN for white

    # Test get_underdefended_pieces
    underdefended = get_underdefended_pieces(initial_fen)
    print(f"Underdefended pieces (White): {underdefended}") # Should be empty for initial FEN for white

    # Test get_fen_after_move
    fen_after_e4 = get_fen_after_move(initial_fen, "e2e4")
    print(f"FEN after e4: {fen_after_e4}")

    # Test get_fen_after_best_move
    # You need to provide the correct path to your Stockfish executable
    stockfish_exe_path = "C:/Users/shrey/OneDrive/Documents/coding_shit/Vs_code/Chess_project/stockfish/stockfish-windows-x86-64-avx2.exe"
    try:
        fen_after_best = get_fen_after_best_move(initial_fen, stockfish_exe_path)
        print(f"FEN after Stockfish best move: {fen_after_best}")
    except RuntimeError as e:
        print(f"Could not get best move: {e}")
