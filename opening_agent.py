from chess_part import ChessAnalyzer
import os
from dotenv import load_dotenv


def get_opening_theory_response(PGN, FEN, played_move,chess_analyzer):
    
    prompt = f"""
    You are an expert in chess opening theory.
You are analyzing a game that followed the opening: {chess_analyzer.opening_data["opening_name"]}. Search the web about the opening and analyse the {played_move} in the position {FEN} for your reference the places of the white pieces are {chess_analyzer.white_pieces} and the black pieces are {chess_analyzer.black_pieces}. The player played the following moves during the opening phase: {', '.join(chess_analyzer.opening_data["opening_moves"])}.Explain how these moves align with standard theory in the {chess_analyzer.opening_data["opening_name"]} opening. Mention any strong, weak aspects of the opening and tactics behind playing {chess_analyzer.opening_data["opening_name"]}. explain in only 1 paragraph. Keep the tone constructive and educational.
"""
    print(prompt)


def get_deviation_response(PGN, FEN, played_move, chess_analyzer):

    if played_move != chess_analyzer.best:
        prompt = f"""
You are an expert chess coach specializing in the opening phase which is the phase of the game. Your goal is to explain, in an intuitive way, why the engine's best move is superior to the move played by {chess_analyzer.turn}, based on the current board position and engine analysis. Here is the board position {FEN} for your reference the places of the white pieces are {chess_analyzer.white_pieces} and the black pieces are {chess_analyzer.black_pieces}. {chess_analyzer.turn} played {played_move}, which led to an evaluation of {chess_analyzer.score_played}. The best move according to the engine was {chess_analyzer.best}, evaluated at {chess_analyzer.score}. Engine's top line (Principal Variation): {chess_analyzer.pv}. Explain *only* why {chess_analyzer.best} is tactically or positionally stronger than {played_move} in this exact position. Focus on concrete, visible elements on the board such as threats, piece activity, king safety, pawn structure, or control of key squares, as suggested by the engine's principal variation. Your explanation should be intuitive and use clear chess terminology. use the stockfish principle variation possible instances of the board. Your answer should be a maximum of 1 paragraph. 
"""
    else:
        prompt = f"""You are an expert chess coach specializing in the opening phase which is the phase of the game. Your goal is to explain why the move {played_move} is strong and aligns with the engine's top recommendation, using concrete, intuitive reasoning based on the current board state. Here is the board position: {FEN}. you must decode this board position properly. {chess_analyzer.turn} played {played_move}, which matches the engine's best move. Evaluation: {chess_analyzer.score}. Engine's top line (Principal Variation): {chess_analyzer.pv}. Explain why {played_move} is the best move in this position. Base your explanation on the tactical or positional ideas visible in the current position (such as threats, piece activity, control of key squares, development, pawn structure, or king safety) and the ideas suggested by the engine's principal variation. Your explanation should be intuitive and phrased clearly using standard chess terminology. Limit your explanation to one paragraph. """
    
    
    print(prompt)



def get_opening_analysis_response(PGN, FEN, played_move):
    """
    Automatically selects the correct analysis (opening theory or deviation)
    based on whether the move is part of the opening line.
    """
    chess_analyzer = ChessAnalyzer()
    chess_analyzer.load_position(FEN, played_move, PGN)


    if played_move in chess_analyzer.opening_data["opening_moves"]:
        # Move is part of known opening theory
        print("opening theory")
        explanation = get_opening_theory_response(PGN, FEN, played_move, chess_analyzer)
    else:
        # Move is a deviation from theory
        explanation = get_deviation_response(PGN, FEN, played_move,chess_analyzer)

    return explanation

if __name__ == "__main__":

    PGN = '''[Event "Live Chess"]
[Site "Chess.com"]
[Date "2025.07.10"]
[Round "?"]
[White "shreyajpalchoudhury"]
[Black "ahmedelalaoui94"]
[Result "0-1"]
[TimeControl "600"]
[WhiteElo "1486"]
[BlackElo "1496"]
[Termination "ahmedelalaoui94 won by resignation"]
[ECO "A40"]
[EndTime "13:44:47 GMT+0000"]
[Link "https://www.chess.com/game/live/140540325116?move=4"]

1. d4 e6 2. Bf4 d5 3. e3 Bd6 4. Bg3 Nf6 5. Bd3 Nc6 6. Nf3 Ne7 7. c3 Ng6 8. Nbd2
Nh5 9. Bb1 Nxg3 10. hxg3 Qf6 11. Qb3 b6 12. Bd3 Bd7 13. O-O-O a5 14. e4 dxe4 15.
Nxe4 Qe7 16. Rh2 h6 17. Rdh1 O-O-O 18. Ba6+ Kb8 19. Nh4 Nxh4 20. Rxh4 Bc6 21.
Nxd6 Qxd6 22. Re4 Bxe4 23. Rg1 Bd5 24. Qc2 Bxa2 25. f4 Bd5 0-1'''
    FEN = "rnbqk2r/ppp2ppp/3bpn2/3p4/3P4/4P1B1/PPP2PPP/RN1QKBNR w KQkq - 3 5"
    played_move = "Bd3"
    print(get_opening_analysis_response(PGN, FEN, played_move))
