from opening_name import get_opening_from_moves
from chess_part import ChessAnalyzer
from notopenai import NotOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
CLIENT = NotOpenAI(api_key=api_key)


def get_opening_theory_response(PGN, FEN, played_move,chess_analyzer):
    

    prompt = f"""
    You are an expert in chess opening theory.
You are analyzing a game that followed the opening: {chess_analyzer.opening_data["opening_name"]}.
searchthe web about the opening and analyse the {played_move} in the position {FEN}.

The player played the following moves during the opening phase:
{', '.join(chess_analyzer.opening_data["opening_moves"])}

Explain how these moves align with standard theory in the {chess_analyzer.opening_data["opening_name"]} opening. Mention any strong, weak aspects of the opening and tactics behind playing {chess_analyzer.opening_data["opening_name"]}. 
explain in only 1 paragraph.
Keep the tone constructive and educational.
"""
    chat_completion = CLIENT.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

def get_deviation_response(PGN, FEN, played_move, chess_analyzer):

    if played_move != chess_analyzer.best:
        prompt = f"""
You are an expert chess coach. Your goal is to explain, in an intuitive way, why the engine's best move is superior to the move played by the player, based on the current board position and engine analysis.

Here is the board position:
{chess_analyzer.fen_to_unicode_board_with_labels(FEN)}

The player played {played_move}, which led to an evaluation of {chess_analyzer.score_played}.
The best move according to the engine was {chess_analyzer.best}, evaluated at {chess_analyzer.score}.

Engine's top line (Principal Variation): {chess_analyzer.pv}

Explain *only* why {chess_analyzer.best} is tactically or positionally stronger than {played_move} in this exact position. Focus on concrete, visible elements on the board such as threats, piece activity, king safety, pawn structure, or control of key squares, as suggested by the engine's principal variation. Do not invent tactics or plans that are not clearly present or provable from this board or the engine's analysis. Your explanation should be intuitive and use clear chess terminology. use the stockfish principle variation possible instances of the board

Your answer should be a maximum of 1 paragraph. 
"""
    else:
        prompt = f""" """
    chat_completion = CLIENT.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a chess coach. Your analysis must be strictly based on the provided board position, the two given moves, and the engine's analysis. Focus on visible tactical and positional elements. Do not speculate or invent scenarios not present on the board or not indicated by the engine's principal variation."},
        {"role": "user", "content": prompt}
    ]
)
    
    return chat_completion.choices[0].message.content


def analyze_opening_position(PGN, FEN, played_move):
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
    FEN = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"
    played_move = "e6"
    print(analyze_opening_position(PGN, FEN, played_move))
