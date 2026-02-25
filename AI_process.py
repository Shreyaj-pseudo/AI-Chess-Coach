from chess_part import ChessAnalyzer
from opening_agent import get_opening_analysis_response
from middlegame_agent import get_middlegame_analysis_response
from endgame_agent import get_endgame_analysis_response

def singular_position_analysis(PGN, FEN, played_move):
    chess_analyzer = ChessAnalyzer()
    chess_analyzer.load_position(FEN, played_move, PGN)
    if chess_analyzer.phase == "opening":
        return get_opening_analysis_response(PGN, FEN, played_move)
    elif chess_analyzer.phase == "endgame":
        return get_endgame_analysis_response(PGN, FEN, played_move)
    else:
        return get_middlegame_analysis_response(PGN, FEN, played_move)

if __name__ == "__main__":
    PGN = '''[Event "Rated Blitz game"]
[Site "https://lichess.org/fz2vgczf"]
[White "vivelafete"]
[Black "chinekeboy"]
[Result "0-1"]
[UTCDate "2013.01.01"]
[UTCTime "20:09:54"]
[WhiteElo "1829"]
[BlackElo "1948"]
[WhiteRatingDiff "-8"]
[BlackRatingDiff "+18"]
[ECO "A45"]
[Opening "Canard Opening"]
[TimeControl "240+0"]
[Termination "Time forfeit"]

1. d4 Nf6 2. f4 d6 3. Nd2 Nbd7 4. e4 e5 5. d5 exf4 6. Qf3 g6 7. Qxf4 Bg7 8. Bd3 O-O 9. Ngf3 Re8 10. O-O Nh5 11. Qe3 Nc5 12. b4 Nxd3 13. cxd3 Bxa1 14. g4 Ng7 15. Nc4 Rf8 16. b5 Bxg4 17. a4 Bxf3 18. Rxf3 Nh5 19. Ba3 Qh4 20. Rf1 Qg4+ 21. Kh1 Bg7 22. a5 f5 23. Rg1 Qf4 24. Bc1 Qxe3 25. Bxe3 fxe4 26. dxe4 Rf3 0-1'''

    FEN = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"