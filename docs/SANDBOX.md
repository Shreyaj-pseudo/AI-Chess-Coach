# Browser testing sandbox

Open **http://127.0.0.1:8765/** on this computer while the local server is running. It uses the real Python services and installed Stockfish; it does not show canned engine results. It is a testing workbench ahead of the integrated T12 / production T14–T15 app, not completion of those tickets.

If the server has stopped, from the project folder run `.\.venv\python.exe -m scripts.sandbox`. Leave that process running while using the page; Ctrl+C stops a foreground server. No Conda activation or new packages are necessary with this explicit interpreter. The server binds only to 127.0.0.1; this URL is not a shareable public deployment and does not work from another device.

## What to test and what you should see

| Test | Browser actions | Expected evidence |
| --- | --- | --- |
| Actual engine | Click **Check engine** | Stockfish name, process reuse true and clean shutdown true. Engine discovery alone is not this live check. |
| Known error | Choose **Fool’s Mate**, review as **White**, choose **Ply 3 · g4**, click **Analyze this position** | Played line selected, **Losing mate in 1**, actual played move g4. Other engine moves/evaluations may vary. |
| Replay and facts | Click **Next** once, then once more | First: black to move, legal checks include **Qh4#**. Second: queen on h4, White in checkmate, no legal checks/captures. The evaluation belongs to the line’s original root, not a newly analyzed score at each replay frame. |
| Alternative evidence | Choose a **Candidate** line and step it | The board resets to the original root and then follows that candidate’s own legal continuation. The score, line and search provenance belong to that candidate. |
| Whole-game selection | Click **Find moments in the whole game** | Two White decisions scanned; g4 selected. f3 can receive a deep check if its provisional gap qualifies. Only one final moment is still possible if other deep evidence stays uncertain. Select a moment to load its deeper lines. Counts are actual, not a fixed target. |
| Learner perspective | Switch **Review as** to **Black**, select **Ply 4 · Qh4#**, analyze | **Winning mate in 1** for Black; one Next reaches checkmate. Switching sides refreshes available decisions and board orientation. |
| Opening lookup | Choose **Ruy Lopez**, use the suggested White ply 9, click **Opening context** | Separate masters/Lichess statuses, current-position opening name if supplied, sample counts, database moves and cache status. Not every source has a name/data. Missing credentials/network/quota must show unavailability or rate limit, not invented statistics. |
| Endgame lookup | Choose **Queen endgame**, White ply 1, click **Endgame tablebase** | Eligible queen-and-kings position, tablebase win evidence and move outcomes when the public API is available. Analyze the same position to replay Qg7#. |
| Ineligibility | Use the Fool’s Mate sample and click **Endgame tablebase** | Ineligible / unknown and engine fallback requested because the position has too many pieces. The flag reports a request; this sandbox does not automatically execute a fallback. |
| Invalid input | Replace the PGN with `1. e5 *`, click **Read game** | An explicit illegal-move error; no new fabricated analysis. Reload a sample to recover. |
| Your own game | Choose **Your own PGN**, paste a single game or select a .pgn file, choose learner color, read and analyze | Correct pre-move board for that learner decision. Review finds up to 15 moments without padding; partial failures are visible. Standard chess only, <=256 KiB PGN and <=200 plies. |
| Draft model | After analysis, inspect **Generate model draft** | Disabled with an explanation if model configuration is incomplete. When configured, clicking uses the T10 adapter and T11 grounded path: concrete structured observations are checked against the board/score, with at most one repair. Strategic interpretation remains qualified and unverified. No keys are entered into the browser. |
| Inspect output | Open **Inspect or download the result JSON** | Exact latest operation result, including source statuses/uncertainty, with a working JSON download. A new operation replaces this view. |

Opening, tablebase and hosted model buttons contact their respective services. PGN headers/comments do not go to the model; supplied position history/engine evidence does. Opening queries use the existing persistent SQLite cache; game/review/draft objects live only in this server's memory. One job and the current review's deep records are retained, not an indefinite history. Model settings are read from the existing server-side .env/shell. Reload the page after changing model configuration to refresh the button status. The browser does not establish training-label permission or human approval.

## Verification and limits

The local sandbox uses standard-library HTTP, a single background worker, and native browser HTML/CSS/JS; no new dependencies, hosted database, queue framework or CDN assets. Operations run one at a time. Waiting is indeterminate, not a fabricated percentage; whole-game search can take minutes. There is no per-job cancel button, login, cloud deployment, persistence/history UI or live test-runner UI. Reloading during an operation can leave the server busy until its bounded calls finish.

This loopback testing server is not hardened for public exposure. Do not change its bind address or put a public tunnel in front of it. Host/origin checks, JSON-only bounded POSTs, static-file allowlisting, no request logs and a restrictive content policy protect the intended local use; they are not production authentication. Production jobs/cancellation/server routing remain T13–T16.

Automated checks: `python -m unittest discover -s tests -p test_sandbox.py -v`, then the full offline suite. These use real loopback HTTP with mocked engine/model/explorer calls. Actual engine/API/browser smoke results are recorded separately in `artifacts/tasks/sandbox.md`.

Incomplete nonterminal played lines now receive one complete-position retry with a larger, capped budget. Recovered results retain the actual retry provenance. Persistent failures name their ply, stage and reason directly on the page. This can increase review time; it does not retry crashes or remove uncertainty.
