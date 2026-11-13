'use strict';
const $ = id => document.getElementById(id);
let samples = {}, positions = [], current = null, chosenLine = null, frameIndex = 0, flip = false, busy = false, modelReady = false, raw = null;
const reasonNames = {mate_swing:'Mate-status change',evaluation_loss:'Observed evaluation loss',mating_line:'Mating continuation',forcing_choice:'Forcing choice',strong_decision:'Strong quiet decision',opening_moment:'Opening moment'};
function node(tag, text, cls) { const e=document.createElement(tag); if(text!==undefined)e.textContent=text; if(cls)e.className=cls; return e; }
function status(text, kind='') { $('status').textContent=text; $('status').className=kind; }
function controls() {
  document.querySelectorAll('[data-run],.source-control').forEach(e=>e.disabled=busy);
  ['analyze','review','opening','tablebase'].forEach(id=>$(id).disabled=busy||!positions.length);
  $('decision').disabled=busy||!positions.length; $('draft').disabled=busy||!current||!modelReady;
  document.querySelectorAll('.moment').forEach(e=>e.disabled=busy);
}
async function api(path, options) { const response=await fetch(path,options); const data=await response.json(); if(!response.ok)throw Error(data.error||'The local server could not complete the request.'); return data; }
function setRaw(value) { raw=value; $('raw').textContent=JSON.stringify(value,null,2); $('download').disabled=false; }
async function run(action, extra={}) {
  if(busy)return; busy=true; controls();
  const progress={inspect:'Reading the game…',analyze:'Stockfish is analyzing candidates and the actual played move…',review:'Scanning learner decisions, then deep-checking shortlisted moments. This can take a couple of minutes…',opening:'Looking up exact-position opening statistics…',tablebase:'Checking endgame eligibility and tablebase evidence…',draft:'Requesting one unapproved model draft…',engine:'Checking two real engine searches and shutdown…',view:'Loading the selected moment’s evidence…'};
  status(progress[action], 'busy');
  try {
    const accepted=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,pgn:$('pgn').value,color:$('color').value,ply:Number($('decision').value),...extra})});
    let job;
    do { await new Promise(resolve=>setTimeout(resolve,400)); job=await api('/api/jobs/'+accepted.id); } while(job.status==='running');
    if(job.status==='failed')throw Error(job.error);
    setRaw(job.result); display(action,job.result); return job.result;
  } catch(error) { status(error.message+' No new result was produced.','error'); }
  finally { busy=false; controls(); }
}
function clearAnalysis() { current=null;chosenLine=null;frameIndex=0;$('line-buttons').replaceChildren(node('p','Analyze a position to compare its lines.','muted'));$('score').textContent='Not analyzed';$('continuation').textContent='';$('search-meta').textContent='Run an analysis for actual engine evidence.';$('warnings-section').hidden=true;$('previous').disabled=true;$('next').disabled=true; }
function invalidate() { positions=[];clearAnalysis();$('decision').replaceChildren(node('option','Read the updated game first'));$('moments-section').hidden=true;$('source-result').replaceChildren();$('facts').replaceChildren();$('board').removeAttribute('src');$('fen').textContent='';$('position-title').textContent='Read a game to continue';$('board-description').textContent='Board cleared after input change.';status('Game input changed. Read game to update the learner decisions.');controls(); }
function showFrame(frame) {
  const color=($('color').value==='white')!==flip?'white':'black';
  const query=new URLSearchParams({fen:frame.fen,color});if(frame.last)query.set('last',frame.last);
  $('board').src='/api/board?'+query;$('fen').textContent=frame.fen;
  const facts=frame.facts, get=(kind,subject)=>facts.find(f=>f.kind===kind&&f.subject===subject)?.value;
  const turn=get('turn','board'), ending=get('termination','board'), check=get('in_check',turn);
  const description=ending?ending.replaceAll('_',' '):(turn?turn[0].toUpperCase()+turn.slice(1)+' to move'+(check?' · in check':''):'Position');
  $('board-description').textContent=description;$('board').alt=description+'; '+frame.label;
  $('facts').replaceChildren();
  const rows=[['Side to move',turn||'Unknown'],['Position status',ending|| (check?'In check':'No check')],['Material points','White '+get('material_points','white')+' · Black '+get('material_points','black')],['Legal checks',facts.filter(f=>f.kind==='legal_check').map(f=>f.value).join(', ')||'None'],['Legal captures',facts.filter(f=>f.kind==='legal_capture').map(f=>f.value).join(', ')||'None']];
  const structure=facts.filter(f=>['absolute_pin','passed_pawn','isolated_pawn','doubled_pawn'].includes(f.kind));
  rows.push(['Pins and pawn structure',structure.map(f=>f.subject+': '+f.kind.replaceAll('_',' ')).join('; ')||'No supported flags on this board']);
  for(const [label,value] of rows)$('facts').append(node('dt',label),node('dd',value));
}
function scoreText(score) { if(score.kind==='mate')return (score.winning?'Winning':'Losing')+' mate in '+score.moves;if(score.kind==='cp')return (score.cp>0?'+':'')+(score.cp/100).toFixed(2)+' pawns';return 'Unavailable'; }
function replay() { if(!chosenLine)return;const frame=chosenLine.frames[frameIndex];showFrame(frame);$('step').textContent=frameIndex===0?'Before move':'Line ply '+frameIndex+' · '+frame.label;$('previous').disabled=frameIndex===0;$('next').disabled=frameIndex===chosenLine.frames.length-1; }
function chooseLine(candidate) {
  chosenLine=current.lines.find(l=>l.id===candidate.id);frameIndex=0;
  document.querySelectorAll('[data-line]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.line===candidate.id)));
  $('score-label').textContent='Evaluation for '+current.analysis.position.learner_color+' · '+candidate.san;$('score').textContent=scoreText(candidate.score);
  $('search-meta').textContent=candidate.search.time_limit_seconds+'s search budget · reported depth '+candidate.search.depth+' · '+candidate.search.engine;
  $('continuation').textContent=chosenLine.frames.slice(1).map(f=>f.label).join(' ');replay();
}
function showAnalysis(data) {
  current=data;clearAnalysisElements();const a=data.analysis;
  if(positions.some(p=>p.ply===a.position.ply))$('decision').value=a.position.ply;
  $('source-result').replaceChildren();
  $('position-title').textContent='Ply '+a.position.ply+' · played '+a.played.san;
  const unique=new Map(a.candidates.map((c,i)=>[c.id,{candidate:c,label:'Candidate '+(i+1)+' · '+c.san}]));
  unique.set(a.played.id,{candidate:a.played,label:'Played · '+a.played.san});
  for(const {candidate,label} of unique.values()) {const b=node('button',label);b.append(node('span',scoreText(candidate.score)));b.dataset.line=candidate.id;b.setAttribute('aria-pressed','false');b.onclick=()=>chooseLine(candidate);$('line-buttons').append(b);}
  $('warnings').replaceChildren(...a.uncertainty.map(t=>node('li',t)));$('warnings-section').hidden=!a.uncertainty.length;chooseLine(a.played);
}
function clearAnalysisElements() { $('line-buttons').replaceChildren(); }
function display(action,data) {
  if(action==='inspect') {
    clearAnalysis();positions=data.positions;$('decision').replaceChildren(...positions.map(p=>{const o=node('option','Ply '+p.ply+' · '+p.san);o.value=p.ply;return o;}));
    const preferred=$('sample').value==='mate'?($('color').value==='white'?3:4):$('sample').value==='castling'?($('color').value==='white'?9:14):null;
    if(positions.some(p=>p.ply===preferred))$('decision').value=preferred;
    const selected=positions.find(p=>p.ply===Number($('decision').value));showFrame(selected?.frame||data.frame);$('position-title').textContent=selected?'Ply '+selected.ply+' · before '+selected.san:'Game loaded';$('step').textContent='Before move';$('moments-section').hidden=true;$('source-result').replaceChildren();
    status('Game accepted: '+data.total_plies+' plies · '+positions.length+' '+$('color').value+' decisions. Choose a decision and analyze.');
  } else if(action==='analyze'||action==='view') { showAnalysis(data);status('Analysis ready. Replay the played move, then compare the candidate lines.'); }
  else if(action==='review') {
    const r=data.review;clearAnalysis();const selected=positions.find(p=>p.ply===Number($('decision').value));if(selected)showFrame(selected.frame);$('step').textContent='Before move';$('moments-section').hidden=false;$('moments').replaceChildren();
    $('review-summary').textContent=r.scanned.length+' scanned · '+r.deep_checks.length+' deep-checked · '+r.moments.length+' selected';
    for(const m of r.moments) {const a=r.deep_checks.find(c=>c.analysis.id===m.analysis_id).analysis;const b=node('button','Ply '+a.position.ply+' · '+a.played.san+' · '+m.reasons.map(t=>reasonNames[t]).join(', '),'moment');b.append(node('small',m.rationale.join(' ')));b.onclick=()=>run('view',{analysis_id:m.analysis_id});$('moments').append(b);}
    if(!r.moments.length)$('moments').append(node('p','No moments passed the deep selection checks. This does not certify error-free play.'));
    $('review-failures').textContent=r.partial?'Partial review: '+r.failures.map(f=>'Ply '+f.position.ply+' ('+f.stage+'): '+f.error).join(' '):'No position-evidence failures were recorded.';
    status('Game scan complete. Select a moment below to replay its deeper evidence.');
  } else if(action==='engine') {status(data.engine+' · process reused: '+data.reused+' · clean shutdown: '+data.closed);}
  else if(action==='opening') {
    const box=$('source-result');box.replaceChildren(node('h3','Opening context'));
    for(const s of data.opening.sources){const total=s.data?s.data.white+s.data.draws+s.data.black:0;box.append(node('p',s.source+' · '+s.status+(s.cache_hit?' · cached':'')+(s.data?' · '+total+' games':''),'source-label'));if(s.data?.opening)box.append(node('p',s.data.opening.eco+' · '+s.data.opening.name));if(s.error)box.append(node('p',s.error));if(s.data?.moves.length)box.append(node('p','Database moves: '+s.data.moves.map(m=>m.san+' ('+m.total_games+')').join(', ')));}
    status('Opening lookup finished. Check each source’s status and sample size.');
  } else if(action==='tablebase') {
    const t=data.tablebase,box=$('source-result');box.replaceChildren(node('h3','Endgame tablebase'),node('p','Status: '+t.status+' · learner outcome: '+t.outcome,'source-label'),node('p',t.reason),node('p','Engine fallback requested: '+t.engine_fallback));
    if(t.moves.length)box.append(node('p','Moves: '+t.moves.slice(0,12).map(m=>m.san+' ('+m.value.outcome+')').join(', ')));
    for(const n of t.notes)box.append(node('p',n,'hint'));status('Tablebase check finished. An ineligible or unavailable result is not a draw.');
  } else if(action==='draft') {
    const d=data.draft,c=d.card,provenance=d.attempts?.at(-1)||d,box=$('source-result');box.replaceChildren(node('h3','Unapproved model draft'),node('p','Serving model: '+provenance.returned_model+' · '+provenance.returned_provider,'source-label'));
    for(const [label,key] of [['Summary','summary'],['Played move','played_move_explanation'],['Recommendation','recommendation_explanation'],['Comparison','candidate_comparison'],['Lesson','lesson']])box.append(node('h3',label),node('p',c[key]));
    for(const note of d.uncertainty||[])box.append(node('p',note,'hint'));
    box.append(node('p','Structured concrete claims passed board/score checks. Strategic interpretation remains unverified; do not use this as an approved training label.','hint'));status('Grounded draft generated. Strategic interpretation still requires review.');
  }
}
$('inspect').onclick=()=>run('inspect');$('analyze').onclick=()=>run('analyze');$('review').onclick=()=>run('review');$('opening').onclick=()=>run('opening');$('tablebase').onclick=()=>run('tablebase');$('draft').onclick=()=>run('draft',{analysis_id:current.analysis.id});$('engine-test').onclick=()=>run('engine');
$('sample').onchange=()=>{if(busy)return;if(samples[$('sample').value]){$('pgn').value=samples[$('sample').value];$('color').value='white';invalidate();run('inspect');}else invalidate();};
$('pgn').oninput=()=>{ $('sample').value='custom';invalidate();};$('color').onchange=()=>{invalidate();run('inspect');};
$('decision').onchange=()=>{clearAnalysis();$('source-result').replaceChildren();$('position-title').textContent='Ply '+$('decision').value+' · not analyzed';const p=positions.find(p=>p.ply===Number($('decision').value));if(p)showFrame(p.frame);$('step').textContent='Before move';status('Decision changed. Analyze this position to refresh its evidence.');controls();};
$('file').onchange=async()=>{const f=$('file').files[0];if(!f)return;if(f.size>256*1024){status('File exceeds 256 KiB. Choose a smaller single-game PGN.','error');return;}$('pgn').value=await f.text();$('sample').value='custom';invalidate();await run('inspect');};
$('previous').onclick=()=>{if(frameIndex>0){frameIndex--;replay();}};$('next').onclick=()=>{if(chosenLine&&frameIndex<chosenLine.frames.length-1){frameIndex++;replay();}};
$('board-wrap').onkeydown=e=>{if(e.key==='ArrowRight'){e.preventDefault();$('next').click();}if(e.key==='ArrowLeft'){e.preventDefault();$('previous').click();}};
$('flip').onclick=()=>{if(!$('board').getAttribute('src'))return;flip=!flip;const image=new URL($('board').src);image.searchParams.set('color',image.searchParams.get('color')==='white'?'black':'white');$('board').src=image.href;};
$('download').onclick=()=>{if(!raw)return;const url=URL.createObjectURL(new Blob([JSON.stringify(raw,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download='chess-sandbox-result.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
async function boot(){try{const s=await api('/api/status');samples=s.samples;modelReady=s.model.configured;$('connection').textContent=s.engine_found?'Local backend · engine found':'Local backend · engine missing';$('model-status').textContent=modelReady?'Configured: '+s.model.backend+' / '+s.model.model+'. Drafts are unverified; generation may contact the configured provider.':'Model not configured. Analysis works without a model. Configure OpenRouter or Ollama on the server to enable draft generation.';$('pgn').value=samples.mate;await run('inspect');}catch(e){status('Cannot connect to the local backend. Restart the sandbox server.','error');}}
if(matchMedia('(max-width:700px)').matches)$('guide').open=false;
boot();
