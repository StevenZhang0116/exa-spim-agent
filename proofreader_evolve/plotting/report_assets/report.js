"use strict";
(() => {
  const data = JSON.parse(document.getElementById("report-data").textContent);
  const $ = id => document.getElementById(id);
  const el = (tag, text, cls) => { const n = document.createElement(tag); if (text != null) n.textContent = String(text); if (cls) n.className = cls; return n; };
  const finite = v => typeof v === "number" && Number.isFinite(v);
  const fmt = (v, digits = 4) => finite(v) ? v.toFixed(digits) : "Not recorded / not measured";
  const count = v => finite(v) ? v.toLocaleString("en-US") : "Not recorded";
  const pretty = v => JSON.stringify(v, null, 2);
  const label = (box, name, value) => box.append(el("p", `${name}: ${value ?? "Not recorded"}`, "labelled"));
  const statusNames = {accepted:"Promoted", rejected:"Not promoted", research_incomplete:"Research assignment incomplete", agent_error:"Agent / SDK error", submission_error:"Submission error", evaluation_error:"Evaluation error", execution_error:"Execution error", training_error:"Training error", interrupted:"Interrupted", error:"Error", incomplete:"Incomplete", evaluated:"Measured", running:"Last recorded: running", completed:"Completed", failed:"Run failed", unknown:"Completion not recorded"};
  const statusName = v => statusNames[v] || v || "Not recorded";
  const reasons = {train_champion:"Overall TRAIN champion", train_specialist:"Complementary / slice specialist", near_best_alternative:"Near-best alternative", train_alternative:"TRAIN alternative / specialist", reference:"Protected reference", protected_reference:"Protected accepted component", promotion_followup:"Priority follow-up from the new accepted component", evidence_required:"Investigation after rounds without new evidence", reference_branch:"Scheduled exploration from the accepted component", plateau:"Plateau-triggered exploration", periodic_exploration:"Periodic exploration", introduce_parameters:"Explore a method with tunable parameters", parameter_tuning:"Numeric parameter tuning"};
  const reasonName = v => reasons[v] || v || "Not recorded";
  const failure = v => v === "interrupted" || v === "error" || String(v).endsWith("_error");
  const badge = (text, cls="") => el("span", text, `badge ${cls}`);
  const statusBadge = v => badge(statusName(v), failure(v) ? "error" : v);
  function disclosure(title, fill) {
    const d = el("details"), body = el("div"); d.append(el("summary", title), body);
    d.addEventListener("toggle", () => { if (d.open && !d.dataset.loaded) { d.dataset.loaded="1"; fill(body); } });
    return d;
  }
  function table(headers, rows) {
    const wrap=el("div",null,"scroll"), t=el("table"), head=el("tr"), body=el("tbody");
    headers.forEach(h=>head.append(el("th",h))); const th=el("thead"); th.append(head); t.append(th,body);
    rows.forEach(values=>{ const r=el("tr"); values.forEach(v=>{const c=el("td"); if(v instanceof Node)c.append(v);else c.textContent=v??"Not recorded"; r.append(c);}); body.append(r); });
    wrap.append(t); return wrap;
  }
  function jsonDetails(box, name, value) { if(value != null) box.append(disclosure(name, b=>b.append(el("pre",pretty(value))))); }
  function fileDetails(box, files) {
    if (!files?.length) {box.append(el("p","No saved code previews.","muted"));return;}
    const select=el("select"), area=el("div"); select.setAttribute("aria-label","Select a source or diff file");
    files.forEach((f,i)=>{const option=el("option",f.path);option.value=String(i);select.append(option);});
    const show=()=>{const f=files[Number(select.value)];area.replaceChildren();
      if(f.truncated)area.append(el("p","Embedded preview truncated. The complete file is in the run directory.","warning"));
      const pre=el("pre"); if(f.path.endsWith(".diff"))f.text.split("\n").forEach(line=>pre.append(el("span",line+"\n",line.startsWith("+")?"diff-add":line.startsWith("-")?"diff-del":"")));else pre.textContent=f.text;
      area.append(pre);}; select.addEventListener("change",show);box.append(select,area);show();
  }
  function metricsView(box, report, title) {
    box.append(el("h4",title));if(!report){box.append(el("p","Not measured / not recorded","muted"));return;}
    label(box,"Mean Precision@K",fmt(report.macro_precision));
    const headers=["Brain / kind","Pool candidates","Pool positives","Top-K hits","Effective K / requested K","Precision@K","Recall@K"];
    box.append(table(headers,Object.entries(report.cells||{}).map(([name,c])=>{
      const row=[name,count(c.pool_size),count(c.positives),count(c.tp),`${count(c.effective_k)} / ${count(c.requested_k)}`,fmt(c.precision),c.positives===0?"N/A (no pool positives)":fmt(c.recall)];
      return row;})));
    for(const [name,c] of Object.entries(report.cells||{}))if(c.local_context){
      const context=c.local_context;
      label(box,`${name} local geometry`,`${count(context.selected_rows)} / ${count(context.pool_rows)} rows; ${count(context.truncated_sites)} truncated sites; ${context.cached?"cached":"extracted"}`);
      jsonDetails(box,`${name} local feature configuration and timing`,context);
    }
    for(const [name,c] of Object.entries(report.cells||{}))if(c.image_context){
      const image=c.image_context, coverage=image.scoring_coverage;
      label(box,`${name} image inputs`,`${count(image.selected_rows)} / ${count(image.pool_rows)} candidates; ${count(coverage?.top_k_with_images)} of final Top K have images. Coverage is not evidence of benefit.`);
      jsonDetails(box,`${name} image selection, coverage and resource use`,image);
    }
  }
  const attempts = new Map(), owners = new Map();
  data.seeds.forEach(a=>attempts.set(a.experiment,a));
  data.generations.forEach(g=>g.attempts.forEach(a=>{attempts.set(a.experiment,a);owners.set(a.experiment,g);}));
  const promoted = new Set(data.generations.filter(g=>g.status==="accepted").map(g=>g.submitted_experiment).filter(Boolean));
  function matchingMember(a,pool) {return Object.values(pool?.kinds||{}).flatMap(k=>[k.reference_branch,...(k.candidates||[])].filter(Boolean)).find(c=>c.experiment===a.experiment || (a.component_sha256 && (c.component_sha256 || attempts.get(c.experiment)?.component_sha256)===a.component_sha256));}
  function idButton(id) {const b=el("button",id||"Not recorded");b.addEventListener("click",()=>showAttempt(id));return b;}
  function rankingView(box,delta) {
    if(!delta){box.append(el("p","Top-K changes were not recorded.","muted"));return;}
    box.append(table(["TRAIN brain / kind","Gained positives","Lost positives","Net gain","Top-K overlap","Same selection"],Object.entries(delta).map(([name,d])=>[
      name,count(d?.gained_positives),count(d?.lost_positives),count(d?.net_tp),count(d?.top_k_overlap),d?.same_top_k==null?"Not recorded":d.same_top_k?"Yes":"No"])));
  }
  function specialistView(box,profile) {
    if(!profile?.slices){box.append(el("p","Slice hit statistics were not saved in this record.","muted"));return;}
    const names={all:"All candidates",gap_le_25um:"Split representative gap <=25 um",gap_gt_25um:"Split representative gap >25 um",degree_3:"Merge node degree =3",degree_ge_4:"Merge node degree >=4"};
    box.append(table(["TRAIN brain / kind","Slice","Slice pool positives","Hits within global Top K","Slice recall"],Object.entries(profile.slices).map(([name,s])=>{
      const parts=name.split("/"),group=parts.pop();return [parts.join("/"),names[group]||group,count(s.positives),count(s.hits),s.positives===0?"N/A":fmt(s.recall)];})));
    box.append(el("p","All slices share the same global Top K; no additional K is allocated per slice.","muted"));
  }
  function attemptView(box,a,g) {
    box.append(el("h3",a.experiment),statusBadge(a.status)); if(a.cached)box.append(badge("Cached")); if(promoted.has(a.experiment))box.append(badge("Promoted","accepted"));
    const member=matchingMember(a,g?.pool_after || data.pool);if(member)box.append(badge(g?.pool_after?"Retained after generation":"Retained in current pool","retained"));
    label(box,"Agent hypothesis",a.hypothesis);label(box,"Strategy",a.strategy);label(box,"Search kind / mode / family",`${a.target_kind || "?"} / ${a.search_mode || "?"} / ${a.family || "?"}`);
    label(box,"Assigned search parent",a.search_parent);
    if(g?.research_status){const r=g.research_status;
      label(box,"Generation assignment",`${reasonName(g.search_reason)}; research ${r.status}; new measurements ${r.new_measurements}; reused results ${r.reused_results}`);
      if(r.investigation_required || r.followup_required)label(box,"Required work",`${r.requirement}; assigned-branch follow-up ${r.followup_required ? (r.followup_complete ? "complete" : "incomplete") : "not required"}`);
    }
    label(box,"Last measured / restored workspace checkpoint",a.lineage?.edit_base_experiment || "Not recorded (not inferred)");
    if(a.cached_from)label(box,"Cached measurement source",a.cached_from);
    if(a.lineage?.restored_from)label(box,"Most recent restore",a.lineage.restored_from);
    label(box,"Selection score for this kind (grouped out-of-fold P@K; in-sample P@K in legacy runs)",fmt(a.target_precision));
    if(finite(a.in_sample_precision))label(box,"In-sample TRAIN P@K for this kind (diagnostic, does not rank)",fmt(a.in_sample_precision));
    if(a.error)box.append(el("p",`${a.failure_stage || "Failure stage not recorded"}: ${a.error}`,"warning"));
    if(member){label(box,"Retention reason",reasonName(member.retention?.reason));
      label(box,"Compared with the TRAIN champion",`${count(member.vs_champion?.additional_positive_hits)} additional positives; ${count(member.vs_champion?.lost_positive_hits)} lost positives`);
      jsonDetails(box,"Retention comparisons",{retention:member.retention,vs_champion:member.vs_champion,vs_other_retained:member.vs_other_retained});}
    jsonDetails(box,"Parameters / model configuration",a.classifier || a.parameters);
    box.append(disclosure("Top-K positive changes versus the accepted parent policy",b=>rankingView(b,a.ranking_delta)));
    box.append(disclosure("Slice hits within the same global Top K",b=>specialistView(b,a.specialist_profile)));
    box.append(disclosure("Counts by TRAIN brain",b=>metricsView(b,a.train,"TRAIN")));
    box.append(disclosure("Source and diffs",b=>fileDetails(b,a.files)));
  }
  function showAttempt(id) {
    const box=$("selected-attempt");box.hidden=false;box.replaceChildren();const a=attempts.get(id);
    if(a)attemptView(box,a,owners.get(id));else box.append(el("h3",id),el("p","No complete attempt record was saved for this candidate. Changes and performance are not inferred."));
    box.scrollIntoView({behavior:"smooth",block:"start"});
  }
  function poolView(box,pool) {
    if(!pool){box.append(el("p","No candidate-pool snapshot was saved at this point.","muted"));return;}
    for(const [kind,item] of Object.entries(pool.kinds||{})) {
      box.append(el("h4",`${kind} · ${item.candidates.length} TRAIN branches`));
      if(item.reference_branch) {
        const c=item.reference_branch;
        box.append(el("p","Protected reference: the current accepted component. It has a separate slot and receives every third scheduled round of this kind.","labelled"));
        box.append(table(["Protected candidate","Method","Selection P@K (this kind)","In-sample P@K"],[[idButton(c.experiment),c.family,fmt(c.target_precision),fmt(c.in_sample_precision)]]));
        if(item.candidates.some(e=>e.component_sha256===c.component_sha256))box.append(el("p","This reference also appears in the TRAIN archive; both roles share one candidate.","muted"));
      }
      box.append(el("p",`TRAIN archive branches jointly cover ${count(item.retained_positive_hits)} positives; all historically retained TRAIN branches cover ${count(item.ever_retained_positive_hits)}. These are unions across candidates, not one scorer's Top-K hits. A reference outside that archive is excluded from these counts.`,"muted"));
      box.append(table(["Candidate","Method","Selection P@K (this kind)","In-sample P@K","Retention reason","Additional hits vs champion"],item.candidates.map(c=>[
        idButton(c.experiment),c.family,fmt(c.target_precision),fmt(c.in_sample_precision),reasonName(c.retention?.reason),count(c.vs_champion?.additional_positive_hits)])));
      jsonDetails(box,"Slice details and plateau counters",item);
    }
  }
  const svgns="http://www.w3.org/2000/svg";
  function svgEl(tag,attrs={},text) {const n=document.createElementNS(svgns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,String(v)));if(text!=null)n.textContent=String(text);return n;}
  function chart(scope,title) {
    const panel=el("div",null,"panel");panel.append(el("h3",title));
    const s=svgEl("svg",{viewBox:"0 0 590 265",class:"chart",role:"img","aria-label":title});
    const generations=[0,...data.generations.map(g=>g.generation)],last=Math.max(1,...generations);
    const x=n=>48+n/last*520,y=n=>225-n*185;
    [0,.25,.5,.75,1].forEach(v=>{s.append(svgEl("line",{x1:48,x2:570,y1:y(v),y2:y(v),stroke:"#e1e9ef"}),svgEl("text",{x:10,y:y(v)+4},v.toFixed(2)));});
    const ticks=generations.filter((n,i)=>i===0 || i===generations.length-1 || i%Math.max(1,Math.ceil(generations.length/8))===0);
    ticks.forEach(n=>s.append(svgEl("text",{x:x(n),y:249,"text-anchor":"middle"},n)));
    let current=data.seed[scope]?.macro_precision ?? null;
    const submitted=[],accepted=[{n:0,v:current}];
    for(const g of data.generations){submitted.push({n:g.generation,v:g.policy_match===false?null:g[scope]?.macro_precision ?? null});
      current=g.parent?.[scope]?.macro_precision ?? null;
      if(g.status==="accepted")current=g.policy_match===false?null:g[scope]?.macro_precision ?? null;
      accepted.push({n:g.generation,v:current});}
    const baseline=data.baseline[scope]?.macro_precision;
    if(finite(baseline))s.append(svgEl("line",{x1:48,x2:570,y1:y(baseline),y2:y(baseline),stroke:"#94a4b1","stroke-dasharray":"4 4"}));
    function draw(points,color,steps=false){let d="",previous=null;
      points.forEach(p=>{if(!finite(p.v)){previous=null;return;}d+=previous ? (steps?` H${x(p.n)} V${y(p.v)}`:` L${x(p.n)} ${y(p.v)}`):` M${x(p.n)} ${y(p.v)}`;previous=p;});
      s.append(svgEl("path",{d,fill:"none",stroke:color,"stroke-width":2}));
      points.filter(p=>finite(p.v)).forEach(p=>{const c=svgEl("circle",{cx:x(p.n),cy:y(p.v),r:4,fill:color,"aria-label":`Generation ${p.n}: ${fmt(p.v)}`});
        c.append(svgEl("title",{},`Generation ${p.n}: ${fmt(p.v)}`));s.append(c);});}
    draw(accepted,"#157a53",true);draw(submitted,"#126dba");panel.append(s);
    const legend=el("div",null,"legend");[["Submitted candidate (gaps = not measured)","#126dba"],["Accepted policy","#157a53"],["Frozen detector baseline","#94a4b1"]].forEach(([name,color])=>{const n=el("span",name);n.style.setProperty("--color",color);legend.append(n);});panel.append(legend);
    panel.append(el("p","Y: equally weighted mean Precision@K. X: generation. TRAIN and validation are shown separately.","muted"));return panel;
  }
  function branchGraph() {
    const box=$("branch-graph"),nodes=new Map();
    for(const [id,a] of attempts)nodes.set(id,{id,a,gen:owners.get(id)?.generation || 0});
    const edges=[];
    for(const n of [...nodes.values()]) {
      const parent=n.a.cached_from || n.a.lineage?.edit_base_experiment;
      if(parent && parent!==n.id){if(!nodes.has(parent))nodes.set(parent,{id:parent,a:null,gen:0});edges.push({from:parent,to:n.id,cached:!!n.a.cached_from});}
    }
    if(!nodes.size){box.append(el("p","No saved candidate nodes."));return;}
    const columns=[...new Set([...nodes.values()].map(n=>n.gen))].sort((a,b)=>a-b),rows=new Map();
    for(const n of nodes.values()){const index=rows.get(n.gen)||0;n.x=24+columns.indexOf(n.gen)*255;n.y=45+index*92;rows.set(n.gen,index+1);}
    const w=Math.max(560,columns.length*255+24),h=Math.max(160,Math.max(...rows.values())*92+60);
    const svg=svgEl("svg",{width:w,height:h,role:"img","aria-label":"Candidate exploration graph"});
    const defs=svgEl("defs"),marker=svgEl("marker",{id:"branch-arrow",viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:6,markerHeight:6,orient:"auto"});
    marker.append(svgEl("path",{d:"M0 0 L10 5 L0 10 Z",fill:"#92a9b8"}));defs.append(marker);svg.append(defs);
    columns.forEach((gen,i)=>svg.append(svgEl("text",{x:24+i*255,y:22,fill:"#596c7d"},gen?`Generation ${gen}`:"Seed / external references")));
    edges.forEach(e=>{const a=nodes.get(e.from),b=nodes.get(e.to);const same=a.x===b.x;
      const d=same?`M${a.x+215},${a.y+36} C${a.x+242},${a.y+36} ${b.x+242},${b.y+36} ${b.x+215},${b.y+36}`:`M${a.x+215},${a.y+36} C${a.x+236},${a.y+36} ${b.x-18},${b.y+36} ${b.x},${b.y+36}`;
      const p=svgEl("path",{d,fill:"none",stroke:e.cached?"#9694aa":"#92a9b8","stroke-width":1.5,"marker-end":"url(#branch-arrow)"});if(e.cached)p.setAttribute("stroke-dasharray","5 4");p.append(svgEl("title",{},`${e.from} → ${e.to}`));svg.append(p);});
    for(const n of nodes.values()) {
      const a=n.a,color=promoted.has(n.id)?"#157a53":a&&failure(a.status)?"#b93b3b":a&&matchingMember(a,data.pool)?"#7746bc":"#648499";
      const g=svgEl("g",{class:"branch-node",tabindex:0,role:"button","aria-label":n.id});g.append(svgEl("rect",{x:n.x,y:n.y,width:215,height:72,rx:7,fill:"#fff",stroke:color,"stroke-width":2}));
      g.append(svgEl("text",{x:n.x+10,y:n.y+20},n.id),
        svgEl("text",{x:n.x+10,y:n.y+41},a?`${a.target_kind || "?"} · selection P@K ${finite(a.target_precision)?a.target_precision.toFixed(3):"—"}`:"Record missing"),
        svgEl("text",{x:n.x+10,y:n.y+60},a?statusName(a.status):""));
      g.addEventListener("click",()=>showAttempt(n.id));g.addEventListener("keydown",e=>{if(e.key==="Enter")showAttempt(n.id);});svg.append(g);
    }box.append(svg);
  }
  $("run-title").textContent=data.run_name;
  $("run-meta").textContent=`${statusName(data.state.status)}${data.state.generation?` · Generation ${data.state.generation}`:""} · Report generated ${data.built_at} · Last status update ${data.state.updated_at || "Not recorded"}`;
  const total=data.totals,statItems=[
    ["Recorded generations / promoted",`${data.generations.length} / ${data.generations.filter(g=>g.status==="accepted").length}`],
    ["TRAIN brains",(data.manifest.train_brains||[]).join(", ")||"Not recorded"],
    ["Validation brains",(data.manifest.development_validation_brains||[]).join(", ")||"Not recorded"],
    ["K: merge / split",`${data.manifest.budgets?.merge??"?"} / ${data.manifest.budgets?.split??"?"}`],
    ["Promotion gate",data.manifest.promotion_gate?`${data.manifest.promotion_gate.mode||"margin"}${data.manifest.promotion_gate.bootstrap?.affects_decision?` (${data.manifest.promotion_gate.bootstrap.draws} draws, alpha ${data.manifest.promotion_gate.bootstrap.alpha})`:""}`:"Not recorded"],
    ["Kind schedule",data.manifest.kind_schedule?`${data.manifest.kind_schedule.mode}${data.manifest.kind_schedule.mode==="adaptive"?` (floor every ${data.manifest.kind_schedule.floor_every})`:""}`:"Not recorded"],
    [total.evaluations_complete?"Evaluation units used":"Recorded evaluation units (incomplete)",count(total.train_evaluations)],
    [total.cost_complete?"API cost":"Recorded API cost (incomplete)",`$${fmt(total.recorded_cost_usd,4)}`],
    [total.timing_complete?"Total generation duration":"Recorded generation duration (incomplete)",`${fmt(total.generation_seconds,1)} s`]];
  statItems.forEach(([name,value])=>{const s=el("div",null,"stat");s.append(el("span",name,"label"),el("span",value,"value"));$("stats").append(s);});
  const imageEvidence=data.generations.map(g=>g.image_evidence).filter(Boolean);
  if(imageEvidence.length){
    const sum=f=>imageEvidence.reduce((n,e)=>n+f(e),0);
    const access=sum(e=>e.volume_analyses_succeeded||0), scored=sum(e=>(e.image_scored_attempts||[]).length);
    const isolated=sum(e=>(e.paired_diagnostics||[]).filter(d=>d.status==="measured"&&d.image_evidence?.image_only_control).length);
    $("stats").append(el("p",`Image evidence (${imageEvidence.length}/${data.generations.length} generations recorded): ${access} successful 3D analyses; ${scored} image-scored attempts (including cached); ${isolated} image-only paired TRAIN diagnostics. Access and coverage do not establish gain.`,"muted"));
    jsonDetails($("stats"),"Image diagnostic outcomes",data.generations.filter(g=>g.image_evidence?.paired_diagnostics?.length).map(g=>({generation:g.generation,diagnostics:g.image_evidence.paired_diagnostics})));
  }
  if(data.state.reason)$("warnings").append(el("p",data.state.reason,"warning"));
  const research=data.generations.filter(g=>g.research_status);
  if(research.length){const countWhere=f=>research.filter(g=>f(g.research_status)).length;
    $("stats").append(el("p",`Research assignments: ${countWhere(r=>r.investigation_required)} investigations; ${countWhere(r=>r.followup_required)} priority follow-ups; ${countWhere(r=>!r.complete)} incomplete. Negative measured findings count as evidence, not as ranking gains.`,"muted"));
  }
  if(data.warnings.length)$("warnings").append(disclosure(`${data.warnings.length} artifact reading notices`,b=>data.warnings.forEach(w=>b.append(el("p",w,"warning")))));
  $("charts").append(chart("validation","Validation: submissions and accepted policies"),chart("train","TRAIN: submitted-policy performance"));
  const reports=$("metric-reports"),sel=el("select"),area=el("div");sel.setAttribute("aria-label","Select a measurement snapshot");
  const snapshots=[["Seed",data.seed],["Frozen detector baseline",data.baseline],["Final accepted",data.final]];
  data.generations.forEach(g=>snapshots.push([`Generation ${g.generation} · Submitted candidate`,g]));snapshots.forEach(([name],i)=>{const o=el("option",name);o.value=i;sel.append(o);});
  const changeMetrics=()=>{area.replaceChildren();const [name,r]=snapshots[Number(sel.value)];metricsView(area,r?.train,`${name} / TRAIN`);metricsView(area,r?.validation,`${name} / validation`);};sel.addEventListener("change",changeMetrics);reports.append(sel,area);changeMetrics();
  reports.append(el("p","Pool positives are GT-positive rows in the fixed candidate pool: sites for merge and fragment pairs for split. They do not count all GT errors.","muted"));
  branchGraph();poolView($("pool-content"),data.pool);
})();
