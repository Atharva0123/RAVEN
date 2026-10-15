"use client";

import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";

type Health = { status?: string; service?: string; models_loaded?: boolean; temporal_fusion_loaded?: boolean; temporal_fusion_status?: string; temporal_fusion_eligible_for_scoring?: boolean; synthetic_demo_checkpoint?: boolean; queue_depth?: number; ws_url?: string };
type EventMessage = { type?: string; job_id?: string; status?: string; progress?: number; completed_frames?: number; summary?: ResultSummary; alert_event?: { level?: string; reason?: string; signal_label?: string; operator_review_required?: boolean; confirmed_threat?: boolean; track_id?: string | null; risk_score?: number; threats?: string[]; timestamp_seconds?: number; threat_confidence?: number | null; model_validation_status?: string; demo_label?: string }; error?: string; [key: string]: unknown };
type TrackSummary = {
  track_id: string; display_id?: string; first_seen_seconds?: number; last_seen_seconds?: number; frames_observed?: number;
  max_risk_score?: number; needs_review?: boolean; alert_status?: string; identity_source?: string;
  reid_status?: string; reid_similarity?: number | null; reid_method?: string; posture?: string | null;
  anomaly_score?: number | null; anomaly_method?: string; anomaly_model_available?: boolean;
  demo_label?: string | null; model_validation_status?: string | null; signal_label?: string | null;
  detected_objects?: Record<string, number>; evidence?: Array<Record<string, unknown>>; alert_count?: number;
  review_frames?: number; alert_frames?: number; review_reasons?: string[];
  trajectory?: Array<{ timestamp?: number; center?: number[]; bbox?: number[] }>;
  merged_tracker_ids?: string[];
  identity_timeline?: Array<{ timestamp?: number; tracker_id?: string; canonical_track_id?: string; reid_status?: string; similarity?: number | null }>;
};
type Capabilities = {
  threat_detection?: { model_loaded?: boolean; model_type?: string; supported_labels?: string[]; unsupported_labels?: string[]; firearm_classes_supported?: string[]; object_detection_only?: boolean; weapon_use_action_detection?: boolean; detection_note?: string | null; open_vocabulary?: boolean };
  confidence_fusion?: { available?: boolean; training_data?: string; scope?: string; validation_status?: string };
  anomaly_detection?: { available?: boolean; reason?: string; method?: string; synthetic_demo?: boolean; synthetic_action_demo?: boolean; action_model_available?: boolean; action_checkpoint?: string; action_model_error?: string };
  appearance_reid?: { available?: boolean; method?: string; similarity_threshold?: number; identifies_person_by_name?: boolean };
  thermal_face_landmarks?: { available?: boolean; enabled?: boolean; model?: string; reason?: string };
};
type SummaryEvent = {
  event_id?: string; timestamp_seconds?: number; track_id?: string | null; display_id?: string;
  level?: string; reason?: string; risk_score?: number; threat_confidence?: number | null;
  thresholds?: Record<string, number>; threats?: string[]; model_validation_status?: string;
  demo_label?: string; signal_label?: string; operator_review_required?: boolean; confirmed_threat?: boolean;
};
type ResultSummary = {
  processed_frames?: number; duration_seconds?: number; processing_fps?: number;
  inference_fps?: number; inference_seconds?: number; export_seconds?: number;
  compute_device?: string;
  stage_profile_ms?: Record<string, number>;
  max_concurrent_tracks?: number; unique_tracks?: number; event_count?: number;
  unique_tracker_ids?: number;
  events?: SummaryEvent[];
  tracks?: TrackSummary[]; threat_detections?: Array<{ label: string; count?: number; max_confidence?: number }>;
  alert_policy?: { warning_threshold?: number; alert_threshold?: number; threat_confidence_threshold?: number };
  no_alert_explanation?: string | null; capabilities?: Capabilities;
  show_trajectories?: boolean;
  processing_errors?: number; processing_mode?: string; truncated?: boolean;
  width?: number; height?: number; fps?: number; available_models?: Record<string, boolean>;
  thermal_note?: string;
  input_kind?: string;
};
type VideoJob = {
  job_id: string; status?: string; phase?: string; progress?: number; completed_frames?: number; total_frames?: number;
  summary?: ResultSummary; result?: { summary?: ResultSummary; rgb_artifact_url?: string; thermal_artifact_url?: string };
  error?: string;
};

function getTrackLevel(track: TrackSummary, objectThreshold: number, linkedEvents: SummaryEvent[] = []): "LOW" | "REVIEW" | "ALERT" {
  const hasAlertObject = Object.values(track.detected_objects ?? {})
    .some((confidence) => confidence >= objectThreshold);
  const hasAlertEvent = linkedEvents.some((event) => getEventLevel(event, objectThreshold) === "ALERT");
  if (hasAlertObject || hasAlertEvent) return "ALERT";
  if (track.needs_review || track.alert_status === "review" || track.alert_status === "warning" ||
      linkedEvents.some((event) => getEventLevel(event, objectThreshold) === "REVIEW")) return "REVIEW";
  return "LOW";
}

function getEventLevel(event: SummaryEvent, objectThreshold: number): "REVIEW" | "ALERT" {
  const threshold = event.thresholds?.threat_confidence ?? objectThreshold;
  const hasQualifiedObject = Boolean(event.threats?.length) &&
    (event.threat_confidence ?? 0) >= threshold;
  return hasQualifiedObject ? "ALERT" : "REVIEW";
}

function levelPriority(level?: string): number {
  switch ((level ?? "LOW").toUpperCase()) {
    case "ALERT": return 100;
    case "REVIEW": return 50;
    default: return 0;
  }
}
type UploadResult = { filename: string; size_bytes: number; video_id: string; job_id: string; status: string };

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [events, setEvents] = useState<EventMessage[]>([]);
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState<VideoJob | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadPhase, setUploadPhase] = useState<"idle" | "uploading" | "processing">("idle");
  const [uploadProgress, setUploadProgress] = useState(0);
  const uploadLock = useRef(false);
  const [socketState, setSocketState] = useState("connecting");
  const [message, setMessage] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceName, setSourceName] = useState("");
  const [inputKind, setInputKind] = useState<"rgb" | "thermal-palette">("rgb");
  const [showTrajectories, setShowTrajectories] = useState(true);
  const summary = job?.summary ?? job?.result?.summary;
  const status = (job?.status ?? "").toLowerCase();
  const progress = job?.progress != null ? Math.round(job.progress * 100) : 0;
  const finished = status === "completed" || status === "success";
  const hasJob = Boolean(jobId);
  const busy = uploading || uploadPhase !== "idle" || (hasJob && !finished && status !== "failed");

  useEffect(() => {
    if (finished || status === "failed") {
      uploadLock.current = false;
      setUploading(false);
      setUploadPhase("idle");
    }
  }, [finished, status]);

  useEffect(() => {
    const queryJob = new URLSearchParams(window.location.search).get("job");
    const savedJob = queryJob || window.localStorage.getItem("raven_last_video_job");
    if (savedJob) setJobId(savedJob);
  }, []);

  useEffect(() => {
    let alive = true;
    let socket: WebSocket | undefined;
    let reconnectTimer: number | undefined;
    const wsUrl = health?.ws_url;
    const refreshHealth = () => fetch("/api/health", { cache: "no-store" }).then((r) => r.json()).then((data: Health) => alive && setHealth(data)).catch(() => alive && setHealth(null));
    void refreshHealth();
    const healthTimer = window.setInterval(refreshHealth, 5000);
    if (!wsUrl) return () => { alive = false; if (healthTimer) window.clearInterval(healthTimer); };
    const connect = () => {
      if (!alive) return;
      const eventUrl = new URL(wsUrl);
      if (jobId) eventUrl.searchParams.set("job_id", jobId);
      socket = new WebSocket(eventUrl.toString());
      socket.onopen = () => alive && setSocketState("live");
      socket.onmessage = (event) => {
        try {
          const item = JSON.parse(event.data) as EventMessage;
          setEvents((current) => [item, ...current].slice(0, 20));
          if (item.job_id && item.job_id === jobId) setJob((current) => ({ ...current, ...item } as VideoJob));
        } catch { /* Ignore malformed status messages. */ }
      };
      socket.onclose = () => { if (alive) { setSocketState("reconnecting"); reconnectTimer = window.setTimeout(connect, 2000); } };
      socket.onerror = () => socket?.close();
    };
    connect();
    return () => { alive = false; if (reconnectTimer) window.clearTimeout(reconnectTimer); if (healthTimer) window.clearInterval(healthTimer); socket?.close(); };
  }, [jobId, health?.ws_url]);

  useEffect(() => {
    if (!jobId || finished || status === "failed") return;
    let alive = true;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
        const data = await response.json() as VideoJob & { detail?: string };
        if (!response.ok && response.status === 404) {
          if (alive) setJob({ job_id: jobId, status: "failed", error: data.detail ?? "This local job is no longer available." });
          return;
        }
        if (!response.ok) throw new Error(data.detail ?? "Could not read video analysis status");
        if (!alive) return;
        setMessage("");
        setJob(data);
        if (!(["completed", "success", "failed"].includes((data.status ?? "").toLowerCase()))) timer = window.setTimeout(poll, 1000);
      } catch (error) {
        if (alive) {
          setMessage(error instanceof Error ? error.message : "Could not read video analysis status");
          timer = window.setTimeout(poll, 2500);
        }
      }
    };
    void poll();
    return () => { alive = false; if (timer) window.clearTimeout(timer); };
  }, [jobId, finished, status]);

  function selectVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    if (!file) return;
    setSourceUrl((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(file); });
    setSourceName(file.name);
  }

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (uploadLock.current) return;
    uploadLock.current = true;
    const fields = new FormData(event.currentTarget);
    const file = fields.get("video");
    if (!(file instanceof File)) { uploadLock.current = false; return; }
    setUploading(true); setUploadPhase("uploading"); setUploadProgress(0); setMessage(""); setJob(null); setJobId("");
    const form = new FormData(); form.set("file", file); form.set("processing_mode", String(fields.get("processing_mode") ?? "Performance")); form.set("input_kind", String(fields.get("input_kind") ?? "rgb")); form.set("show_trajectories", String(fields.get("show_trajectories") === "on"));
    try {
      const result = await new Promise<UploadResult>((resolve, reject) => {
        const xhr = new XMLHttpRequest(); xhr.open("POST", "/api/videos");
        xhr.setRequestHeader("Idempotency-Key", crypto.randomUUID());
        xhr.upload.onprogress = (progressEvent) => { if (progressEvent.lengthComputable) setUploadProgress(Math.round(progressEvent.loaded / progressEvent.total * 100)); };
        xhr.onload = () => {
          let body: UploadResult & { detail?: string };
          try { body = JSON.parse(xhr.responseText) as UploadResult & { detail?: string }; }
          catch { reject(new Error("The video service returned an invalid response")); return; }
          if (xhr.status >= 200 && xhr.status < 300) resolve(body);
          else reject(new Error(body.detail ?? "Upload failed"));
        };
        xhr.onerror = () => reject(new Error("Video upload service is unavailable")); xhr.send(form);
      });
      window.localStorage.setItem("raven_last_video_job", result.job_id);
      setJobId(result.job_id);
      setJob({ job_id: result.job_id, status: result.status, progress: 0 });
      setUploadPhase("processing");
      setMessage(`${result.filename} uploaded · ${(result.size_bytes / (1024 * 1024)).toFixed(1)} MB · analysis queued`);
    } catch (error) {
      uploadLock.current = false;
      setUploading(false); setUploadPhase("idle");
      setMessage(error instanceof Error ? error.message : "Upload failed");
    }
  }

  const metric = (label: string, value: string | number) => <article className="metric" key={label}><span>{label}</span><strong>{value}</strong></article>;

  return <main className="shell">
    <header className="topbar"><div><p className="eyebrow">RESEARCH PROTOTYPE · FEASIBILITY STUDY</p><h1>RAVEN</h1><p className="muted">Multi-Modal Quality-Aware Sensor Fusion & Temporal Event Ranking</p></div><span className={`status ${health?.status === "ready" ? "up" : !health ? "" : "down"}`}>{health?.status === "ready" ? "API ONLINE" : !health ? "CHECKING API" : "API OFFLINE"}</span></header>

    <section className="cards">
      <article className="card"><p className="eyebrow">INFERENCE SERVICE</p><h2>{health?.service ?? "Waiting for API"}</h2><p className="muted">{health?.synthetic_demo_checkpoint ? "Synthetic temporal baseline loaded for architecture checks only · excluded from scoring" : health?.temporal_fusion_status === "checkpoint_provenance_unknown_scoring_disabled" ? "Temporal checkpoint provenance unknown · scores disabled" : `Models ${health ? (health.models_loaded ? "loaded" : "load on first analysis") : "—"}`}</p></article>
      <article className="card"><p className="eyebrow">VIDEO QUEUE</p><h2>{job ? status.toUpperCase() : "READY"}</h2><p className="muted">{job?.completed_frames != null ? `${job.completed_frames} of ${job.total_frames ?? "?"} frames` : "One video is analyzed at a time locally"}</p></article>
      <article className="card"><p className="eyebrow">REAL-TIME UPDATES</p><h2>{socketState === "live" ? "CONNECTED" : socketState.toUpperCase()}</h2><p className="muted">WebSocket status stream</p></article>
    </section>

    <section className="card upload-card"><p className="eyebrow">VIDEO INGESTION</p><h2>Upload and analyze</h2><p className="muted">Results include person tracks, weapon detections, posture estimates, and review alerts. {inputKind === "rgb" ? "RGB-only clips also get a display-only thermal-style visualization." : "Thermal palette clips use the thermal person-detector setting; palette values are not calibrated temperatures."}</p>
      <form onSubmit={upload}>
        <label className="file-field">Video file<input name="video" type="file" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska" onChange={selectVideo} required /></label>
        <label className="mode-field">Input feed<select name="input_kind" value={inputKind} onChange={(event) => setInputKind(event.currentTarget.value as "rgb" | "thermal-palette")}><option value="rgb">RGB video</option><option value="thermal-palette">Thermal palette video</option></select></label>
        <label className="mode-field">Processing mode<select name="processing_mode" defaultValue="Performance"><option>Performance</option><option>Balanced</option><option>Accuracy</option></select></label>
        <label className="trajectory-toggle"><input name="show_trajectories" type="checkbox" checked={showTrajectories} onChange={(event) => setShowTrajectories(event.currentTarget.checked)} /><span>Show Trajectories in output video</span></label>
        <button disabled={busy || !health || health.status !== "ready"}>{uploadPhase === "uploading" ? `Uploading ${uploadProgress}%…` : uploadPhase === "processing" || busy ? "Analyzing…" : "Upload and analyze"}</button>
      </form>
      <p className="muted small mode-help"><b>Performance (default):</b> checks objects every other frame to speed up long clips; very brief objects may be missed. <b>Balanced:</b> checks every frame and estimates pose every other frame. <b>Accuracy:</b> runs both at every frame.</p>
      {uploadPhase === "uploading" && <div className="upload-progress" role="status" aria-live="polite"><div className="progress-copy"><span>Uploading video</span><b>{uploadProgress}%</b></div><progress max="100" value={uploadProgress} /></div>}
      {(uploadPhase === "processing" || (busy && hasJob)) && <div className="upload-progress processing-progress" role="status" aria-live="polite"><div className="progress-copy"><span>Analyzing Video Pipeline…</span><b>{status === "queued" ? "QUEUED" : status === "processing" ? `${progress}%` : "STARTING"}</b></div><div className="indeterminate-track"><span /></div></div>}
      {message && <p className="notice" role="status">{message}</p>}
    </section>

    {(jobId || sourceUrl) && <section className="card result-section" aria-live="polite">
      <div className="section-heading"><div><p className="eyebrow">ANALYSIS RESULT</p><h2>{!hasJob ? "Ready to analyze" : status === "failed" ? "Analysis failed" : finished ? "Video analysis complete" : "Processing your video"}</h2></div>{hasJob && <span className={`status ${finished ? "up" : ""}`}>{status || "QUEUED"}</span>}</div>
      {!hasJob && <p className="muted">Choose a mode and upload the selected video to start. Source playback is ready below.</p>}
      {hasJob && !finished && status !== "failed" && <div className="progress-wrap"><div className="progress-copy"><span>{status === "queued" ? "Waiting for an analysis worker" : `${job?.phase ?? "Analyzing video"}. Progress refreshes automatically; reloading reconnects to this job.`}</span><b>{progress}%</b></div><progress max="100" value={progress} />{job?.completed_frames != null && <p className="muted">Frame {job.completed_frames}{job.total_frames ? ` of ${job.total_frames}` : ""}</p>}</div>}
      {status === "failed" && <p className="error-box">{job?.error ?? "The video could not be processed."}</p>}
      <div className="video-grid">
        {sourceUrl && <article className="video-panel"><div className="video-title"><div><span className="eyebrow">SOURCE</span><h3>{sourceName}</h3></div></div><video src={sourceUrl} controls playsInline preload="metadata" /></article>}
        <article className="video-panel"><div className="video-title"><div><span className="eyebrow">{summary?.input_kind === "thermal-palette" ? "THERMAL INPUT" : "RGB FEED"}</span><h3>Person and weapon detections</h3></div></div>{finished && jobId ? <video src={`/api/jobs/${encodeURIComponent(jobId)}/artifacts/rgb`} controls playsInline preload="metadata" /> : <div className="video-placeholder">The annotated detection result will appear here.</div>}</article>
        <article className="video-panel"><div className="video-title"><div><span className="eyebrow">{summary?.input_kind === "thermal-palette" ? "THERMAL PALETTE FEED" : "ESTIMATED THERMAL-STYLE FEED"}</span><h3>{summary?.input_kind === "thermal-palette" ? "Thermal camera detections" : "Derived from RGB appearance"}</h3></div></div>{finished && jobId ? <video src={`/api/jobs/${encodeURIComponent(jobId)}/artifacts/estimated-thermal`} controls playsInline preload="metadata" /> : <div className="video-placeholder">The {summary?.input_kind === "thermal-palette" ? "thermal-palette" : "estimated thermal-style"} result will appear here.</div>}<p className="muted small">Thermal palette video has no calibrated temperature values.</p></article>
      </div>
      {finished && summary && <>
        {(() => {
          const tracks = summary.tracks ?? [];
          const summaryEvents = summary.events ?? [];
          const threatConfidenceThreshold = summary.alert_policy?.threat_confidence_threshold ?? 0.35;
          const linkedEventsByTrack = tracks.map((track) => summaryEvents.filter((event) =>
            event.track_id != null && String(event.track_id) === String(track.track_id)));
          const trackLevels = tracks.map((track, index) =>
            getTrackLevel(track, threatConfidenceThreshold, linkedEventsByTrack[index]));
          const reviewCount = trackLevels.filter((level) => level === "REVIEW").length;
          const alertTrackCount = trackLevels.filter((level) => level === "ALERT").length;
          const trackIds = new Set(tracks.map((track) => String(track.track_id)));
          const orphanReviewEvents = summaryEvents.filter((event) =>
            getEventLevel(event, threatConfidenceThreshold) === "REVIEW" &&
            (event.track_id == null || !trackIds.has(String(event.track_id)))).length;
          const peakPriority = Math.max(0, ...trackLevels.map(levelPriority),
            ...summaryEvents.map((event) => levelPriority(getEventLevel(event, threatConfidenceThreshold))));
          const threat = summary.capabilities?.threat_detection;
          const anomaly = summary.capabilities?.anomaly_detection;
          const reid = summary.capabilities?.appearance_reid;
          const trajectoryOverlay = summary.show_trajectories ?? showTrajectories;
          const firearmClasses = threat?.firearm_classes_supported ?? [];
          return <>
        <div className="metrics-grid">
          {metric("FRAMES ANALYZED", summary.processed_frames ?? 0)}{metric("VIDEO DURATION", `${(summary.duration_seconds ?? 0).toFixed(1)} s`)}{metric("TRACK SEGMENTS", summary.unique_tracker_ids ?? summary.unique_tracks ?? 0)}{metric("UNIQUE PEOPLE (EST.)", summary.unique_tracks ?? 0)}{metric("TRACKER IDS MERGED", summary.unique_tracker_ids != null ? Math.max(0, summary.unique_tracker_ids - (summary.unique_tracks ?? 0)) : 0)}{metric("REVIEW / ALERT EVENTS", summary.event_count ?? 0)}
        </div>
        <div className="detail-grid">
          <article className="subcard"><p className="eyebrow">PIPELINE STATUS</p><h3>{summary.processing_mode ?? "Performance"} analysis · {summary.compute_device ?? "device unknown"}</h3><ul className="model-list">{Object.entries(summary.available_models ?? {}).map(([name, available]) => <li key={name}><span>{name.replaceAll("_", " ")}</span><b className={available ? "available" : "unavailable"}>{available ? "AVAILABLE" : "NOT CONFIGURED"}</b></li>)}</ul>{summary.processing_errors ? <p className="muted small">{summary.processing_errors} frame errors{summary.truncated ? " · output ended before the final source frame" : ""}</p> : summary.truncated ? <p className="muted small">Output ended before the final source frame.</p> : null}</article>
          <article className="subcard"><p className="eyebrow">REVIEW AND ALERT EVENTS</p><h3>{summary.event_count ?? 0} generated</h3>{summary.events?.length ? <ol className="event-list">{summary.events.map((item, index) => <li key={item.event_id ?? index}><time>{(item.timestamp_seconds ?? 0).toFixed(1)}s</time><span><b>{getEventLevel(item, threatConfidenceThreshold)}</b>{item.signal_label ? ` · ${item.signal_label}` : " · PROTOTYPE_REVIEW_CUE"}{item.display_id ? ` · ${item.display_id}` : ""} · {item.threats?.length ? `${item.threats.join(", ")} · object confidence ${((item.threat_confidence ?? 0) * 100).toFixed(0)}%` : item.reason ?? "Human review recommended"}</span></li>)}</ol> : <p className="muted">{summary.no_alert_explanation ?? "No review or alert rules were triggered."}</p>}<p className="muted small">Object detections are candidates for operator review, not confirmed threats.</p></article>
        </div>
        <article className="subcard rookie-guide">
          <p className="eyebrow">PLAIN-LANGUAGE REVIEW</p><h3>What happened in this video?</h3>
          <p className="guide-lead">{summary.event_count ? `${summary.event_count} event${summary.event_count === 1 ? "" : "s"}: object detections are alerts; action and motion cues request human review.` : summary.no_alert_explanation ?? "No review or alert rule was crossed."}</p>
          <div className="risk-explainer"><div><span className="eyebrow">PEAK REVIEW PRIORITY</span><strong>{peakPriority}/100</strong><p>This triage scale reflects the highest evidence level: ALERT 100, REVIEW 50, LOW 0. It is not a probability or model-confidence score.</p></div><div><span className="eyebrow">REVIEW FLAGS</span><strong>{reviewCount} of {tracks.length} tracks need review</strong><p>Counts tracks classified REVIEW only; {alertTrackCount} tracks are classified ALERT. {orphanReviewEvents ? `${orphanReviewEvents} review event${orphanReviewEvents === 1 ? " is" : "s are"} not linked to a person track.` : "Unlinked review events are reported separately from person tracks."} A flag calls for human checking; it does not label a person suspicious.</p></div></div>
          <div className="capability-grid">
            <div><b>Threat objects</b><p>{threat?.model_loaded ? `Object detector is active for ${(threat.supported_labels ?? []).length} configured classes; detections need human review.` : "Threat-object model loads on first analysis."}</p><p className={firearmClasses.length ? "available" : "unavailable"}>{firearmClasses.length ? `Configured firearm-related labels include ${firearmClasses.join(", ")}.` : "No firearm-related labels are active."}</p><p>{threat?.detection_note ?? "The detector reports objects; it does not determine intent or how a person is using one."}</p></div>
            <div><b>Action / anomaly</b><p>{anomaly?.synthetic_demo ? `Synthetic temporal/action weights are loaded for software checks only and are excluded from scoring. Motion output is a PROTOTYPE_REVIEW_CUE, not a trained action label. ${anomaly.reason ?? "Use its output as a prompt for human review only."}` : anomaly?.action_model_available ? `Action model loaded from ${anomaly.action_checkpoint ?? "the configured checkpoint"}. ${anomaly.reason ?? "Check results against labeled footage before relying on them."}` : anomaly?.method === "rule_based_motion_outlier" ? `A motion-based cue can point out unusual movement for review, but it does not identify an action or intent. ${anomaly.action_model_error ?? "No action checkpoint is installed."}` : anomaly?.reason ?? "No temporal action model is configured."}</p></div>
            <div><b>Confidence-only fusion</b><p>{summary.capabilities?.confidence_fusion?.available ? `${summary.capabilities.confidence_fusion.training_data ?? "LLVIP paired visible/infrared experiment"}: ${summary.capabilities.confidence_fusion.scope ?? "person-detection consensus calibration only."} ${summary.capabilities.confidence_fusion.validation_status ?? "Experimental; verify checkpoint provenance."}` : "Paper-aligned confidence checkpoint is not installed; rule-based fusion is active."}</p></div>
            <div><b>Visual Re-ID</b><p>{reid?.available ? `${reid.method ?? "Appearance features"}; reconnects anonymous track IDs by visual similarity after a short tracking gap.` : "Appearance matching is not enabled."}</p><p>IDs are pseudonyms scoped to this video, not names or biometric identities. Matches are approximate.</p></div>
            <div><b>Thermal face landmarks</b><p>{summary.capabilities?.thermal_face_landmarks?.enabled ? "Pretrained thermal landmark model enabled for true thermal input." : "Optional T-FAKE pretrained thermal landmark model is disabled."}</p><p>{summary.capabilities?.thermal_face_landmarks?.reason ?? "Set RAVEN_THERMAL_FACE_LANDMARKS=1 to enable."}</p></div>
          </div>
        </article>
        <article className="subcard track-review"><div className="section-heading"><div><p className="eyebrow">PERSON-BY-PERSON EVIDENCE</p><h3>Track review</h3></div><span className="muted">Alert level and appearance continuity</span></div>
          {summary.tracks?.length ? <div className="track-list">{summary.tracks.map((track) => {
            const detectedObjects = Object.values(track.detected_objects ?? {});
            const hasAlertObject = detectedObjects.some((confidence) => confidence >= threatConfidenceThreshold);
            const linkedEvents = summaryEvents.filter((event) =>
              event.track_id != null && String(event.track_id) === String(track.track_id));
            const level = getTrackLevel(track, threatConfidenceThreshold, linkedEvents);
            const reidText = track.reid_status === "appearance_match" ? "Appearance match reconnected this track" : track.reid_status === "tracker_continuity" ? "Tracker kept this ID across frames; no appearance re-link was needed" : track.reid_status === "new_track" ? "No confident appearance match; kept as a separate track" : track.reid_status ?? "No Re-ID result";
            const anomalyText = track.anomaly_score == null ? "" : ` · ${track.signal_label ?? "PROTOTYPE_REVIEW_CUE"} ${track.anomaly_score}/100`;
            const objects = Object.entries(track.detected_objects ?? {});
            const reviewFrameSummary = track.review_frames == null ? "Per-frame review count is unavailable in this saved result." : `Review cue in ${track.review_frames} analyzed frame${track.review_frames === 1 ? "" : "s"}${track.review_reasons?.length ? ` (${track.review_reasons.join(", ")})` : ""}.`;
            const alertFrameSummary = track.alert_frames == null ? "Per-frame alert count is unavailable in this saved result." : `Configured object signal in ${track.alert_frames} analyzed frame${track.alert_frames === 1 ? "" : "s"}.`;
            return <div className="track-row" key={track.track_id}><div className="track-head"><strong>Person {track.display_id ?? track.track_id}</strong><span className={`track-badge ${level.toLowerCase()}`}>{level}</span>{hasAlertObject && <b>object confidence {(Math.max(...detectedObjects) * 100).toFixed(0)}%</b>}</div><p>{track.posture && track.posture !== "unknown" ? `Posture estimate: ${track.posture}. ` : "No confident posture label. "}{track.frames_observed ?? 0} frames · {(track.first_seen_seconds ?? 0).toFixed(1)}–{(track.last_seen_seconds ?? 0).toFixed(1)}s · {trajectoryOverlay ? `${track.trajectory?.length ?? 0} trajectory points` : "trajectory overlay off"}{anomalyText}</p><p className="muted">Visual Re-ID: {reidText}{track.reid_status === "appearance_match" && track.reid_similarity != null ? ` · appearance similarity ${(track.reid_similarity * 100).toFixed(0)}%` : ""}{track.merged_tracker_ids?.length ? ` · merged tracker IDs ${track.merged_tracker_ids.join(", ")}` : ""}</p>{objects.length > 0 && <p className="track-threat">Detected objects: {objects.map(([name, confidence]) => `${name} ${(confidence * 100).toFixed(0)}%`).join(" · ")}</p>}{level === "REVIEW" && <p className="muted small">{reviewFrameSummary} Please confirm against the source video; this is not an attacker determination.</p>}{level === "ALERT" && <p className="muted small">{alertFrameSummary} Please confirm the object in the source video.</p>}</div>;
          })}</div> : <p className="muted">No person tracks were returned for this clip.</p>}
          {summary.threat_detections?.length ? <p className="muted small threat-counts">Object detections by class: {summary.threat_detections.map((item) => `${item.label} (${item.count} frames, peak ${(item.max_confidence ?? 0) * 100 | 0}%)`).join(" · ")}</p> : <p className="muted small threat-counts">No configured threat-object detections were recorded.</p>}
        </article>
        {summary.thermal_note && <p className="muted small">{summary.thermal_note}</p>}
          </>;
        })()}
      </>}
    </section>}

    <section className="card"><div className="section-heading"><div><p className="eyebrow">EVENTS & STATUS</p><h2>Live updates</h2></div><span className="muted">Latest 20</span></div>{events.length ? <ul className="live-list">{events.map((item, index) => {
      const label = item.type === "alert" ? (item.alert_event?.level ?? "ALERT").toUpperCase() : item.type === "job" ? (item.status ?? "Job update").toUpperCase() : "API CONNECTION";
      const alert = item.alert_event;
      const alertDetail = alert ? `${(alert.level ?? "REVIEW").toUpperCase()} · ${alert.signal_label ?? "PROTOTYPE_REVIEW_CUE"} · human review required · ${alert.threats?.length ? `${alert.threats.join(", ")} · object confidence ${((alert.threat_confidence ?? 0) * 100).toFixed(0)}%` : alert.reason ?? "Human review recommended"}` : "";
      const detail = item.type === "alert" ? alertDetail : item.error ?? (item.completed_frames != null ? `${item.completed_frames} frames processed` : item.summary ? `${item.summary.processed_frames ?? 0} frames · ${item.summary.event_count ?? 0} alerts` : item.type === "status" ? "Live status stream connected" : "Status updated");
      return <li key={`${item.job_id ?? "event"}-${index}`}><b>{label}</b><span>{detail}</span>{item.job_id && <code>{item.job_id.slice(0, 8)}</code>}</li>;
    })}</ul> : <p className="muted">Waiting for API or job status messages.</p>}</section>
    <footer>RGB-only uploads produce an estimated thermal-style visualization. Do not treat that visualization as a physical thermal measurement.</footer>
  </main>;
}
