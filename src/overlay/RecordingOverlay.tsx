import { listen } from "@tauri-apps/api/event";
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import "./RecordingOverlay.css";
import { commands, events } from "@/bindings";
import type {
  StreamPhase,
  StreamPhaseEvent,
  StreamTextEvent,
  StreamWorkKind,
} from "@/bindings";
import i18n, { syncLanguageFromSettings } from "@/i18n";
import { getLanguageDirection } from "@/lib/utils/rtl";

type OverlayState = "recording" | "streaming" | "transcribing" | "processing";

// Handy Beta's compact pill uses four measured bands. Live receives 16 FFT
// buckets, which are summarized into four bars in its smaller waveform.
const WAVE_BARS = 4;

const RecordingOverlay: React.FC = () => {
  const { t } = useTranslation();
  const [isVisible, setIsVisible] = useState(false);
  const [shouldRender, setShouldRender] = useState(false);
  const [state, setState] = useState<OverlayState>("recording");
  // `Stream::play()` returning does not mean hardware callbacks are flowing.
  // Stay visually in an arming state until the backend processes the first
  // actual microphone sample chunk.
  const [captureReady, setCaptureReady] = useState(false);
  const [levels, setLevels] = useState<number[]>(Array(WAVE_BARS).fill(0));
  const [accentColor, setAccentColor] = useState("#ffffff");
  const [streamText, setStreamText] = useState<StreamTextEvent>({
    committed: "",
    tentative: "",
  });
  const [phase, setPhase] = useState<StreamPhase>("listening");
  const [workKind, setWorkKind] = useState<StreamWorkKind>("transcribing");
  const [elapsed, setElapsed] = useState(0);
  // Bumped on each new streaming session so the Live card remounts fresh (replays
  // the pop-in, and never animates in from the previous panel's open size).
  const [session, setSession] = useState(0);
  // Overlay placement (top vs bottom of the screen). The Live panel grows downward
  // from a top overlay (oldest line under the pill) and upward from a bottom one.
  const [position, setPosition] = useState<"top" | "bottom">("bottom");
  // True once live text overflows the cap. A top overlay fades its top edge only
  // while overflowing, so the resting first line stays crisp flush under the pill.
  const [overflowing, setOverflowing] = useState(false);

  const smoothedLevelsRef = useRef<number[]>(Array(WAVE_BARS).fill(0));
  const compactTargetsRef = useRef<number[]>(Array(WAVE_BARS).fill(0));
  const compactBarsRef = useRef<(HTMLSpanElement | null)[]>([]);
  const stateRef = useRef<OverlayState | null>(null);
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Live-text scroll-back: the text region "sticks" to the newest line while the
  // user is at the bottom; if they scroll up to read history, auto-follow pauses
  // until they scroll back down.
  const capRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const direction = getLanguageDirection(i18n.language);

  useEffect(() => {
    const setupEventListeners = async () => {
      const unlistenShow = await listen("show-overlay", (event) => {
        const overlayState = event.payload as OverlayState;
        stateRef.current = overlayState;
        // Render the new state before doing any settings I/O. In particular, a
        // short MLX transcription can finish while those reads are pending.
        if (overlayState === "recording" || overlayState === "streaming") {
          setCaptureReady(false);
          smoothedLevelsRef.current = Array(WAVE_BARS).fill(0);
          compactTargetsRef.current = Array(WAVE_BARS).fill(0);
          setLevels(Array(WAVE_BARS).fill(0));
          setStreamText({ committed: "", tentative: "" });

          void syncLanguageFromSettings();
          // Placement and accent only need refreshing at the start of a session.
          // Neither may hold up recording, transcribing, or the hide animation.
          void commands
            .getAppSettings()
            .then((settings) => {
              if (settings.status === "ok") {
                setAccentColor(settings.data.overlay_accent_color || "#ffffff");
                setPosition(
                  settings.data.overlay_position === "top" ? "top" : "bottom",
                );
              }
            })
            .catch(() => {
              // Keep the previous/default placement if settings can't be read.
            });
        }
        setState(overlayState);
        if (overlayState === "streaming") {
          setPhase("listening");
          setWorkKind("transcribing");
          setElapsed(0);
          setSession((s) => s + 1); // remount the card fresh for this session
        }
        if (hideTimerRef.current !== null) {
          clearTimeout(hideTimerRef.current);
          hideTimerRef.current = null;
        }
        setShouldRender(true);
        setIsVisible(true);
      });

      const unlistenHide = await listen("hide-overlay", () => {
        stateRef.current = null;
        setIsVisible(false);
        if (hideTimerRef.current !== null) clearTimeout(hideTimerRef.current);
        // Rust hides the native window after 300ms. Keep the DOM through the
        // compact fade and the Live card's leave transition, then stop animating.
        hideTimerRef.current = setTimeout(() => {
          setShouldRender(false);
          hideTimerRef.current = null;
        }, 300);
      });

      const unlistenReady = await listen("recording-ready", () => {
        setElapsed(0);
        setCaptureReady(true);
      });

      const unlistenLevel = await listen<number[]>("mic-level", (event) => {
        const overlayState = stateRef.current;
        if (overlayState !== "recording" && overlayState !== "streaming")
          return;
        const newLevels = event.payload as number[];
        if (overlayState === "recording") {
          // Audio arrives at 20 Hz. Store the latest measured shape without
          // rerendering the overlay; the frame loop interpolates between samples.
          compactTargetsRef.current = Array.from(
            { length: WAVE_BARS },
            (_, i) => {
              const value = newLevels[i];
              return Number.isFinite(value)
                ? Math.max(0, Math.min(1, value))
                : 0;
            },
          );
          return;
        }
        // The backend sends Beta's original four RMS-weighted bands for the
        // compact pill and the newer 16-bucket spectrum for Live.
        const bands = Array.from({ length: WAVE_BARS }, (_, i) =>
          Math.max(0, ...newLevels.slice(i * 4, i * 4 + 4)),
        );
        const smoothed = smoothedLevelsRef.current.map((prev, i) => {
          const target = bands[i] || 0;
          return prev * 0.2 + target * 0.8;
        });
        smoothedLevelsRef.current = smoothed;
        setLevels(smoothed);
      });

      const unlistenStream = await events.streamTextEvent.listen((event) => {
        setStreamText(event.payload);
      });

      const unlistenPhase = await events.streamPhaseEvent.listen((event) => {
        const payload: StreamPhaseEvent = event.payload;
        setPhase(payload.phase);
        if (payload.kind) setWorkKind(payload.kind);
      });

      return () => {
        unlistenShow();
        unlistenHide();
        unlistenReady();
        unlistenLevel();
        unlistenStream();
        unlistenPhase();
      };
    };

    let disposed = false;
    let cleanup: (() => void) | undefined;
    void setupEventListeners().then((unlisten) => {
      if (disposed) unlisten();
      else cleanup = unlisten;
    });
    return () => {
      disposed = true;
      cleanup?.();
      if (hideTimerRef.current !== null) clearTimeout(hideTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (!isVisible || state !== "recording" || !captureReady) return;
    let frame: number;
    let previous = performance.now();
    const displayed = Array(WAVE_BARS).fill(0) as number[];
    const animate = (now: number) => {
      // Time-based easing is consistent on 60 Hz and ProMotion displays.
      const blend = 1 - Math.exp(-Math.min(now - previous, 100) / 35);
      previous = now;
      compactBarsRef.current.forEach((bar, i) => {
        if (!bar) return;
        displayed[i] += (compactTargetsRef.current[i] - displayed[i]) * blend;
        const level = displayed[i];
        bar.style.transform = `scaleY(${(4 + Math.sqrt(level) * 16) / 20})`;
        bar.style.opacity = String(Math.min(1, 0.5 + level * 3));
      });
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [isVisible, state, captureReady]);

  // Elapsed capture timer starts only once microphone samples are flowing.
  useEffect(() => {
    if (state !== "streaming" || !isVisible || !captureReady) return;
    const id = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(id);
  }, [state, isVisible, captureReady]);

  // Stick to the bottom as text streams in — but only while pinned, so a user who
  // has scrolled up to read history isn't yanked back down by the next chunk.
  useLayoutEffect(() => {
    const el = capRef.current;
    if (!el) return;
    // Fade the top edge only once text actually overflows the cap.
    setOverflowing(el.scrollHeight > el.clientHeight + 1);
    if (pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [streamText]);

  // Each fresh streaming session starts pinned to the bottom, fade cleared.
  useEffect(() => {
    pinnedRef.current = true;
    setOverflowing(false);
  }, [session]);

  if (!shouldRender) return null;

  // Re-pin when the user is within ~a line of the bottom; unpin otherwise.
  const handleStreamScroll = () => {
    const el = capRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight <= 16;
  };

  const fmtTime = (s: number) =>
    `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

  // ---- Shared building blocks (one visual language for every overlay form) ----
  const waveform = (
    <div className={`swave ${captureReady ? "ready" : "arming"}`}>
      {levels.map((v, i) => (
        <i
          key={i}
          style={{
            height: `${Math.max(3, Math.min(18, 3 + Math.pow(v, 0.7) * 15))}px`,
          }}
        />
      ))}
    </div>
  );

  const cancelBtn = (
    <button
      className="sx"
      aria-label="cancel"
      onClick={() => commands.cancelOperation()}
    >
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path
          d="M4 4 L12 12 M12 4 L4 12"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
    </button>
  );

  // dot (left) | waveform (center) | timer + cancel (right) — same structure for
  // pill & panel, so the Live morph is a pure width change.
  const listeningRow = (showTimer: boolean, showCancel: boolean) => (
    <div className="sbase">
      <div className="sbase-l">
        <span className={`sdot ${captureReady ? "ready" : "arming"}`} />
      </div>
      {waveform}
      <div className="sbase-r">
        {showTimer && <span className="stimer">{fmtTime(elapsed)}</span>}
        {showCancel && cancelBtn}
      </div>
    </div>
  );

  // spinner (left) | label (center) | cancel (right) — same 3-zone grid as the
  // listening row, so the label is centered.
  const workingRow = (label: string, showCancel: boolean) => (
    <div className="sbase">
      <div className="sbase-l">
        <span className="sspinner" />
      </div>
      <span className="swork-label">{label}</span>
      <div className="sbase-r">{showCancel && cancelBtn}</div>
    </div>
  );

  // ---- Live overlay: a pill that sculpts open into a panel ----
  if (state === "streaming") {
    const hasText =
      streamText.committed.length > 0 || streamText.tentative.length > 0;
    const working = phase === "working";
    // Keep the panel open whenever there's text — even while finalizing — so the
    // transcript stays put under a working spinner instead of collapsing and
    // squishing the text mid-stream. Only fall back to the small working pill
    // when there was no text to preserve.
    const open = hasText;
    const collapsed = working && !hasText;

    return (
      <div
        dir={direction}
        className={`ov-stage ${position}`}
        style={{ "--s-accent": accentColor } as React.CSSProperties}
      >
        <div
          key={session}
          className={`scard ${open ? "open" : ""} ${collapsed ? "working" : ""} ${
            isVisible ? "" : "leaving"
          }`}
        >
          <div className="stext">
            <div className="stext-clip">
              <div
                className={`stext-cap ${overflowing ? "overflowing" : ""}`}
                ref={capRef}
                onScroll={handleStreamScroll}
              >
                <p>
                  <span className="committed">
                    {streamText.committed ? streamText.committed + " " : ""}
                  </span>
                  <span className="tentative">{streamText.tentative}</span>
                  {/* Drop the blinking caret once finalizing — it's no longer
                      capturing, and a static spinner conveys the work. */}
                  {!working && <span className="scaret" />}
                </p>
              </div>
            </div>
          </div>
          {working
            ? workingRow(
                workKind === "polishing"
                  ? t("overlay.processing")
                  : t("overlay.transcribing"),
                true,
              )
            : listeningRow(open, true)}
        </div>
      </div>
    );
  }

  // ---- Keep Handy Beta's minimal four-bar pill for batch transcription. ----
  const working = state === "transcribing" || state === "processing";

  return (
    <div
      dir={direction}
      className={`ov-stage ${position} ov-fade ${isVisible ? "show" : ""}`}
    >
      <div className="beta-compact">
        {Array.from({ length: WAVE_BARS }, (_, i) => (
          <span
            key={i}
            ref={(bar) => {
              compactBarsRef.current[i] = bar;
            }}
            className={working ? "beta-bar loading" : "beta-bar"}
            style={{
              transform: working ? undefined : "scaleY(0.2)",
              opacity: working ? undefined : 0.5,
              background: accentColor,
              animationDelay: `${i * 0.12}s`,
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default RecordingOverlay;
