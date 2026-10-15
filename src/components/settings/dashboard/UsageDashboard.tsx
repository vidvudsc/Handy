import React, { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { commands, events } from "@/bindings";
import type { UsageStats } from "@/bindings";

// Number of week-columns to show in the contribution grid (~1 year).
const WEEKS = 53;
// Cell size / gap (px). Sized so a full year of columns fits without scrolling.
const CELL = 10;
const GAP = 3;
const TOOLTIP_WIDTH = 180;

interface TooltipState {
  key: string;
  dateLabel: string;
  activityLabel: string;
  wordsLabel: string;
  left: number;
  top: number;
  placement: "above" | "below";
}

function dateKey(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
    d.getDate(),
  ).padStart(2, "0")}`;
}

function addDays(base: Date, n: number): Date {
  const d = new Date(base);
  d.setDate(d.getDate() + n);
  return d;
}

// Map a transcription count to one of 5 intensity buckets (GitHub-style).
function intensityClass(count: number): string {
  if (count <= 0) return "bg-mid-gray/15";
  if (count <= 2) return "bg-logo-primary/30";
  if (count <= 5) return "bg-logo-primary/50";
  if (count <= 9) return "bg-logo-primary/75";
  return "bg-logo-primary";
}

interface StatCardProps {
  label: string;
  value: string;
  hint?: string;
}

const StatCard: React.FC<StatCardProps> = ({ label, value, hint }) => (
  <div className="flex flex-col gap-1 rounded-lg border border-mid-gray/20 bg-mid-gray/5 px-4 py-3">
    <span className="text-2xl font-semibold tabular-nums">{value}</span>
    <span className="text-xs text-mid-gray">{label}</span>
    {hint && <span className="text-[10px] text-mid-gray/70">{hint}</span>}
  </div>
);

export const UsageDashboard: React.FC = () => {
  const { t, i18n } = useTranslation();
  const [stats, setStats] = useState<UsageStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  // Horizontal scroll container for the contribution grid. When the grid is
  // wider than the panel we anchor it to the right edge so the most recent
  // week is always in view on load (GitHub-style), instead of the default
  // left-anchored scroll that shows the oldest weeks.
  const gridScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;

    const refreshStats = async () => {
      try {
        const res = await commands.getUsageStats();
        if (!active) return;
        if (res.status === "ok") setStats(res.data);
        setLoading(false);
      } catch {
        if (active) setLoading(false);
      }
    };

    void refreshStats();

    const unlisten = events.usageStatsUpdatePayload.listen((event) => {
      if (active) {
        setStats(event.payload.stats);
        setLoading(false);
      }
    });

    return () => {
      active = false;
      unlisten.then((fn) => fn());
    };
  }, []);

  // Once stats load (and the grid has rendered), jump to the right edge so the
  // current week is visible. Runs after `loading` flips so scrollWidth is set.
  useEffect(() => {
    if (loading) return;
    const el = gridScrollRef.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [loading, stats]);

  // day-key -> { count, words }
  const dayMap = useMemo(() => {
    const map = new Map<string, { count: number; words: number }>();
    if (stats) {
      for (const d of stats.days) {
        const date = new Date(d.day * 1000);
        map.set(dateKey(date), { count: d.count, words: d.words });
      }
    }
    return map;
  }, [stats]);

  // Derived stats: active days, streaks, averages.
  const derived = useMemo(() => {
    const activeDays = dayMap.size;
    const totalTranscriptions = stats?.total_transcriptions ?? 0;
    const totalWords = stats?.total_words ?? 0;

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Current streak: consecutive active days ending today (or yesterday).
    let currentStreak = 0;
    let cursor = new Date(today);
    if (!dayMap.has(dateKey(cursor))) cursor = addDays(cursor, -1);
    while (dayMap.has(dateKey(cursor))) {
      currentStreak++;
      cursor = addDays(cursor, -1);
    }

    // Longest streak: scan sorted active days, counting calendar-consecutive runs.
    const keys = [...dayMap.keys()].sort();
    let longestStreak = 0;
    let run = 0;
    let prevKey: string | null = null;
    for (const k of keys) {
      const expected = prevKey
        ? dateKey(addDays(new Date(`${prevKey}T00:00:00`), 1))
        : null;
      run = expected === k ? run + 1 : 1;
      longestStreak = Math.max(longestStreak, run);
      prevKey = k;
    }

    const dailyAverage = activeDays > 0 ? totalTranscriptions / activeDays : 0;
    const dailyWordAverage = activeDays > 0 ? totalWords / activeDays : 0;

    return {
      activeDays,
      totalTranscriptions,
      totalWords,
      currentStreak,
      longestStreak,
      dailyAverage,
      dailyWordAverage,
    };
  }, [dayMap, stats]);

  // Build the contribution grid (columns = weeks, rows = weekdays, Sun at top).
  const { weeks, monthLabels } = useMemo(() => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const currentWeekSunday = addDays(today, -today.getDay());
    const startSunday = addDays(currentWeekSunday, -(WEEKS - 1) * 7);

    const cols: {
      key: string;
      count: number;
      words: number;
      future: boolean;
      date: Date;
    }[][] = [];
    const labels: { col: number; label: string }[] = [];
    let lastMonth = -1;

    for (let w = 0; w < WEEKS; w++) {
      const col: {
        key: string;
        count: number;
        words: number;
        future: boolean;
        date: Date;
      }[] = [];
      const topDate = addDays(startSunday, w * 7);
      const month = topDate.getMonth();
      if (month !== lastMonth) {
        labels.push({
          col: w,
          label: topDate.toLocaleString(i18n.language, { month: "short" }),
        });
        lastMonth = month;
      }
      for (let dow = 0; dow < 7; dow++) {
        const date = addDays(startSunday, w * 7 + dow);
        const key = dateKey(date);
        const day = dayMap.get(key);
        col.push({
          key,
          count: day?.count ?? 0,
          words: day?.words ?? 0,
          future: date > today,
          date,
        });
      }
      cols.push(col);
    }
    return { weeks: cols, monthLabels: labels };
  }, [dayMap, i18n.language]);

  const formatDate = (date: Date): string =>
    date.toLocaleDateString(i18n.language, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });

  const activityLabel = (count: number): string => {
    if (count === 0) return t("dashboard.grid.noActivity");
    if (count === 1) return t("dashboard.grid.oneTranscription");
    return t("dashboard.grid.nTranscriptions", { count });
  };

  const wordsLabel = (words: number): string => {
    if (words === 0) return t("dashboard.grid.noWords");
    if (words === 1) return t("dashboard.grid.oneWord");
    return t("dashboard.grid.nWords", { count: words });
  };

  const showTooltip = (
    event:
      | React.PointerEvent<HTMLDivElement>
      | React.FocusEvent<HTMLDivElement>,
    cell: { key: string; count: number; words: number; date: Date },
  ) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const viewportWidth = window.innerWidth || TOOLTIP_WIDTH;
    const left = Math.min(
      Math.max(rect.left + rect.width / 2, TOOLTIP_WIDTH / 2 + 12),
      viewportWidth - TOOLTIP_WIDTH / 2 - 12,
    );
    const hasRoomAbove = rect.top > 96;

    setTooltip({
      key: cell.key,
      dateLabel: formatDate(cell.date),
      activityLabel: activityLabel(cell.count),
      wordsLabel: wordsLabel(cell.words),
      left,
      top: hasRoomAbove ? rect.top - 8 : rect.bottom + 8,
      placement: hasRoomAbove ? "above" : "below",
    });
  };

  const fmt = (n: number) => n.toLocaleString(i18n.language);

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <div>
        <h2 className="text-lg font-semibold">{t("dashboard.title")}</h2>
        <p className="text-sm text-mid-gray">{t("dashboard.subtitle")}</p>
      </div>

      {loading ? (
        <p className="text-sm text-mid-gray">…</p>
      ) : derived.totalTranscriptions === 0 ? (
        <p className="text-sm text-mid-gray">{t("dashboard.empty")}</p>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <StatCard
              label={t("dashboard.stats.totalTranscriptions")}
              value={fmt(derived.totalTranscriptions)}
            />
            <StatCard
              label={t("dashboard.stats.totalWords")}
              value={fmt(derived.totalWords)}
            />
            <StatCard
              label={t("dashboard.stats.activeDays")}
              value={fmt(derived.activeDays)}
            />
            <StatCard
              label={t("dashboard.stats.currentStreak")}
              value={`${fmt(derived.currentStreak)} ${t("dashboard.stats.days")}`}
            />
            <StatCard
              label={t("dashboard.stats.longestStreak")}
              value={`${fmt(derived.longestStreak)} ${t("dashboard.stats.days")}`}
            />
            <StatCard
              label={t("dashboard.stats.dailyAverage")}
              value={derived.dailyAverage.toFixed(1)}
              hint={t("dashboard.stats.perActiveDay")}
            />
            <StatCard
              label={t("dashboard.stats.dailyWordAverage")}
              value={fmt(Math.round(derived.dailyWordAverage))}
              hint={t("dashboard.stats.wordsPerActiveDay")}
            />
          </div>

          <div
            ref={gridScrollRef}
            className="rounded-lg border border-mid-gray/20 bg-mid-gray/5 p-4 overflow-x-auto"
            onPointerLeave={() => setTooltip(null)}
          >
            <div className="inline-flex flex-col">
              {/* Month labels */}
              <div
                className="flex text-[10px] text-mid-gray mb-2"
                style={{ gap: GAP }}
              >
                {weeks.map((_, w) => {
                  const label = monthLabels.find((m) => m.col === w);
                  return (
                    <div
                      key={w}
                      className="relative"
                      style={{ width: CELL, height: 12 }}
                    >
                      {label && (
                        <span className="absolute left-0 top-0 whitespace-nowrap">
                          {label.label}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
              {/* Grid */}
              <div className="flex" style={{ gap: GAP }}>
                {weeks.map((col, w) => (
                  <div key={w} className="flex flex-col" style={{ gap: GAP }}>
                    {col.map((cell) => (
                      <div
                        key={cell.key}
                        role={cell.future ? undefined : "gridcell"}
                        tabIndex={cell.future ? undefined : 0}
                        aria-label={
                          cell.future
                            ? undefined
                            : `${formatDate(cell.date)}: ${activityLabel(cell.count)}, ${wordsLabel(cell.words)}`
                        }
                        onPointerEnter={(event) => {
                          if (!cell.future) showTooltip(event, cell);
                        }}
                        onFocus={(event) => {
                          if (!cell.future) showTooltip(event, cell);
                        }}
                        onBlur={() => setTooltip(null)}
                        className={`rounded-sm transition-transform duration-75 ${
                          cell.future
                            ? "bg-transparent"
                            : `${intensityClass(cell.count)} cursor-default focus:outline-none focus:ring-2 focus:ring-logo-primary/60 focus:ring-offset-1 focus:ring-offset-white hover:scale-125`
                        }`}
                        style={{ width: CELL, height: CELL }}
                      />
                    ))}
                  </div>
                ))}
              </div>
              {/* Legend */}
              <div className="flex items-center gap-1 justify-end text-[10px] text-mid-gray mt-2">
                <span>{t("dashboard.grid.less")}</span>
                <div
                  className="rounded-sm bg-mid-gray/15"
                  style={{ width: CELL, height: CELL }}
                />
                <div
                  className="rounded-sm bg-logo-primary/30"
                  style={{ width: CELL, height: CELL }}
                />
                <div
                  className="rounded-sm bg-logo-primary/50"
                  style={{ width: CELL, height: CELL }}
                />
                <div
                  className="rounded-sm bg-logo-primary/75"
                  style={{ width: CELL, height: CELL }}
                />
                <div
                  className="rounded-sm bg-logo-primary"
                  style={{ width: CELL, height: CELL }}
                />
                <span>{t("dashboard.grid.more")}</span>
              </div>
            </div>
          </div>

          {tooltip && (
            <div
              role="tooltip"
              className="fixed z-50 rounded-md border border-mid-gray/20 bg-white px-3 py-2 text-xs shadow-lg pointer-events-none"
              style={{
                width: TOOLTIP_WIDTH,
                left: tooltip.left,
                top: tooltip.top,
                transform:
                  tooltip.placement === "above"
                    ? "translate(-50%, -100%)"
                    : "translate(-50%, 0)",
              }}
            >
              <div className="font-medium text-mid-gray">
                {tooltip.dateLabel}
              </div>
              <div className="mt-1 text-mid-gray">{tooltip.activityLabel}</div>
              <div className="text-mid-gray">{tooltip.wordsLabel}</div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
