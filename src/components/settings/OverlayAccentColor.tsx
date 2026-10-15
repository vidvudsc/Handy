import React from "react";
import { useTranslation } from "react-i18next";
import { SettingContainer } from "../ui/SettingContainer";
import { useSettings } from "../../hooks/useSettings";

interface OverlayAccentColorProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

// A few handy presets alongside the freeform color picker.
const PRESETS = [
  "#ffffff",
  "#ffe5ee",
  "#7dd3fc",
  "#86efac",
  "#fcd34d",
  "#f87171",
  "#c084fc",
];

export const OverlayAccentColor: React.FC<OverlayAccentColorProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const color = (getSetting("overlay_accent_color") || "#ffffff") as string;
    const disabled = isUpdating("overlay_accent_color");

    return (
      <SettingContainer
        title={t("settings.advanced.overlayColor.title")}
        description={t("settings.advanced.overlayColor.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      >
        <div className="flex items-center gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset}
              type="button"
              aria-label={preset}
              disabled={disabled}
              onClick={() => updateSetting("overlay_accent_color", preset)}
              className={`w-5 h-5 rounded-full border transition-transform hover:scale-110 ${
                color.toLowerCase() === preset.toLowerCase()
                  ? "border-logo-primary ring-2 ring-logo-primary/50"
                  : "border-mid-gray/40"
              }`}
              style={{ background: preset }}
            />
          ))}
          <input
            type="color"
            value={color}
            disabled={disabled}
            onChange={(e) =>
              updateSetting("overlay_accent_color", e.target.value)
            }
            className="w-8 h-8 rounded cursor-pointer bg-transparent border border-mid-gray/40"
          />
        </div>
      </SettingContainer>
    );
  },
);
