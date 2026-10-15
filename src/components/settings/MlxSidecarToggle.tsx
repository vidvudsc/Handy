import React from "react";
import { useTranslation } from "react-i18next";
import { platform } from "@tauri-apps/plugin-os";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface MlxSidecarToggleProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const MlxSidecarToggle: React.FC<MlxSidecarToggleProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    if (platform() !== "macos") return null;

    const enabled = getSetting("mlx_sidecar") ?? false;

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(value) => updateSetting("mlx_sidecar", value)}
        isUpdating={isUpdating("mlx_sidecar")}
        label={t("settings.advanced.acceleration.mlxSidecar.label")}
        description={t("settings.advanced.acceleration.mlxSidecar.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  },
);
