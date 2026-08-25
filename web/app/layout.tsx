import type { Metadata, Viewport } from "next";
import { playerDatabaseSummary } from "../data/player-database-summary.generated";
import "./globals.css";

const catalogueCount = playerDatabaseSummary.count.toLocaleString("en-US");

export const metadata: Metadata = {
  metadataBase: new URL("https://baseline-tennis-labs.sorakabeta.chatgpt.site"),
  title: "Baseline Labs — Serious Tennis Forecasting",
  description: `Compare ${catalogueCount} players across eras with injuries, travel, weather, coaching reports, custom brackets, and live tournament forecasts.`,
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: {
    title: "Baseline Labs — Every Match Has a Hidden Shape.",
    description: "Any-era forecasts with source-visible injuries, travel, weather, coaching changes, custom brackets, and live draws.",
    type: "website",
    images: [{ url: "/og-baseline-labs-v2.png", width: 1731, height: 909, alt: "Baseline Labs probability trajectories crossing an abstract tennis court" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Baseline Labs — Every Match Has a Hidden Shape.",
    description: `Search ${catalogueCount} players and add current injuries, travel, conditions, coaching reports, brackets, and live tour probabilities.`,
    images: ["/og-baseline-labs-v2.png"],
  },
};

export const viewport: Viewport = { themeColor: "#0c1229", colorScheme: "light" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
