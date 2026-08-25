import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Baseline Labs — Tennis Match & Bracket Forecasts",
  description: "Compare 7,255 players across eras with injuries, travel, weather, coaching reports, custom brackets, and live tournament forecasts.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: {
    title: "Baseline Labs — Forecast the Point. Then the Path.",
    description: "Any-era forecasts with source-visible injuries, travel, weather, coaching changes, custom brackets, and live draws.",
    type: "website",
    images: [{ url: "/og-baseline-labs.png", width: 1729, height: 910, alt: "Abstract tennis court with thousands of simulated ball trajectories" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Baseline Labs — Tennis Match & Bracket Forecasts",
    description: "Search 7,255 players and add current injuries, travel, conditions, coaching reports, brackets, and live tour probabilities.",
    images: ["/og-baseline-labs.png"],
  },
};

export const viewport: Viewport = { themeColor: "#0e1a16", colorScheme: "light" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
