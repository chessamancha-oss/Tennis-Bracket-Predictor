import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Baseline Labs — Probabilistic Tennis Forecasts",
  description: "Surface-aware, point-by-point tennis match forecasting for professional and custom player matchups.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: {
    title: "Baseline Labs — Probabilistic Tennis Forecasts",
    description: "Model tennis point by point with surface-aware ratings, Bayesian uncertainty, and complete scoring simulations.",
    type: "website",
    images: [{ url: "/og-baseline-labs.png", width: 1729, height: 910, alt: "Abstract tennis court with thousands of simulated ball trajectories" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Baseline Labs — Probabilistic Tennis Forecasts",
    description: "A professional match studio for current tour players and custom scouting profiles.",
    images: ["/og-baseline-labs.png"],
  },
};

export const viewport: Viewport = { themeColor: "#0e1a16", colorScheme: "light" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
