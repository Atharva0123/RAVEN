import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "RAVEN Research Prototype", description: "Multi-Modal Quality-Aware Sensor Fusion & Temporal Event Ranking" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
