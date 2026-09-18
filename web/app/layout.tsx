import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Manu | Protein Stability",
  description: "Research-support protein stability estimates for single substitutions.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
