import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PMTM",
  description: "AI lyric generation service",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
