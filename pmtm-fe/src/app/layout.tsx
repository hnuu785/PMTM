import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PMTM",
  description: "BPM based AI rap lyric assistant",
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
