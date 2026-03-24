"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import LaunchAppButton from "./LaunchAppButton";

export default function Navigation() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 60);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-[100] border-b transition-all duration-500 ${
        scrolled
          ? "border-white/10 bg-black/70 backdrop-blur-xl"
          : "border-transparent bg-transparent"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-full border border-emerald-400/50 flex items-center justify-center group-hover:border-emerald-400 transition-colors bg-emerald-400/10">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              className="w-4 h-4 text-emerald-400"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path d="M12 22c0 0 8-4 8-12C20 6 16 2 12 2S4 6 4 10c0 8 8 12 8 12z" />
              <path d="M12 22V10" />
              <path d="M8 14c2-1 4-3 4-4" />
              <path d="M16 14c-2-1-4-3-4-4" />
            </svg>
          </div>
          <span className="font-syncopate font-bold text-sm tracking-[0.15em] text-metallic">
            SMART CROP
          </span>
        </Link>

        <div className="hidden md:flex items-center gap-8">
          {["Technology", "Features", "About"].map((item) => (
            <Link
              key={item}
              href={`#${item.toLowerCase()}`}
              className="text-xs uppercase tracking-[0.2em] text-white/60 hover:text-emerald-400 transition-colors"
            >
              {item}
            </Link>
          ))}
        </div>

        <LaunchAppButton className="px-6 py-2.5 border border-emerald-400/30 hover:bg-emerald-400 hover:text-black transition-all duration-300 text-xs uppercase tracking-widest font-syncopate text-emerald-400 hover:border-emerald-400">
          Launch App
        </LaunchAppButton>
      </div>
    </nav>
  );
}
