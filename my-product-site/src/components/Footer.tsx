import React from "react";
import Link from "next/link";

export default function Footer() {
  return (
    <footer className="bg-black pt-20 pb-10 px-6 border-t border-white/10 relative overflow-hidden">
      <div className="max-w-7xl mx-auto relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 md:gap-8 mb-20">
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-6 h-6 rounded-full border border-emerald-400/40 flex items-center justify-center">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 opacity-50" />
              </div>
              <span className="font-syncopate font-bold text-sm tracking-[0.15em] text-metallic">
                SMART CROP
              </span>
            </div>
            <p className="text-xs uppercase tracking-[0.15em] text-white/40 max-w-sm leading-loose">
              AI-powered crop advisory system for Indian agriculture. Region-aware
              recommendations backed by machine learning, market data, and agronomic science.
            </p>
          </div>

          <div>
            <h4 className="text-xs font-syncopate uppercase tracking-[0.2em] text-white/80 mb-6">
              System
            </h4>
            <ul className="space-y-4">
              {["Technology", "Features", "About"].map(
                (link) => (
                  <li key={link}>
                    <Link
                      href={`#${link.toLowerCase()}`}
                      className="text-xs uppercase tracking-widest text-white/40 hover:text-emerald-400 transition-colors"
                    >
                      {link}
                    </Link>
                  </li>
                )
              )}
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-syncopate uppercase tracking-[0.2em] text-white/80 mb-6">
              Data Sources
            </h4>
            <ul className="space-y-4">
              {[
                "ICAR Guidelines",
                "data.gov.in API",
                "NIPHM Publications",
                "Kaggle Datasets",
              ].map((link) => (
                <li key={link}>
                  <span className="text-xs uppercase tracking-widest text-white/40">
                    {link}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-[10px] uppercase tracking-[0.2em] text-white/30">
            Smart Agriculture Advisory System &mdash; Open Source
          </div>
          <div className="flex gap-6">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[10px] uppercase tracking-[0.2em] text-white/30 hover:text-emerald-400 transition-colors"
            >
              GitHub
            </a>
            <a
              href="http://localhost:8501"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[10px] uppercase tracking-[0.2em] text-white/30 hover:text-emerald-400 transition-colors"
            >
              Launch App
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
