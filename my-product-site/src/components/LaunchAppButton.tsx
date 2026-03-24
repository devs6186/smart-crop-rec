"use client";

import React, { useState, useCallback } from "react";

const STREAMLIT_URL = "http://localhost:8501";

interface LaunchAppButtonProps {
  className?: string;
  children: React.ReactNode;
}

export default function LaunchAppButton({
  className = "",
  children,
}: LaunchAppButtonProps) {
  const [toast, setToast] = useState<string | null>(null);

  const handleClick = useCallback(
    async (e: React.MouseEvent) => {
      e.preventDefault();
      setToast(null);

      try {
        // Try to reach the Streamlit server (fast timeout)
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 2000);

        await fetch(STREAMLIT_URL, {
          mode: "no-cors",
          signal: controller.signal,
        });
        clearTimeout(timeout);

        // If fetch didn't throw, the server is likely up — open it
        window.open(STREAMLIT_URL, "_blank", "noopener,noreferrer");
      } catch {
        // Server not reachable
        setToast(
          "Streamlit app is not running. Start it with: streamlit run app.py"
        );
        setTimeout(() => setToast(null), 5000);
      }
    },
    []
  );

  return (
    <>
      <button onClick={handleClick} className={className}>
        {children}
      </button>

      {/* Toast notification */}
      {toast && (
        <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-[200] animate-fade-in">
          <div className="bg-red-950/90 border border-red-500/30 backdrop-blur-xl rounded-lg px-6 py-4 shadow-2xl max-w-lg">
            <div className="flex items-start gap-3">
              <span className="text-red-400 text-lg mt-0.5">!</span>
              <div>
                <p className="text-sm text-red-200 font-medium">
                  App not running
                </p>
                <p className="text-xs text-red-300/70 mt-1 font-mono">
                  streamlit run app.py
                </p>
              </div>
              <button
                onClick={() => setToast(null)}
                className="text-red-400/60 hover:text-red-300 ml-4 text-lg leading-none"
              >
                &times;
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
