"use client";

import React from "react";

export default function CarScroll() {
  return (
    <div className="relative w-full h-screen bg-black flex flex-col items-center justify-center overflow-hidden">
      {/* Video background */}
      <video
        autoPlay
        loop
        muted
        playsInline
        className="absolute inset-0 w-full h-full object-cover"
      >
        <source src="/seed-to-plant.mp4" type="video/mp4" />
      </video>

      {/* Overlay for text contrast */}
      <div className="absolute inset-0 bg-black/50 pointer-events-none" />

      {/* Title */}
      <h1 className="relative z-10 text-metallic text-5xl md:text-8xl font-black uppercase tracking-[0.2em] md:tracking-[0.4em] drop-shadow-2xl text-center px-4 font-syncopate">
        SMART CROP
      </h1>
      <p className="relative z-10 mt-4 text-sm md:text-lg uppercase tracking-[0.3em] text-white/60 text-center">
        Advisory System
      </p>
    </div>
  );
}
