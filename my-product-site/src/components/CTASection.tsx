"use client";

import React from "react";
import { motion } from "framer-motion";

const steps = [
  {
    number: "01",
    title: "Select Your Region",
    description: "Choose your state and district for localised soil and climate defaults.",
  },
  {
    number: "02",
    title: "Enter Land Size",
    description: "Specify your land in bigha — auto-converted to acres using state-specific rates.",
  },
  {
    number: "03",
    title: "Get Recommendations",
    description: "Receive top 5 crops ranked by suitability with production, pricing, and risk data.",
  },
];

export default function CTASection() {
  return (
    <section
      id="about"
      className="py-40 px-6 relative overflow-hidden border-t border-white/10 bg-black"
    >
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl h-full bg-emerald-500/5 blur-[150px] pointer-events-none rounded-full" />

      <div className="max-w-5xl mx-auto relative z-10">
        {/* How it works */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="mb-24"
        >
          <p className="text-xs font-syncopate uppercase tracking-[0.3em] text-emerald-400/70 mb-4 text-center">
            How It Works
          </p>
          <h2 className="text-3xl md:text-5xl font-syncopate font-bold uppercase tracking-[0.15em] text-metallic mb-16 text-center">
            THREE STEPS
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {steps.map((step, i) => (
              <motion.div
                key={step.number}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: i * 0.15 }}
                className="text-center"
              >
                <div className="text-5xl font-syncopate font-bold text-emerald-400/20 mb-4">
                  {step.number}
                </div>
                <h3 className="text-sm font-syncopate uppercase tracking-[0.15em] text-white mb-3">
                  {step.title}
                </h3>
                <p className="text-sm text-white/40 leading-relaxed">
                  {step.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-center"
        >
          <h2 className="text-4xl md:text-7xl font-syncopate font-black uppercase tracking-[0.15em] text-metallic mb-8">
            GROW SMARTER
          </h2>

          <p className="text-sm md:text-lg tracking-[0.15em] text-white/60 mb-12 max-w-2xl mx-auto uppercase">
            Advisory-powered crop intelligence for every Indian farmer.
            Open source. Free to use. Built on real data.
          </p>

          <a
            href="http://localhost:8501"
            target="_blank"
            rel="noopener noreferrer"
            className="group relative inline-flex items-center justify-center px-12 py-5 font-syncopate text-xs uppercase tracking-[0.3em] font-bold text-black bg-emerald-400 overflow-hidden transition-all hover:scale-105 hover:bg-emerald-300 duration-300 rounded-sm"
          >
            <span className="absolute w-0 h-0 transition-all duration-500 ease-out bg-black/10 rounded-full group-hover:w-64 group-hover:h-56"></span>
            <span className="relative">Launch Advisory System</span>
          </a>

          <div className="mt-8 text-xs font-syncopate uppercase tracking-[0.2em] text-white/30">
            Powered by Streamlit + scikit-learn
          </div>
        </motion.div>
      </div>
    </section>
  );
}
