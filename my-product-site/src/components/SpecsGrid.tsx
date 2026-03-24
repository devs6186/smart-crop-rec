"use client";

import React from "react";
import { motion } from "framer-motion";

const specifications = [
  {
    label: "Crop Database",
    value: "51",
    detail: "Crops with full agronomic profiles",
  },
  {
    label: "ML Accuracy",
    value: "96%",
    detail: "Test F1-macro across all crop classes",
  },
  {
    label: "States Covered",
    value: "28+",
    detail: "All Indian states and union territories",
  },
  {
    label: "ML Models Compared",
    value: "6",
    detail: "RF, SVM, KNN, Decision Tree, Extra Trees, LR",
  },
  {
    label: "Risk Factors",
    value: "120+",
    detail: "Disease, pest, and climate risk entries",
  },
  {
    label: "Market Data",
    value: "Live",
    detail: "Real-time prices via data.gov.in API",
  },
];

export default function SpecsGrid() {
  return (
    <section
      id="technology"
      className="py-32 px-6 relative border-t border-white/5 bg-black"
    >
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3/4 h-[500px] bg-emerald-500/5 blur-[120px] pointer-events-none rounded-full" />

      <div className="max-w-7xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.8 }}
          className="mb-20 text-center md:text-left"
        >
          <p className="text-xs font-syncopate uppercase tracking-[0.3em] text-emerald-400/70 mb-4">
            The Engine
          </p>
          <h2 className="text-3xl md:text-5xl font-syncopate font-bold uppercase tracking-[0.15em] text-metallic mb-6">
            BUILT ON SCIENCE
          </h2>
          <p className="text-sm md:text-base uppercase tracking-[0.2em] text-white/50 max-w-2xl">
            Six ML models trained on real agricultural data. Validated against
            ICAR and state agriculture department benchmarks.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-y-16 gap-x-8">
          {specifications.map((spec, i) => (
            <motion.div
              key={spec.label}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: i * 0.1 }}
              className="border-l border-emerald-400/20 pl-6 hover:border-emerald-400 transition-colors duration-500 group"
            >
              <h3 className="text-xs font-syncopate uppercase tracking-[0.2em] text-white/40 mb-4 group-hover:text-emerald-400/80 transition-colors">
                {spec.label}
              </h3>
              <div className="text-4xl md:text-5xl font-bold tracking-tight text-white mb-2 group-hover:text-metallic transition-all duration-300">
                {spec.value}
              </div>
              <p className="text-sm text-white/50">{spec.detail}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
