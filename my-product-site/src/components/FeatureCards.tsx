"use client";

import React from "react";
import { motion } from "framer-motion";

const features = [
  {
    tag: "Intelligence",
    title: "ML-Powered Recommendations",
    description:
      "Six machine learning models compete head-to-head. The best is auto-selected by cross-validated F1-macro score. Your soil parameters go in, the top 5 crops come out — ranked by suitability for your exact conditions.",
    metric: "96%",
    metricLabel: "Accuracy",
  },
  {
    tag: "Precision",
    title: "Region-Aware Analysis",
    description:
      "Every recommendation is localised. State and district-level soil profiles, agro-climatic zone defaults, and yield history ensure the model knows the difference between Rajasthan's arid plains and Kerala's coastal tropics.",
    metric: "28+",
    metricLabel: "States",
  },
  {
    tag: "Defense",
    title: "Risk & Disease Engine",
    description:
      "A curated knowledge base of 120+ crop-disease entries from ICAR and NIPHM publications. Each crop gets a composite risk score blending climate vulnerability and disease severity — with actionable prevention measures.",
    metric: "120+",
    metricLabel: "Risk Factors",
  },
  {
    tag: "Economics",
    title: "Market Price Integration",
    description:
      "Live market prices from data.gov.in feed into yield-adjusted production estimates. See estimated output in kg, current price per kg in rupees, and sale quantities — grounded in real regional economics.",
    metric: "Live",
    metricLabel: "Market Data",
  },
];

export default function FeatureCards() {
  return (
    <section id="features" className="py-32 px-6 bg-black">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row gap-12 justify-between items-end mb-20">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="flex-1"
          >
            <p className="text-xs font-syncopate uppercase tracking-[0.3em] text-emerald-400/70 mb-4">
              Capabilities
            </p>
            <h2 className="text-3xl md:text-5xl font-syncopate font-bold uppercase tracking-[0.15em] text-metallic mb-6">
              FOUR PILLARS
            </h2>
            <p className="text-sm md:text-base uppercase tracking-[0.2em] text-white/50">
              From soil to market — every decision backed by data.
            </p>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.6, delay: i * 0.15 }}
              className="bg-white/[0.02] border border-white/10 rounded-2xl p-8 hover:bg-white/[0.04] hover:border-emerald-400/30 transition-all duration-500 group relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />

              <div className="flex justify-between items-start mb-12 relative z-10">
                <div className="text-xs font-syncopate tracking-[0.2em] text-emerald-400/60 uppercase bg-emerald-400/5 px-3 py-1 rounded-full border border-emerald-400/10">
                  {feature.tag}
                </div>
                <div className="text-right">
                  <div className="text-2xl font-syncopate font-bold text-metallic opacity-30 group-hover:opacity-100 transition-opacity">
                    {feature.metric}
                  </div>
                  <div className="text-[10px] uppercase tracking-widest text-white/30 mt-1">
                    {feature.metricLabel}
                  </div>
                </div>
              </div>

              <h3 className="text-xl md:text-2xl font-bold tracking-wider mb-4 text-white relative z-10">
                {feature.title}
              </h3>

              <p className="text-white/50 leading-relaxed relative z-10">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
