"use client";

import { useEffect, useState } from "react";
import { DollarSign, Zap, Cpu, Database } from "lucide-react";

interface PricingTier {
  name: string;
  resource: string;
  rate: string;
  description: string;
  icon: any;
}

export default function PricingPage() {
  const pricingTiers: PricingTier[] = [
    {
      name: "GPU Compute",
      resource: "Training & Fine-tuning",
      rate: "$3.00/hour",
      description: "Per GPU hour. Covers compute costs and infrastructure.",
      icon: Cpu,
    },
    {
      name: "GPU Compute",
      resource: "Inference",
      rate: "$1.00/hour",
      description: "Per GPU hour for model serving (lower overhead).",
      icon: Zap,
    },
    {
      name: "Storage",
      resource: "Artifact Storage",
      rate: "$0.05/GB/month",
      description: "For datasets, checkpoints, and reports.",
      icon: Database,
    },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Pricing</h1>
        <p className="mt-2 text-slate-400">Simple pay-as-you-go pricing. No hidden fees.</p>
      </div>

      <div className="rounded-lg bg-slate-900 border border-slate-800 p-6 mb-6">
        <h2 className="text-xl font-bold text-white mb-4">How It Works</h2>
        <p className="text-slate-300 mb-4">
          Nexa Forge uses metered billing. You only pay for what you use:
        </p>
        <ul className="list-disc list-inside space-y-2 text-slate-300">
          <li>GPU hours are tracked automatically when jobs complete</li>
          <li>Billing is calculated per job based on actual runtime</li>
          <li>Storage is billed monthly based on artifact size</li>
          <li>No upfront costs or commitments</li>
        </ul>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {pricingTiers.map((tier, idx) => {
          const Icon = tier.icon;
          return (
            <div key={idx} className="rounded-lg bg-slate-900 border border-slate-800 p-6">
              <div className="flex items-center mb-4">
                <div className="rounded-lg bg-cyan-900/30 p-3 mr-3">
                  <Icon className="h-6 w-6 text-cyan-400" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white">{tier.name}</h3>
                  <p className="text-sm text-slate-400">{tier.resource}</p>
                </div>
              </div>
              <div className="mb-4">
                <p className="text-3xl font-bold text-white">{tier.rate}</p>
              </div>
              <p className="text-sm text-slate-400">{tier.description}</p>
            </div>
          );
        })}
      </div>

      <div className="rounded-lg bg-gradient-to-r from-cyan-900/20 to-blue-900/20 border border-cyan-800/30 p-6">
        <h3 className="text-lg font-medium text-white mb-2">Example Costs</h3>
        <div className="space-y-2 text-slate-300 text-sm">
          <p>• 1-hour training job (1 GPU): <span className="font-medium text-white">$3.00</span></p>
          <p>• 4-hour fine-tuning (2 GPUs): <span className="font-medium text-white">$24.00</span></p>
          <p>• 100GB checkpoint storage (1 month): <span className="font-medium text-white">$5.00</span></p>
        </div>
      </div>
    </div>
  );
}

