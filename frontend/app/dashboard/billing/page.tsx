"use client";

import { DollarSign, TrendingUp, CreditCard } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";

const usageData = [
  { name: "Training", value: 45, color: "#06b6d4" },
  { name: "Evaluation", value: 25, color: "#8b5cf6" },
  { name: "Generation", value: 20, color: "#10b981" },
  { name: "Distillation", value: 10, color: "#f59e0b" },
];

export default function BillingPage() {
  return (
    <div className="space-y-8">
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-3xl font-bold text-white">Billing & Usage</h1>
          <span className="inline-flex items-center rounded-full bg-yellow-500/10 px-2.5 py-0.5 text-xs font-medium text-yellow-400 ring-1 ring-inset ring-yellow-500/20">
            🚧 Under Development
          </span>
        </div>
        <p className="mt-2 text-slate-400">Monitor your usage and costs. (Demo data shown)</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="rounded-lg bg-slate-900 border border-slate-800 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Current Month</p>
              <p className="text-3xl font-bold text-white mt-2">$127.50</p>
            </div>
            <DollarSign className="h-10 w-10 text-cyan-400" />
          </div>
        </div>

        <div className="rounded-lg bg-slate-900 border border-slate-800 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">GPU Hours</p>
              <p className="text-3xl font-bold text-white mt-2">42.5</p>
            </div>
            <TrendingUp className="h-10 w-10 text-purple-400" />
          </div>
        </div>

        <div className="rounded-lg bg-slate-900 border border-slate-800 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Credits Remaining</p>
              <p className="text-3xl font-bold text-white mt-2">$872.50</p>
            </div>
            <CreditCard className="h-10 w-10 text-green-400" />
          </div>
        </div>
      </div>

      <div className="rounded-lg bg-slate-900 border border-slate-800 p-6">
        <h2 className="text-xl font-bold text-white mb-6">Usage Breakdown</h2>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={usageData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
            >
              {usageData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
