"use client";

import { DollarSign, TrendingUp, CreditCard } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";
import { useEffect, useState } from "react";
import { getBillingSummary, BillingSummary } from "@/lib/api";

export default function BillingPage() {
  const [loading, setLoading] = useState(true);
  const [billing, setBilling] = useState<BillingSummary | null>(null);

  useEffect(() => {
    async function fetchBilling() {
      try {
        const data = await getBillingSummary();
        setBilling(data);
      } catch (error) {
        console.error("Failed to fetch billing data:", error);
      } finally {
        setLoading(false);
      }
    }
    fetchBilling();
  }, []);

  const usageData = billing ? Object.entries(billing.usage_by_type).map(([name, value]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    value: value,
    color: name === 'train' ? "#06b6d4" : name === 'evaluate' ? "#8b5cf6" : name === 'generate' ? "#10b981" : "#f59e0b"
  })) : [];

  return (
    <div className="space-y-8">
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-3xl font-bold text-white">Billing & Usage</h1>
        </div>
        <p className="mt-2 text-slate-400">Monitor your usage and costs.</p>
      </div>

      {loading ? (
        <div className="text-center py-12 text-slate-500">Loading billing data...</div>
      ) : !billing ? (
        <div className="text-center py-12 bg-slate-900 rounded-lg border border-slate-800">
          <CreditCard className="mx-auto h-12 w-12 text-slate-600" />
          <h3 className="mt-4 text-lg font-medium text-white">No Billing Data</h3>
          <p className="mt-2 text-slate-400">Billing data will appear here once jobs are executed.</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="rounded-lg bg-slate-900 border border-slate-800 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">Current Period</p>
                  <p className="text-3xl font-bold text-white mt-2">
                    {billing.currency === 'USD' ? '$' : ''}{billing.total_cost.toFixed(2)}
                  </p>
                </div>
                <DollarSign className="h-10 w-10 text-cyan-400" />
              </div>
            </div>

            <div className="rounded-lg bg-slate-900 border border-slate-800 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">Period</p>
                  <p className="text-sm font-medium text-white mt-2">
                    {new Date(billing.period_start).toLocaleDateString()} - {new Date(billing.period_end).toLocaleDateString()}
                  </p>
                </div>
                <TrendingUp className="h-10 w-10 text-purple-400" />
              </div>
            </div>

            <div className="rounded-lg bg-slate-900 border border-slate-800 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">Job Types</p>
                  <p className="text-sm font-medium text-white mt-2">
                    {Object.keys(billing.usage_by_type).length} types
                  </p>
                </div>
                <CreditCard className="h-10 w-10 text-green-400" />
              </div>
            </div>
          </div>

          {usageData.length > 0 && (
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
          )}
        </>
      )}
    </div>
  );
}
