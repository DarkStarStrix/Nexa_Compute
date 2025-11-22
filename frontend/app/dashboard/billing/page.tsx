"use client";

import { useEffect, useState } from "react";
import { DollarSign, CreditCard, Database, Zap, Cpu } from "lucide-react";
import { MetricsCard } from "@/components/MetricsCard";
import { getBillingSummary, BillingSummary } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

export default function BillingPage() {
    const [summary, setSummary] = useState<BillingSummary | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function fetchData() {
            try {
                const data = await getBillingSummary();
                setSummary(data);
            } catch (error) {
                console.error("Failed to fetch billing:", error);
            } finally {
                setLoading(false);
            }
        }
        fetchData();
    }, []);

    if (loading) return <div className="text-slate-400">Loading billing data...</div>;
    if (!summary) return <div className="text-red-400">Failed to load billing data.</div>;

    const costData = Object.entries(summary.cost_by_type).map(([name, value]) => ({
        name: name.replace(/_/g, ' ').toUpperCase(),
        value: parseFloat(value.toFixed(2))
    }));

    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-3xl font-bold text-white">Billing & Usage</h1>
                <p className="mt-2 text-slate-400">Track your compute usage and costs.</p>
            </div>

            <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
                <MetricsCard
                    title="Total Cost (MTD)"
                    value={`$${summary.total_cost.toFixed(2)}`}
                    icon={DollarSign}
                    trend="+15%"
                    trendUp={false}
                />
                <MetricsCard
                    title="GPU Hours"
                    value={summary.usage_by_type['gpu_hour']?.toFixed(1) || '0'}
                    icon={Zap}
                />
                <MetricsCard
                    title="Tokens Processed"
                    value={`${((summary.usage_by_type['token_input'] || 0) + (summary.usage_by_type['token_output'] || 0)).toLocaleString()}`}
                    icon={Cpu}
                />
                <MetricsCard
                    title="Storage"
                    value={`${summary.usage_by_type['storage_gb_month']?.toFixed(1) || '0'} GB`}
                    icon={Database}
                />
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
                <div className="rounded-lg bg-slate-900 border border-slate-800 p-6">
                    <h2 className="text-lg font-medium text-white mb-6">Cost Distribution</h2>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={costData}
                                    cx="50%"
                                    cy="50%"
                                    labelLine={false}
                                    label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}
                                    outerRadius={80}
                                    fill="#8884d8"
                                    dataKey="value"
                                >
                                    {costData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#fff' }}
                                    itemStyle={{ color: '#fff' }}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="rounded-lg bg-slate-900 border border-slate-800 p-6">
                    <h2 className="text-lg font-medium text-white mb-6">Usage Breakdown</h2>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={costData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis type="number" stroke="#94a3b8" />
                                <YAxis dataKey="name" type="category" width={150} stroke="#94a3b8" style={{ fontSize: '12px' }} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#fff' }}
                                    cursor={{ fill: '#334155', opacity: 0.2 }}
                                />
                                <Bar dataKey="value" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <div className="rounded-lg bg-slate-900 border border-slate-800">
                <div className="px-6 py-4 border-b border-slate-800 flex justify-between items-center">
                    <h2 className="text-lg font-medium text-white">Invoice History</h2>
                    <button className="text-sm text-cyan-400 hover:text-cyan-300 font-medium">Download All</button>
                </div>
                <div className="p-6">
                    <table className="min-w-full text-left text-sm">
                        <thead>
                            <tr className="text-slate-400 border-b border-slate-800">
                                <th className="pb-3 font-medium">Invoice ID</th>
                                <th className="pb-3 font-medium">Date</th>
                                <th className="pb-3 font-medium">Amount</th>
                                <th className="pb-3 font-medium">Status</th>
                                <th className="pb-3 font-medium text-right">Action</th>
                            </tr>
                        </thead>
                        <tbody className="text-slate-300">
                            <tr className="border-b border-slate-800 last:border-0">
                                <td className="py-4">INV-2024-001</td>
                                <td className="py-4">Oct 1, 2024</td>
                                <td className="py-4">$124.50</td>
                                <td className="py-4"><span className="px-2 py-1 rounded-full bg-green-900 text-green-300 text-xs">Paid</span></td>
                                <td className="py-4 text-right"><CreditCard className="h-4 w-4 inline text-slate-400 hover:text-white cursor-pointer" /></td>
                            </tr>
                            <tr className="border-b border-slate-800 last:border-0">
                                <td className="py-4">INV-2024-002</td>
                                <td className="py-4">Nov 1, 2024</td>
                                <td className="py-4">$245.80</td>
                                <td className="py-4"><span className="px-2 py-1 rounded-full bg-green-900 text-green-300 text-xs">Paid</span></td>
                                <td className="py-4 text-right"><CreditCard className="h-4 w-4 inline text-slate-400 hover:text-white cursor-pointer" /></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
