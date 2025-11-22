"use client";

import { Activity, CheckCircle, Clock, Server, XCircle, TrendingUp } from "lucide-react";
import { MetricsCard } from "@/components/MetricsCard";

// Simulated data for dashboard demo
const simulatedJobs = [
    { job_id: "job_a1b2c3d4", job_type: "generate", status: "completed", created_at: "2024-11-21T20:30:00Z" },
    { job_id: "job_e5f6g7h8", job_type: "train", status: "running", created_at: "2024-11-21T20:45:00Z" },
    { job_id: "job_i9j0k1l2", job_type: "distill", status: "running", created_at: "2024-11-21T20:50:00Z" },
    { job_id: "job_m3n4o5p6", job_type: "evaluate", status: "completed", created_at: "2024-11-21T19:15:00Z" },
    { job_id: "job_q7r8s9t0", job_type: "train", status: "failed", created_at: "2024-11-21T21:00:00Z" },
    { job_id: "job_u1v2w3x4", job_type: "audit", status: "completed", created_at: "2024-11-21T18:30:00Z" },
    { job_id: "job_y5z6a7b8", job_type: "generate", status: "failed", created_at: "2024-11-21T20:15:00Z" },
    { job_id: "job_c9d0e1f2", job_type: "deploy", status: "completed", created_at: "2024-11-21T17:00:00Z" },
];

export default function DashboardPage() {
    const activeJobs = simulatedJobs.filter(j => j.status === 'running').length;
    const completedJobs = simulatedJobs.filter(j => j.status === 'completed').length;
    const failedJobs = simulatedJobs.filter(j => j.status === 'failed').length;
    const totalJobs = simulatedJobs.length;

    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-3xl font-bold text-white">Dashboard</h1>
                <p className="mt-2 text-slate-400">Overview of your AI foundry operations.</p>
            </div>

            <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
                <MetricsCard
                    title="Active Jobs"
                    value={activeJobs}
                    icon={Activity}
                    trend="+2 today"
                    trendUp={true}
                />
                <MetricsCard
                    title="Completed"
                    value={completedJobs}
                    icon={CheckCircle}
                    trend="+12% this week"
                    trendUp={true}
                />
                <MetricsCard
                    title="Failed"
                    value={failedJobs}
                    icon={XCircle}
                    trend="2 in last 24h"
                    trendUp={false}
                />
                <MetricsCard
                    title="Active Workers"
                    value="3"
                    icon={Server}
                    trend="All healthy"
                    trendUp={true}
                />
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
                <div className="rounded-lg bg-slate-900 border border-slate-800">
                    <div className="px-6 py-4 border-b border-slate-800">
                        <h2 className="text-lg font-medium text-white">Recent Activity</h2>
                    </div>
                    <div className="p-6">
                        <div className="space-y-4">
                            {simulatedJobs.slice(0, 6).map((job) => (
                                <div key={job.job_id} className="flex items-center justify-between border-b border-slate-800 pb-4 last:border-0 last:pb-0">
                                    <div>
                                        <p className="font-medium text-white capitalize">{job.job_type} Job</p>
                                        <p className="text-sm text-slate-400 font-mono">{job.job_id}</p>
                                    </div>
                                    <div className="flex items-center space-x-4">
                                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${job.status === 'completed' ? 'bg-green-900 text-green-300' :
                                                job.status === 'running' ? 'bg-blue-900 text-blue-300' :
                                                    job.status === 'failed' ? 'bg-red-900 text-red-300' :
                                                        'bg-slate-800 text-slate-300'
                                            }`}>
                                            {job.status}
                                        </span>
                                        <span className="text-sm text-slate-500">
                                            {new Date(job.created_at).toLocaleTimeString()}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="rounded-lg bg-slate-900 border border-slate-800">
                    <div className="px-6 py-4 border-b border-slate-800">
                        <h2 className="text-lg font-medium text-white">Performance Metrics</h2>
                    </div>
                    <div className="p-6 space-y-6">
                        <div>
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-sm text-slate-400">Success Rate</span>
                                <span className="text-sm font-medium text-green-400">
                                    {((completedJobs / totalJobs) * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div className="w-full bg-slate-800 rounded-full h-2">
                                <div
                                    className="bg-green-500 h-2 rounded-full"
                                    style={{ width: `${(completedJobs / totalJobs) * 100}%` }}
                                />
                            </div>
                        </div>

                        <div>
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-sm text-slate-400">GPU Utilization</span>
                                <span className="text-sm font-medium text-cyan-400">87%</span>
                            </div>
                            <div className="w-full bg-slate-800 rounded-full h-2">
                                <div className="bg-cyan-500 h-2 rounded-full" style={{ width: '87%' }} />
                            </div>
                        </div>

                        <div>
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-sm text-slate-400">Avg Job Duration</span>
                                <span className="text-sm font-medium text-slate-300">12.5 min</span>
                            </div>
                            <div className="w-full bg-slate-800 rounded-full h-2">
                                <div className="bg-purple-500 h-2 rounded-full" style={{ width: '65%' }} />
                            </div>
                        </div>

                        <div className="pt-4 border-t border-slate-800">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center">
                                    <TrendingUp className="h-5 w-5 text-green-400 mr-2" />
                                    <span className="text-sm text-slate-400">Cost Efficiency</span>
                                </div>
                                <span className="text-sm font-medium text-green-400">+18% vs last week</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="rounded-lg bg-gradient-to-r from-cyan-900/20 to-blue-900/20 border border-cyan-800/30 p-6">
                <div className="flex items-start justify-between">
                    <div>
                        <h3 className="text-lg font-medium text-white mb-2">🚀 Ready to scale?</h3>
                        <p className="text-slate-300">Install the SDK and start submitting jobs programmatically.</p>
                    </div>
                    <a
                        href="/docs"
                        className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-md text-sm font-medium transition-colors"
                    >
                        View Docs
                    </a>
                </div>
            </div>
        </div>
    );
}
