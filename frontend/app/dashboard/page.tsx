"use client";

import { Activity, CheckCircle, Clock, Server, XCircle, TrendingUp } from "lucide-react";
import { MetricsCard } from "@/components/MetricsCard";
import { useEffect, useState } from "react";
import { getJobs, getHealth } from "@/lib/api";

export default function DashboardPage() {
    const [loading, setLoading] = useState(true);
    const [activeJobs, setActiveJobs] = useState(0);
    const [completedJobs, setCompletedJobs] = useState(0);
    const [failedJobs, setFailedJobs] = useState(0);
    const [totalJobs, setTotalJobs] = useState(0);
    const [recentJobs, setRecentJobs] = useState<any[]>([]);
    const [apiHealthy, setApiHealthy] = useState(false);

    useEffect(() => {
        async function fetchData() {
            try {
                // Check API health
                const health = await getHealth();
                setApiHealthy(health.status === "healthy");

                // Fetch jobs
                const jobs = await getJobs(0, 10);
                setRecentJobs(jobs);
                
                const active = jobs.filter((j: any) => j.status === 'running' || j.status === 'pending').length;
                const completed = jobs.filter((j: any) => j.status === 'completed').length;
                const failed = jobs.filter((j: any) => j.status === 'failed').length;
                
                setActiveJobs(active);
                setCompletedJobs(completed);
                setFailedJobs(failed);
                setTotalJobs(jobs.length);
            } catch (error) {
                console.error("Failed to fetch dashboard data:", error);
                setApiHealthy(false);
            } finally {
                setLoading(false);
            }
        }

        fetchData();
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    const successRate = totalJobs > 0 ? ((completedJobs / totalJobs) * 100).toFixed(0) : 0;

    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-3xl font-bold text-white">Dashboard</h1>
                <p className="mt-2 text-slate-400">Overview of your AI foundry operations.</p>
            </div>

            {!apiHealthy && (
                <div className="rounded-lg bg-yellow-900/20 border border-yellow-800/50 p-4">
                    <p className="text-yellow-400 text-sm">
                        API backend not connected. Make sure the backend is running at {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
                    </p>
                </div>
            )}

            <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
                <MetricsCard
                    title="Active Jobs"
                    value={loading ? "..." : activeJobs}
                    icon={Activity}
                    trend={loading ? "" : `${activeJobs} active`}
                    trendUp={true}
                />
                <MetricsCard
                    title="Completed"
                    value={loading ? "..." : completedJobs}
                    icon={CheckCircle}
                    trend={loading ? "" : `${completedJobs} total`}
                    trendUp={true}
                />
                <MetricsCard
                    title="Failed"
                    value={loading ? "..." : failedJobs}
                    icon={XCircle}
                    trend={loading ? "" : `${failedJobs} total`}
                    trendUp={false}
                />
                <MetricsCard
                    title="API Status"
                    value={apiHealthy ? "Online" : "Offline"}
                    icon={Server}
                    trend={apiHealthy ? "Connected" : "Disconnected"}
                    trendUp={apiHealthy}
                />
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
                <div className="rounded-lg bg-slate-900 border border-slate-800">
                    <div className="px-6 py-4 border-b border-slate-800">
                        <h2 className="text-lg font-medium text-white">Recent Activity</h2>
                    </div>
                    <div className="p-6">
                        {loading ? (
                            <div className="text-center py-8 text-slate-500">Loading jobs...</div>
                        ) : recentJobs.length === 0 ? (
                            <div className="text-center py-8 text-slate-500">No jobs yet. Submit a job to get started.</div>
                        ) : (
                            <div className="space-y-4">
                                {recentJobs.slice(0, 6).map((job) => (
                                    <div key={job.job_id} className="flex items-center justify-between border-b border-slate-800 pb-4 last:border-0 last:pb-0">
                                        <div>
                                            <p className="font-medium text-white capitalize">{job.job_type} Job</p>
                                            <p className="text-sm text-slate-400 font-mono">{job.job_id}</p>
                                        </div>
                                        <div className="flex items-center space-x-4">
                                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                                                job.status === 'completed' ? 'bg-green-900 text-green-300' :
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
                        )}
                    </div>
                </div>

                <div className="rounded-lg bg-slate-900 border border-slate-800">
                    <div className="px-6 py-4 border-b border-slate-800">
                        <h2 className="text-lg font-medium text-white">Performance Metrics</h2>
                    </div>
                    <div className="p-6 space-y-6">
                        {loading || totalJobs === 0 ? (
                            <div className="text-center py-8 text-slate-500">No data available</div>
                        ) : (
                            <>
                                <div>
                                    <div className="flex justify-between items-center mb-2">
                                        <span className="text-sm text-slate-400">Success Rate</span>
                                        <span className="text-sm font-medium text-green-400">
                                            {successRate}%
                                        </span>
                                    </div>
                                    <div className="w-full bg-slate-800 rounded-full h-2">
                                        <div
                                            className="bg-green-500 h-2 rounded-full"
                                            style={{ width: `${successRate}%` }}
                                        />
                                    </div>
                                </div>

                                <div className="pt-4 border-t border-slate-800">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center">
                                            <TrendingUp className="h-5 w-5 text-green-400 mr-2" />
                                            <span className="text-sm text-slate-400">Total Jobs</span>
                                        </div>
                                        <span className="text-sm font-medium text-white">{totalJobs}</span>
                                    </div>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </div>

            <div className="rounded-lg bg-gradient-to-r from-cyan-900/20 to-blue-900/20 border border-cyan-800/30 p-6">
                <div className="flex items-start justify-between">
                    <div>
                        <h3 className="text-lg font-medium text-white mb-2">Ready to scale?</h3>
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
