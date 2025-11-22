"use client";

import { useEffect, useState } from "react";
import { Server, Cpu, Activity, Power } from "lucide-react";
import { Worker } from "@/lib/api";
import clsx from "clsx";

export default function WorkersPage() {
    const [workers, setWorkers] = useState<Worker[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function fetchWorkers() {
            try {
                const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/workers`);
                if (res.ok) {
                    const data = await res.json();
                    setWorkers(data);
                }
            } catch (error) {
                console.error("Failed to fetch workers:", error);
            } finally {
                setLoading(false);
            }
        }
        fetchWorkers();
        const interval = setInterval(fetchWorkers, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-3xl font-bold text-white">Workers</h1>
                <p className="mt-2 text-slate-400">Monitor your GPU compute fleet.</p>
            </div>

            {loading && workers.length === 0 ? (
                <div className="text-slate-400">Loading workers...</div>
            ) : workers.length === 0 ? (
                <div className="text-center py-12 bg-slate-900 rounded-lg border border-slate-800">
                    <Server className="mx-auto h-12 w-12 text-slate-600" />
                    <h3 className="mt-4 text-lg font-medium text-white">No Workers Connected</h3>
                    <p className="mt-2 text-slate-400">Start a worker agent to see it here.</p>
                </div>
            ) : (
                <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                    {workers.map((worker) => (
                        <div key={worker.worker_id} className="bg-slate-900 rounded-lg border border-slate-800 p-6 relative overflow-hidden">
                            <div className={clsx(
                                "absolute top-0 right-0 w-24 h-24 -mr-8 -mt-8 rounded-full opacity-10",
                                worker.status === 'busy' ? "bg-blue-500" :
                                    worker.status === 'idle' ? "bg-green-500" : "bg-slate-500"
                            )} />

                            <div className="flex justify-between items-start mb-4">
                                <div>
                                    <h3 className="text-lg font-bold text-white">{worker.hostname}</h3>
                                    <p className="text-xs font-mono text-slate-500">{worker.worker_id}</p>
                                </div>
                                <span className={clsx(
                                    "px-2 py-1 rounded-full text-xs font-medium capitalize flex items-center",
                                    worker.status === 'busy' ? "bg-blue-900/50 text-blue-300 border border-blue-800" :
                                        worker.status === 'idle' ? "bg-green-900/50 text-green-300 border border-green-800" :
                                            "bg-slate-800 text-slate-400 border border-slate-700"
                                )}>
                                    <span className={clsx(
                                        "w-1.5 h-1.5 rounded-full mr-1.5",
                                        worker.status === 'busy' ? "bg-blue-400 animate-pulse" :
                                            worker.status === 'idle' ? "bg-green-400" : "bg-slate-400"
                                    )} />
                                    {worker.status}
                                </span>
                            </div>

                            <div className="space-y-3">
                                <div className="flex items-center text-sm text-slate-300">
                                    <Cpu className="h-4 w-4 mr-2 text-slate-500" />
                                    {worker.gpu_count}x {worker.gpu_type || 'Unknown GPU'}
                                </div>
                                <div className="flex items-center text-sm text-slate-300">
                                    <Activity className="h-4 w-4 mr-2 text-slate-500" />
                                    Last heartbeat: {new Date(worker.last_heartbeat).toLocaleTimeString()}
                                </div>
                                {worker.current_job_id && (
                                    <div className="mt-4 pt-4 border-t border-slate-800">
                                        <p className="text-xs text-slate-500 uppercase tracking-wider font-bold mb-1">Current Job</p>
                                        <p className="text-sm font-mono text-cyan-400 truncate">{worker.current_job_id}</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
