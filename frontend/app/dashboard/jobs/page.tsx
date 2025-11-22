"use client";

import React, { useEffect, useState } from "react";
import { Play, Filter, RefreshCw, Clock, CheckCircle, XCircle, AlertCircle, ChevronDown, ChevronUp, Terminal } from "lucide-react";
import { getJobs, submitJob, Job } from "@/lib/api";
import clsx from "clsx";

export default function JobsPage() {
    const [jobs, setJobs] = useState<Job[]>([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState<string>('all');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [expandedJob, setExpandedJob] = useState<string | null>(null);

    const fetchJobs = async () => {
        setLoading(true);
        try {
            const data = await getJobs(0, 50);
            setJobs(data);
        } catch (error) {
            console.error("Failed to fetch jobs:", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchJobs();
        const interval = setInterval(fetchJobs, 5000);
        return () => clearInterval(interval);
    }, []);

    const handleCreateTestJob = async () => {
        setIsSubmitting(true);
        try {
            await submitJob('generate', { domain: 'test', num_samples: 10 });
            await fetchJobs();
        } catch (error) {
            console.error("Failed to create job:", error);
        } finally {
            setIsSubmitting(false);
        }
    };

    const filteredJobs = filter === 'all' ? jobs : jobs.filter(j => j.status === filter);

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'completed': return <CheckCircle className="h-4 w-4 text-green-400" />;
            case 'failed': return <XCircle className="h-4 w-4 text-red-400" />;
            case 'running': return <RefreshCw className="h-4 w-4 text-blue-400 animate-spin" />;
            case 'pending': return <Clock className="h-4 w-4 text-slate-400" />;
            default: return <AlertCircle className="h-4 w-4 text-slate-400" />;
        }
    };

    const getMockLogs = (job: Job) => {
        if (job.status === 'failed') {
            return `[2024-11-21 21:00:00] INFO: Starting job ${job.job_id}
[2024-11-21 21:00:01] INFO: Allocating GPU worker...
[2024-11-21 21:00:05] INFO: Worker assigned: worker-gpu-01
[2024-11-21 21:00:10] INFO: Downloading dataset...
[2024-11-21 21:01:30] ERROR: Connection timeout to S3 bucket
[2024-11-21 21:01:31] ERROR: Job failed: Unable to access dataset
[2024-11-21 21:01:31] INFO: Cleaning up resources...`;
        } else if (job.status === 'completed') {
            return `[2024-11-21 20:50:00] INFO: Starting job ${job.job_id}
[2024-11-21 20:50:01] INFO: Allocating GPU worker...
[2024-11-21 20:50:05] INFO: Worker assigned: worker-gpu-02
[2024-11-21 20:50:10] INFO: Loading model...
[2024-11-21 20:51:00] INFO: Processing batch 1/10...
[2024-11-21 20:52:00] INFO: Processing batch 5/10...
[2024-11-21 20:53:00] INFO: Processing batch 10/10...
[2024-11-21 20:53:30] INFO: Uploading results to S3...
[2024-11-21 20:53:45] INFO: Job completed successfully`;
        } else if (job.status === 'running') {
            return `[2024-11-21 21:08:00] INFO: Starting job ${job.job_id}
[2024-11-21 21:08:01] INFO: Allocating GPU worker...
[2024-11-21 21:08:05] INFO: Worker assigned: worker-gpu-01
[2024-11-21 21:08:10] INFO: Loading model...
[2024-11-21 21:09:00] INFO: Processing batch 3/20...
[2024-11-21 21:09:20] INFO: GPU utilization: 95%`;
        } else {
            return `[2024-11-21 21:09:00] INFO: Job ${job.job_id} queued
[2024-11-21 21:09:01] INFO: Waiting for available worker...`;
        }
    };

    return (
        <div className="space-y-8">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-white">Jobs</h1>
                    <p className="mt-2 text-slate-400">Manage and monitor your AI pipelines.</p>
                </div>
                <button
                    onClick={handleCreateTestJob}
                    disabled={isSubmitting}
                    className="flex items-center px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-md font-medium transition-colors disabled:opacity-50"
                >
                    <Play className="h-4 w-4 mr-2" />
                    {isSubmitting ? 'Starting...' : 'New Test Job'}
                </button>
            </div>

            <div className="flex space-x-2 border-b border-slate-800 pb-4">
                {['all', 'running', 'completed', 'failed'].map((f) => (
                    <button
                        key={f}
                        onClick={() => setFilter(f)}
                        className={clsx(
                            "px-3 py-1.5 rounded-md text-sm font-medium capitalize transition-colors",
                            filter === f
                                ? "bg-slate-800 text-white"
                                : "text-slate-400 hover:text-white hover:bg-slate-800/50"
                        )}
                    >
                        {f}
                    </button>
                ))}
            </div>

            <div className="rounded-lg bg-slate-900 border border-slate-800 overflow-hidden">
                <table className="min-w-full text-left text-sm">
                    <thead className="bg-slate-950/50">
                        <tr>
                            <th className="px-6 py-3 font-medium text-slate-400 w-8"></th>
                            <th className="px-6 py-3 font-medium text-slate-400">Job ID</th>
                            <th className="px-6 py-3 font-medium text-slate-400">Type</th>
                            <th className="px-6 py-3 font-medium text-slate-400">Status</th>
                            <th className="px-6 py-3 font-medium text-slate-400">Created</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                        {loading && jobs.length === 0 ? (
                            <tr>
                                <td colSpan={5} className="px-6 py-8 text-center text-slate-500">Loading jobs...</td>
                            </tr>
                        ) : filteredJobs.length === 0 ? (
                            <tr>
                                <td colSpan={5} className="px-6 py-8 text-center text-slate-500">No jobs found.</td>
                            </tr>
                        ) : (
                            filteredJobs.map((job) => (
                                <React.Fragment key={job.job_id}>
                                    <tr className="hover:bg-slate-800/50 transition-colors">
                                        <td className="px-6 py-4">
                                            <button
                                                onClick={() => setExpandedJob(expandedJob === job.job_id ? null : job.job_id)}
                                                className="text-slate-400 hover:text-white"
                                            >
                                                {expandedJob === job.job_id ?
                                                    <ChevronUp className="h-4 w-4" /> :
                                                    <ChevronDown className="h-4 w-4" />
                                                }
                                            </button>
                                        </td>
                                        <td className="px-6 py-4 font-mono text-slate-300">{job.job_id}</td>
                                        <td className="px-6 py-4">
                                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-800 text-slate-300 capitalize">
                                                {job.job_type}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex items-center space-x-2">
                                                {getStatusIcon(job.status)}
                                                <span className={clsx(
                                                    "capitalize",
                                                    job.status === 'completed' ? "text-green-400" :
                                                        job.status === 'failed' ? "text-red-400" :
                                                            job.status === 'running' ? "text-blue-400" :
                                                                "text-slate-400"
                                                )}>
                                                    {job.status}
                                                </span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-slate-400">
                                            {new Date(job.created_at).toLocaleString()}
                                        </td>
                                    </tr>
                                    {expandedJob === job.job_id && (
                                        <tr className="bg-slate-950/50">
                                            <td colSpan={5} className="px-6 py-4">
                                                <div className="space-y-4">
                                                    <div className="grid grid-cols-2 gap-4">
                                                        <div>
                                                            <p className="text-xs text-slate-500 uppercase font-bold mb-1">Status</p>
                                                            <p className="text-slate-300 capitalize">{job.status}</p>
                                                        </div>
                                                        <div>
                                                            <p className="text-xs text-slate-500 uppercase font-bold mb-1">Worker</p>
                                                            <p className="text-slate-300 font-mono">{job.worker_id || 'Not assigned'}</p>
                                                        </div>
                                                        <div>
                                                            <p className="text-xs text-slate-500 uppercase font-bold mb-1">Created</p>
                                                            <p className="text-slate-300">{new Date(job.created_at).toLocaleString()}</p>
                                                        </div>
                                                        <div>
                                                            <p className="text-xs text-slate-500 uppercase font-bold mb-1">User</p>
                                                            <p className="text-slate-300">{job.user_id}</p>
                                                        </div>
                                                    </div>

                                                    <div>
                                                        <div className="flex items-center mb-2">
                                                            <Terminal className="h-4 w-4 text-slate-500 mr-2" />
                                                            <p className="text-xs text-slate-500 uppercase font-bold">Logs</p>
                                                        </div>
                                                        <div className="bg-slate-900 rounded-md p-4 border border-slate-800 font-mono text-xs text-slate-300 overflow-x-auto max-h-64 overflow-y-auto">
                                                            <pre className="whitespace-pre-wrap">{getMockLogs(job)}</pre>
                                                        </div>
                                                    </div>
                                                </div>
                                            </td>
                                        </tr>
                                    )}
                                </React.Fragment>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
