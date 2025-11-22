const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Job {
    job_id: string;
    job_type: 'generate' | 'audit' | 'distill' | 'train' | 'evaluate' | 'deploy';
    status: 'pending' | 'provisioning' | 'assigned' | 'running' | 'completed' | 'failed' | 'cancelled';
    created_at: string;
    updated_at: string;
    payload: any;
    user_id?: string;
    worker_id?: string;
    result?: any;
    error?: string;
}

export interface Worker {
    worker_id: string;
    hostname: string;
    status: 'idle' | 'busy' | 'offline' | 'bootstrapping';
    gpu_type?: string;
    gpu_count: number;
    last_heartbeat: string;
    current_job_id?: string;
}

export async function getJobs(skip = 0, limit = 100, status?: string): Promise<Job[]> {
    const params = new URLSearchParams({ skip: skip.toString(), limit: limit.toString() });
    if (status) params.append('status', status);

    const res = await fetch(`${API_URL}/api/jobs?${params}`);
    if (!res.ok) throw new Error('Failed to fetch jobs');
    return res.json();
}

export async function getJob(jobId: string): Promise<Job> {
    const res = await fetch(`${API_URL}/api/jobs/${jobId}`);
    if (!res.ok) throw new Error('Failed to fetch job');
    return res.json();
}

export async function submitJob(type: string, payload: any): Promise<Job> {
    const res = await fetch(`${API_URL}/api/jobs/${type}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ payload }),
    });
    if (!res.ok) throw new Error('Failed to submit job');
    return res.json();
}

export async function getHealth(): Promise<{ status: string; version: string }> {
    const res = await fetch(`${API_URL}/health`);
    if (!res.ok) throw new Error('Failed to check health');
    return res.json();
}

export interface BillingSummary {
    total_cost: number;
    currency: string;
    period_start: string;
    period_end: string;
    usage_by_type: Record<string, number>;
    cost_by_type: Record<string, number>;
}

export async function getBillingSummary(): Promise<BillingSummary> {
    const res = await fetch(`${API_URL}/api/billing/summary`);
    if (!res.ok) throw new Error('Failed to fetch billing summary');
    return res.json();
}

export interface ApiKey {
    key_id: string;
    name: string;
    prefix: string;
    created_at: string;
    raw_key?: string;
}

export async function getApiKeys(): Promise<ApiKey[]> {
    const res = await fetch(`${API_URL}/api/auth/api-keys`);
    if (!res.ok) throw new Error('Failed to fetch API keys');
    return res.json();
}

export async function createApiKey(name: string): Promise<ApiKey> {
    const res = await fetch(`${API_URL}/api/auth/api-keys`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
    });
    if (!res.ok) throw new Error('Failed to create API key');
    return res.json();
}

export async function revokeApiKey(keyId: string): Promise<void> {
    const res = await fetch(`${API_URL}/api/auth/api-keys/${keyId}`, {
        method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to revoke API key');
}
