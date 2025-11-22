"use client";

import { useState, useEffect } from "react";
import { Key, Copy, Check, Trash2, Plus, AlertTriangle, X } from "lucide-react";
import { getApiKeys, createApiKey, revokeApiKey, ApiKey } from "@/lib/api";

export default function SettingsPage() {
    const [keys, setKeys] = useState<ApiKey[]>([]);
    const [loading, setLoading] = useState(true);
    const [copied, setCopied] = useState<string | null>(null);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [newKeyName, setNewKeyName] = useState("");
    const [createdKey, setCreatedKey] = useState<ApiKey | null>(null);

    useEffect(() => {
        fetchKeys();
    }, []);

    const fetchKeys = async () => {
        try {
            const data = await getApiKeys();
            setKeys(data);
        } catch (error) {
            console.error("Failed to fetch keys:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleCreateKey = async () => {
        try {
            const key = await createApiKey(newKeyName);
            setCreatedKey(key);
            setNewKeyName("");
            fetchKeys();
        } catch (error) {
            console.error("Failed to create key:", error);
        }
    };

    const handleRevokeKey = async (keyId: string) => {
        if (!confirm("Are you sure you want to revoke this key? This action cannot be undone.")) return;
        try {
            await revokeApiKey(keyId);
            fetchKeys();
        } catch (error) {
            console.error("Failed to revoke key:", error);
        }
    };

    const handleCopy = (text: string) => {
        navigator.clipboard.writeText(text);
        setCopied(text);
        setTimeout(() => setCopied(null), 2000);
    };

    return (
        <div className="space-y-8 relative">
            <div>
                <h1 className="text-3xl font-bold text-white">Settings</h1>
                <p className="mt-2 text-slate-400">Manage your account and API keys.</p>
            </div>

            <div className="rounded-lg bg-slate-900 border border-slate-800">
                <div className="px-6 py-4 border-b border-slate-800 flex justify-between items-center">
                    <div>
                        <h2 className="text-lg font-medium text-white">API Keys</h2>
                        <p className="text-sm text-slate-400">Use these keys to authenticate with the Nexa Forge SDK.</p>
                    </div>
                    <button
                        onClick={() => setShowCreateModal(true)}
                        className="flex items-center px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-md text-sm font-medium transition-colors"
                    >
                        <Plus className="h-4 w-4 mr-2" /> Generate New Key
                    </button>
                </div>
                <div className="p-6">
                    {loading ? (
                        <div className="text-slate-400">Loading keys...</div>
                    ) : keys.length === 0 ? (
                        <div className="text-slate-400">No API keys found. Generate one to get started.</div>
                    ) : (
                        <div className="space-y-4">
                            {keys.map((key) => (
                                <div key={key.key_id} className="flex items-center justify-between p-4 rounded-lg bg-slate-950 border border-slate-800">
                                    <div className="flex items-center space-x-4">
                                        <div className="p-2 rounded-lg bg-slate-900">
                                            <Key className="h-5 w-5 text-cyan-400" />
                                        </div>
                                        <div>
                                            <p className="font-medium text-white">{key.name}</p>
                                            <p className="text-sm font-mono text-slate-500">{key.prefix}</p>
                                        </div>
                                    </div>
                                    <div className="flex items-center space-x-6">
                                        <div className="text-right text-sm">
                                            <p className="text-slate-400">Created: {new Date(key.created_at).toLocaleDateString()}</p>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <button
                                                onClick={() => handleCopy(key.prefix)}
                                                className="p-2 text-slate-400 hover:text-white transition-colors"
                                                title="Copy Prefix"
                                            >
                                                {copied === key.prefix ? <Check className="h-4 w-4 text-green-400" /> : <Copy className="h-4 w-4" />}
                                            </button>
                                            <button
                                                onClick={() => handleRevokeKey(key.key_id)}
                                                className="p-2 text-slate-400 hover:text-red-400 transition-colors"
                                                title="Revoke Key"
                                            >
                                                <Trash2 className="h-4 w-4" />
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            <div className="rounded-lg bg-slate-900 border border-slate-800 p-6">
                <h2 className="text-lg font-medium text-white mb-4">SDK Usage</h2>
                <div className="bg-slate-950 rounded-lg p-4 border border-slate-800 font-mono text-sm text-slate-300 overflow-x-auto">
                    <p className="text-slate-500 mb-2"># Install the SDK</p>
                    <p className="mb-4">pip install nexa-forge</p>

                    <p className="text-slate-500 mb-2"># Initialize client</p>
                    <p><span className="text-purple-400">from</span> nexa_forge <span className="text-purple-400">import</span> NexaForgeClient</p>
                    <p className="mb-4">client = NexaForgeClient(api_key=<span className="text-green-400">"YOUR_API_KEY"</span>)</p>

                    <p className="text-slate-500 mb-2"># Submit a job</p>
                    <p>job = client.generate(domain=<span className="text-green-400">"biology"</span>, num_samples=<span className="text-orange-400">100</span>)</p>
                    <p>print(f<span className="text-green-400">"Job started: &#123;job['job_id']&#125;"</span>)</p>
                </div>
            </div>

            {/* Create Key Modal */}
            {showCreateModal && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="bg-slate-900 rounded-lg border border-slate-800 p-6 w-full max-w-md">
                        <h3 className="text-xl font-bold text-white mb-4">Generate New API Key</h3>
                        {!createdKey ? (
                            <>
                                <div className="mb-4">
                                    <label className="block text-sm font-medium text-slate-400 mb-2">Key Name</label>
                                    <input
                                        type="text"
                                        value={newKeyName}
                                        onChange={(e) => setNewKeyName(e.target.value)}
                                        placeholder="e.g. Production App"
                                        className="w-full bg-slate-950 border border-slate-800 rounded-md px-3 py-2 text-white focus:outline-none focus:border-cyan-500"
                                    />
                                </div>
                                <div className="flex justify-end space-x-3">
                                    <button
                                        onClick={() => setShowCreateModal(false)}
                                        className="px-4 py-2 text-slate-400 hover:text-white"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={handleCreateKey}
                                        disabled={!newKeyName}
                                        className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-md disabled:opacity-50"
                                    >
                                        Generate Key
                                    </button>
                                </div>
                            </>
                        ) : (
                            <div className="space-y-4">
                                <div className="bg-yellow-900/20 border border-yellow-900/50 rounded-md p-4 flex items-start">
                                    <AlertTriangle className="h-5 w-5 text-yellow-500 mr-3 flex-shrink-0 mt-0.5" />
                                    <p className="text-sm text-yellow-200">
                                        This is the only time you will see this key. Please copy it and store it securely. You won't be able to see it again!
                                    </p>
                                </div>

                                <div className="bg-slate-950 border border-slate-800 rounded-md p-3 flex items-center justify-between">
                                    <code className="text-green-400 font-mono break-all">{createdKey.raw_key}</code>
                                    <button
                                        onClick={() => handleCopy(createdKey.raw_key!)}
                                        className="ml-3 p-2 text-slate-400 hover:text-white"
                                    >
                                        {copied === createdKey.raw_key ? <Check className="h-4 w-4 text-green-400" /> : <Copy className="h-4 w-4" />}
                                    </button>
                                </div>

                                <div className="flex justify-end">
                                    <button
                                        onClick={() => {
                                            setShowCreateModal(false);
                                            setCreatedKey(null);
                                        }}
                                        className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-md"
                                    >
                                        I've copied it
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
