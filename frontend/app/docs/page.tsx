"use client";

import Link from "next/link";
import { Book, Code, Terminal, FileText, Copy, Check } from "lucide-react";
import { useState } from "react";

export default function DocsPage() {
    const [copied, setCopied] = useState<string | null>(null);

    const handleCopy = (text: string) => {
        navigator.clipboard.writeText(text);
        setCopied(text);
        setTimeout(() => setCopied(null), 2000);
    };

    const CodeBlock = ({ code }: { code: string }) => (
        <div className="relative bg-slate-950 rounded-lg p-4 border border-slate-800 font-mono text-sm text-slate-300 overflow-x-auto group">
            <button
                onClick={() => handleCopy(code)}
                className="absolute top-2 right-2 p-2 rounded-md bg-slate-900 text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity hover:text-white"
            >
                {copied === code ? <Check className="h-4 w-4 text-green-400" /> : <Copy className="h-4 w-4" />}
            </button>
            <pre>{code}</pre>
        </div>
    );

    return (
        <div className="min-h-screen bg-slate-950 text-white p-8">
            <div className="max-w-5xl mx-auto">
                <div className="mb-12">
                    <h1 className="text-4xl font-bold text-white">Documentation</h1>
                    <p className="mt-4 text-xl text-slate-400">Learn how to use Nexa Forge to orchestrate your AI workflows.</p>
                </div>

                <div className="grid gap-8 md:grid-cols-2 mb-16">
                    {/* ... (Previous links remain same) ... */}
                    <Link href="/docs/api" className="block group">
                        <div className="h-full p-6 rounded-xl bg-slate-900 border border-slate-800 hover:border-cyan-500/50 transition-colors">
                            <div className="w-12 h-12 rounded-lg bg-cyan-900/30 flex items-center justify-center mb-4 group-hover:bg-cyan-900/50 transition-colors">
                                <Code className="h-6 w-6 text-cyan-400" />
                            </div>
                            <h2 className="text-xl font-bold text-white mb-2">API Reference</h2>
                            <p className="text-slate-400">Complete reference for the Nexa Forge REST API. Endpoints for jobs, workers, and artifacts.</p>
                        </div>
                    </Link>

                    <Link href="/docs/sdk" className="block group">
                        <div className="h-full p-6 rounded-xl bg-slate-900 border border-slate-800 hover:border-purple-500/50 transition-colors">
                            <div className="w-12 h-12 rounded-lg bg-purple-900/30 flex items-center justify-center mb-4 group-hover:bg-purple-900/50 transition-colors">
                                <Terminal className="h-6 w-6 text-purple-400" />
                            </div>
                            <h2 className="text-xl font-bold text-white mb-2">Python SDK</h2>
                            <p className="text-slate-400">Official Python client for interacting with Nexa Forge programmatically.</p>
                        </div>
                    </Link>
                </div>

                <div className="space-y-12">
                    <section>
                        <h2 className="text-2xl font-bold text-white mb-6">Quickstart</h2>
                        <div className="space-y-6">
                            <div>
                                <h3 className="text-lg font-medium text-white mb-2">1. Install the SDK</h3>
                                <CodeBlock code="pip install nexa-forge" />
                            </div>
                            <div>
                                <h3 className="text-lg font-medium text-white mb-2">2. Initialize Client</h3>
                                <CodeBlock code={`from nexa_forge import NexaForgeClient\n\nclient = NexaForgeClient(api_key="YOUR_API_KEY")`} />
                            </div>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-bold text-white mb-6">Common Workflows</h2>

                        <div className="space-y-8">
                            <div>
                                <h3 className="text-lg font-medium text-cyan-400 mb-2">Data Generation</h3>
                                <p className="text-slate-400 mb-3">Generate synthetic data for a specific domain.</p>
                                <CodeBlock code={`job = client.generate(\n    domain="medical_imaging",\n    num_samples=1000,\n    params={"resolution": "1024x1024"}\n)\nprint(f"Job ID: {job['job_id']}")`} />
                            </div>

                            <div>
                                <h3 className="text-lg font-medium text-purple-400 mb-2">Model Distillation</h3>
                                <p className="text-slate-400 mb-3">Distill a large teacher model into a smaller student model.</p>
                                <CodeBlock code={`job = client.distill(\n    teacher_model="gpt-4",\n    student_model="llama-3-8b",\n    dataset_uri="s3://my-bucket/dataset.parquet"\n)`} />
                            </div>

                            <div>
                                <h3 className="text-lg font-medium text-green-400 mb-2">Evaluation</h3>
                                <p className="text-slate-400 mb-3">Run benchmarks on a trained model.</p>
                                <CodeBlock code={`job = client.evaluate(\n    model_id="my-finetuned-model-v1",\n    benchmark="mmlu"\n)`} />
                            </div>
                        </div>
                    </section>
                </div>
            </div>
        </div>
    );
}
