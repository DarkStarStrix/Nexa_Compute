"use client";

import { Database, FileText, Download } from "lucide-react";

// Mock artifacts for now
const artifacts = [
  { id: "art_1", name: "training_dataset_v1.parquet", type: "dataset", size: "1.2 GB", created: "2024-11-20T10:00:00Z" },
  { id: "art_2", name: "mistral_finetuned_v1.pt", type: "checkpoint", size: "14.5 GB", created: "2024-11-20T14:30:00Z" },
  { id: "art_3", name: "eval_report_run_123.json", type: "report", size: "45 KB", created: "2024-11-20T15:00:00Z" },
];

export default function ArtifactsPage() {
  return (
    <div className="space-y-8">
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-3xl font-bold text-white">Artifacts</h1>
          <span className="inline-flex items-center rounded-full bg-yellow-500/10 px-2.5 py-0.5 text-xs font-medium text-yellow-400 ring-1 ring-inset ring-yellow-500/20">
            🚧 Under Development
          </span>
        </div>
        <p className="mt-2 text-slate-400">Browse datasets, checkpoints, and reports. (Demo data shown)</p>
      </div>

      <div className="rounded-lg bg-slate-900 border border-slate-800 overflow-hidden">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-slate-950/50">
            <tr>
              <th className="px-6 py-3 font-medium text-slate-400">Name</th>
              <th className="px-6 py-3 font-medium text-slate-400">Type</th>
              <th className="px-6 py-3 font-medium text-slate-400">Size</th>
              <th className="px-6 py-3 font-medium text-slate-400">Created</th>
              <th className="px-6 py-3 font-medium text-slate-400 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {artifacts.map((art) => (
              <tr key={art.id} className="hover:bg-slate-800/50 transition-colors">
                <td className="px-6 py-4 font-medium text-white flex items-center">
                  {art.type === 'dataset' ? <Database className="h-4 w-4 mr-2 text-blue-400" /> :
                   art.type === 'checkpoint' ? <Database className="h-4 w-4 mr-2 text-purple-400" /> :
                   <FileText className="h-4 w-4 mr-2 text-slate-400" />}
                  {art.name}
                </td>
                <td className="px-6 py-4 capitalize text-slate-300">{art.type}</td>
                <td className="px-6 py-4 text-slate-400 font-mono">{art.size}</td>
                <td className="px-6 py-4 text-slate-400">{new Date(art.created).toLocaleString()}</td>
                <td className="px-6 py-4 text-right">
                  <button className="text-cyan-400 hover:text-cyan-300 font-medium flex items-center justify-end ml-auto">
                    <Download className="h-4 w-4 mr-1" /> Download
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
