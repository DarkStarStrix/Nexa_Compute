"use client";

import { Database, FileText, Download } from "lucide-react";
import { useEffect, useState } from "react";
import { getArtifacts, Artifact } from "@/lib/api";

export default function ArtifactsPage() {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>('all');

  useEffect(() => {
    async function fetchArtifacts() {
      try {
        const data = await getArtifacts(filter === 'all' ? undefined : filter);
        setArtifacts(data);
      } catch (error) {
        console.error("Failed to fetch artifacts:", error);
      } finally {
        setLoading(false);
      }
    }
    fetchArtifacts();
    const interval = setInterval(fetchArtifacts, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [filter]);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Artifacts</h1>
        <p className="mt-2 text-slate-400">Browse datasets, checkpoints, and reports.</p>
      </div>

      <div className="flex space-x-2 border-b border-slate-800 pb-4">
        {['all', 'dataset', 'checkpoint', 'eval'].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-md text-sm font-medium capitalize transition-colors ${
              filter === f
                ? "bg-slate-800 text-white"
                : "text-slate-400 hover:text-white hover:bg-slate-800/50"
            }`}
          >
            {f}
          </button>
        ))}
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
            {loading ? (
              <tr>
                <td colSpan={5} className="px-6 py-8 text-center text-slate-500">Loading artifacts...</td>
              </tr>
            ) : artifacts.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-6 py-12 text-center">
                  <Database className="mx-auto h-12 w-12 text-slate-600 mb-4" />
                  <h3 className="text-lg font-medium text-white mb-2">No Artifacts</h3>
                  <p className="text-slate-400">Artifacts will appear here once jobs complete.</p>
                </td>
              </tr>
            ) : (
              artifacts.map((art) => (
                <tr key={art.id} className="hover:bg-slate-800/50 transition-colors">
                  <td className="px-6 py-4 font-medium text-white flex items-center">
                    {art.type === 'dataset' ? <Database className="h-4 w-4 mr-2 text-blue-400" /> :
                     art.type === 'checkpoint' ? <Database className="h-4 w-4 mr-2 text-purple-400" /> :
                     <FileText className="h-4 w-4 mr-2 text-slate-400" />}
                    {art.name}
                  </td>
                  <td className="px-6 py-4 capitalize text-slate-300">{art.type}</td>
                  <td className="px-6 py-4 text-slate-400 font-mono">{art.size_human}</td>
                  <td className="px-6 py-4 text-slate-400">{new Date(art.created).toLocaleString()}</td>
                  <td className="px-6 py-4 text-right">
                    {art.uri && (
                      <a
                        href={art.uri}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-cyan-400 hover:text-cyan-300 font-medium flex items-center justify-end ml-auto"
                      >
                        <Download className="h-4 w-4 mr-1" /> Download
                      </a>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
