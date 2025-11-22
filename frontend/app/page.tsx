import Link from "next/link";
import { ArrowRight, Cpu, Shield, Zap } from "lucide-react";

export default function LandingPage() {
  console.log('LandingPage rendered');
  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))]" />
        <div className="relative pt-24 pb-16 sm:pt-32 sm:pb-24">
          <div className="mx-auto max-w-7xl px-6 lg:px-8">
            <div className="mx-auto max-w-2xl text-center">
              <h1 className="text-4xl font-bold tracking-tight sm:text-6xl bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent">
                Nexa Forge
              </h1>
              <p className="mt-6 text-lg leading-8 text-slate-300">
                The AI Foundry. Orchestrate data generation, distillation, training, and evaluation on ephemeral GPU compute.
              </p>
              <div className="mt-10 flex items-center justify-center gap-x-6">
                <Link
                  href="/dashboard"
                  className="rounded-md bg-cyan-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-cyan-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-600 flex items-center"
                >
                  Enter Forge <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
                <Link href="/docs" className="text-sm font-semibold leading-6 text-white">
                  Read Documentation <span aria-hidden="true">→</span>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="mx-auto max-w-7xl px-6 lg:px-8 py-24">
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-3">
          <div className="rounded-2xl bg-slate-900 p-8 border border-slate-800">
            <div className="mb-4 rounded-lg bg-cyan-900/30 p-3 w-fit">
              <Zap className="h-6 w-6 text-cyan-400" />
            </div>
            <h3 className="text-xl font-bold text-white">Ephemeral Compute</h3>
            <p className="mt-2 text-slate-400">
              Spin up GPUs on demand. Pay only for what you use. Auto-scaling and auto-termination built-in.
            </p>
          </div>
          <div className="rounded-2xl bg-slate-900 p-8 border border-slate-800">
            <div className="mb-4 rounded-lg bg-purple-900/30 p-3 w-fit">
              <Cpu className="h-6 w-6 text-purple-400" />
            </div>
            <h3 className="text-xl font-bold text-white">Distillation Pipeline</h3>
            <p className="mt-2 text-slate-400">
              Turn raw data into high-quality training sets using teacher models and automated feedback loops.
            </p>
          </div>
          <div className="rounded-2xl bg-slate-900 p-8 border border-slate-800">
            <div className="mb-4 rounded-lg bg-green-900/30 p-3 w-fit">
              <Shield className="h-6 w-6 text-green-400" />
            </div>
            <h3 className="text-xl font-bold text-white">Full Provenance</h3>
            <p className="mt-2 text-slate-400">
              Every artifact is tracked. Complete lineage from dataset to trained model to evaluation report.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
