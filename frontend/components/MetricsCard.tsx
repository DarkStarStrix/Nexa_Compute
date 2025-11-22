import { LucideIcon } from "lucide-react";

interface MetricsCardProps {
    title: string;
    value: string | number;
    icon: LucideIcon;
    trend?: string;
    trendUp?: boolean;
}

export function MetricsCard({ title, value, icon: Icon, trend, trendUp }: MetricsCardProps) {
    return (
        <div className="rounded-lg bg-slate-900 p-6 shadow-lg border border-slate-800">
            <div className="flex items-center justify-between">
                <div>
                    <p className="text-sm font-medium text-slate-400">{title}</p>
                    <p className="mt-2 text-3xl font-bold text-white">{value}</p>
                </div>
                <div className="rounded-full bg-slate-800 p-3">
                    <Icon className="h-6 w-6 text-cyan-400" />
                </div>
            </div>
            {trend && (
                <div className="mt-4 flex items-center text-sm">
                    <span className={trendUp ? "text-green-400" : "text-red-400"}>
                        {trend}
                    </span>
                    <span className="ml-2 text-slate-500">from last month</span>
                </div>
            )}
        </div>
    );
}
