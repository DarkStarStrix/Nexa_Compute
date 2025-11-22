"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, List, Server, Database, CreditCard, FileText, Settings } from "lucide-react";
import clsx from "clsx";

const navigation = [
    { name: "Overview", href: "/dashboard", icon: LayoutDashboard },
    { name: "Jobs", href: "/dashboard/jobs", icon: List },
    { name: "Workers", href: "/dashboard/workers", icon: Server },
    { name: "Artifacts", href: "/dashboard/artifacts", icon: Database },
    { name: "Billing", href: "/dashboard/billing", icon: CreditCard },
    { name: "Settings", href: "/dashboard/settings", icon: Settings },
];

const externalLinks = [
    { name: "Docs", href: "http://localhost:3001", icon: FileText },
];

export function Sidebar() {
    const pathname = usePathname();

    return (
        <div className="flex h-full w-64 flex-col bg-slate-900 text-white">
            <div className="flex h-16 items-center px-6 font-bold text-xl tracking-wider border-b border-slate-800">
                <span className="text-cyan-400 mr-2">NEXA</span> FORGE
            </div>
            <nav className="flex-1 space-y-1 px-2 py-4">
                {navigation.map((item) => {
                    const isActive = pathname === item.href || pathname?.startsWith(item.href + "/");
                    return (
                        <Link
                            key={item.name}
                            href={item.href}
                            className={clsx(
                                isActive
                                    ? "bg-slate-800 text-cyan-400"
                                    : "text-slate-300 hover:bg-slate-800 hover:text-white",
                                "group flex items-center px-4 py-3 text-sm font-medium rounded-md transition-colors"
                            )}
                        >
                            <item.icon
                                className={clsx(
                                    isActive ? "text-cyan-400" : "text-slate-400 group-hover:text-white",
                                    "mr-3 h-5 w-5 flex-shrink-0"
                                )}
                                aria-hidden="true"
                            />
                            {item.name}
                        </Link>
                    );
                })}

                {/* External Links */}
                <div className="pt-2 border-t border-slate-800 mt-2">
                    {externalLinks.map((item) => (
                        <a
                            key={item.name}
                            href={item.href}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="group flex items-center px-4 py-3 text-sm font-medium rounded-md text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
                        >
                            <item.icon
                                className="text-slate-400 group-hover:text-white mr-3 h-5 w-5 flex-shrink-0"
                                aria-hidden="true"
                            />
                            {item.name}
                        </a>
                    ))}
                </div>
            </nav>
            <div className="border-t border-slate-800 p-4">
                <div className="flex items-center">
                    <div className="h-8 w-8 rounded-full bg-cyan-500 flex items-center justify-center text-xs font-bold text-white">
                        AL
                    </div>
                    <div className="ml-3">
                        <p className="text-sm font-medium text-white">Atheron Labs</p>
                        <p className="text-xs text-slate-400">Pro Plan</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
