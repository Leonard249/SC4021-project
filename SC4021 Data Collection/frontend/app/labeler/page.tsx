'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getLabelers } from '@/app/lib/api';

export default function LabelerSelectionPage() {
    const [labelers, setLabelers] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const router = useRouter();

    useEffect(() => {
        // Check auth
        if (typeof window !== 'undefined') {
            const secret = localStorage.getItem('auth_secret');
            if (!secret) {
                router.push('/');
                return;
            }
        }

        async function fetchLabelers() {
            try {
                const data = await getLabelers();
                setLabelers(data);
            } catch (err) {
                setError('Failed to load labelers. Please try again later.');
            } finally {
                setLoading(false);
            }
        }

        fetchLabelers();
    }, [router]);

    const handleSelectLabeler = (name: string) => {
        if (typeof window !== 'undefined') {
            localStorage.setItem('labeler_name', name);
        }
        router.push('/label');
    };

    if (loading) {
        return (
            <main className="flex min-h-screen items-center justify-center bg-gradient-to-br from-indigo-900 via-purple-900 to-slate-900 p-4">
                <div className="flex animate-pulse flex-col items-center gap-4">
                    <div className="h-12 w-12 rounded-full border-4 border-purple-500 border-t-transparent animate-spin"></div>
                    <p className="text-purple-200">Loading labelers...</p>
                </div>
            </main>
        );
    }

    return (
        <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-indigo-900 via-purple-900 to-slate-900 p-4">
            <div className="w-full max-w-md rounded-2xl bg-white/10 p-8 backdrop-blur-lg shadow-2xl border border-white/20">
                <div className="text-center mb-8">
                    <h1 className="text-3xl font-extrabold tracking-tight text-white mb-2">
                        Who are you?
                    </h1>
                    <p className="text-sm text-purple-200">
                        Select your profile to start labeling
                    </p>
                </div>

                {error ? (
                    <div className="rounded-lg bg-red-500/20 p-4 text-sm text-red-200 border border-red-500/50 text-center">
                        {error}
                        <button
                            onClick={() => window.location.reload()}
                            className="mt-4 block w-full rounded bg-red-500/40 px-4 py-2 hover:bg-red-500/60 transition-colors"
                        >
                            Retry
                        </button>
                    </div>
                ) : (
                    <div className="grid gap-4">
                        {labelers.map((name) => (
                            <button
                                key={name}
                                onClick={() => handleSelectLabeler(name)}
                                className="group relative flex items-center justify-between rounded-xl border border-white/10 bg-black/20 p-4 text-left transition-all hover:border-purple-500/50 hover:bg-purple-900/40 active:scale-[0.98]"
                            >
                                <div className="flex items-center gap-4">
                                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 text-lg font-bold text-white shadow-inner">
                                        {name.charAt(0).toUpperCase()}
                                    </div>
                                    <span className="text-lg font-medium text-white capitalize">
                                        {name}
                                    </span>
                                </div>
                                <svg
                                    className="h-5 w-5 text-purple-400 opacity-0 transition-all group-hover:translate-x-1 group-hover:opacity-100"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                </svg>
                            </button>
                        ))}
                    </div>
                )}
            </div>
        </main>
    );
}
